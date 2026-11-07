import ctypes
import gc
import importlib.util
import weakref

import torch

from . import _compat

PAGE = 2 * 1024**2  # the caching allocator requests segments in multiples of 2 MiB

_lib = None
_allocator = None
_regions = weakref.WeakValueDictionary()  # address -> _Region
_active = {}  # device index -> Loan
_dropped = []  # (device index, pool) of regions dropped during a loan on their device, released after it


def _load():
    global _lib, _allocator
    if _lib is None:
        spec = importlib.util.find_spec("memlend._allocator")
        if spec is None:
            raise ImportError("memlend's allocator library is missing: install memlend with pip so that it gets compiled")
        lib = ctypes.CDLL(spec.origin)
        if lib.memlend_init() != 0:
            raise RuntimeError("memlend could not load the CUDA driver (libcuda.so.1)")
        p = ctypes.c_void_p
        lib.memlend_create.argtypes, lib.memlend_create.restype = [ctypes.c_int, ctypes.c_size_t], p
        lib.memlend_destroy.argtypes = [p]
        lib.memlend_base.argtypes, lib.memlend_base.restype = [p], ctypes.c_uint64
        lib.memlend_begin.argtypes, lib.memlend_begin.restype = [p, p], ctypes.c_int
        lib.memlend_end.argtypes, lib.memlend_end.restype = [p, p, ctypes.c_int], ctypes.c_int64
        lib.memlend_extra_peak.argtypes, lib.memlend_extra_peak.restype = [p], ctypes.c_int64
        _allocator = _compat.pluggable_allocator(spec.origin)
        _lib = lib
    return _lib


class _Region:
    """The memory behind a tensor made by `empty`, exposed through `__cuda_array_interface__`.

    memlend maps it page by page itself, so that it can map the same pages into the borrower's segments. It lives as long
    as a tensor view of it. The borrower's segments live in `pool` from one loan to the next.
    """

    def __init__(self, nbytes, device):
        lib = _load()
        self._destroy = lib.memlend_destroy  # module globals may be gone at interpreter exit
        self.handle = lib.memlend_create(device.index, nbytes)
        if not self.handle:
            raise torch.OutOfMemoryError(f"memlend could not allocate {nbytes} bytes on {device}")
        self.device, self.npages, self.base = device, nbytes // PAGE, lib.memlend_base(self.handle)
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (self.base, False), "version": 3}
        self.pool = _compat.new_pool(_allocator, device)

    def __del__(self):
        if self.handle:
            if _active and self.device.index in _active:  # torch < 2.13 aborts if it is released during a loan
                _dropped.append((self.device.index, self.pool))
            del self.pool  # releases the borrower's segments
            self._destroy(self.handle)  # the memory goes with the last segment using it


def empty(*size, dtype=None, device=None) -> torch.Tensor:
    """Like `torch.empty`, but its memory can be lent with `lend`.

    Args:
        *size (`int` or `tuple` of `int`):
            Shape of the tensor.
        dtype (`torch.dtype`, *optional*):
            Defaults to `torch.get_default_dtype()`.
        device (`torch.device`, `str` or `int`, *optional*):
            CUDA device, defaults to the current one.
    """
    meta = torch.empty(*size, dtype=dtype, device="meta")
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device or "cuda")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _compat.check_supported(device)
    nbytes = meta.numel() * meta.element_size()
    region = _Region(max(1, -(-nbytes // PAGE)) * PAGE, device)
    _regions[region.base] = region
    return torch.as_tensor(region, device=device)[:nbytes].view(meta.dtype).view(meta.shape)


def lend(tensor: torch.Tensor, keep=None) -> "Loan":
    """Context manager: inside it, every CUDA allocation on the tensor's device may use the tensor's memory.

    When it exits, what `keep` covers is back, bit-identical, and the rest of the tensor may have changed. The borrower
    gets the tensor's memory that is not kept first, then free GPU memory, then kept memory, which is copied to the CPU
    and back. The owner must not use the tensor inside, and
    everything allocated inside must be freed by the end (else `LeakError`).

    Args:
        tensor (`torch.Tensor`):
            A tensor made by `empty`, or a view of one: its whole memory is lent.
        keep (`torch.Tensor` or `list` of `torch.Tensor`, *optional*):
            Views of `tensor` to get back, such as `tensor[3]` or `tensor[:, 2:5]`. Defaults to all of it; `[]` gets
            nothing back and copies nothing.
    """
    return Loan(tensor, keep)


class LeakError(RuntimeError):
    """Allocations made during a loan are still alive when it ends."""


class Loan:
    """A loan made by `lend`. After it:

    - `copied_bytes`: kept memory the borrower used, copied to the CPU and back.
    - `extra_bytes`: peak memory the borrower used beyond the tensor.
    """

    def __init__(self, tensor, keep):
        self.region = _regions.get(tensor.untyped_storage().data_ptr()) if tensor.is_cuda else None
        if self.region is None:
            raise ValueError("lend needs a tensor made by memlend.empty, or a view of one")
        self.keep = _kept_pages(self.region, keep)
        self.copied_bytes = self.extra_bytes = None

    def __enter__(self):
        region, device = self.region, self.region.device
        if device.index in _active:
            raise RuntimeError(f"a loan is already active on {device}; loans on one device cannot be nested")
        torch.cuda.synchronize(device)  # the owner's last kernels may still use the memory
        keep = self.keep.to(torch.uint8)  # alive during the call
        if _lib.memlend_begin(region.handle, keep.data_ptr()) != 0:
            # The memory the borrower had beyond the tensor is gone, or pinned CPU memory ran out: start from no segments.
            region.pool = _compat.new_pool(_allocator, device)
            if _lib.memlend_begin(region.handle, keep.data_ptr()) != 0:
                raise torch.OutOfMemoryError("memlend could not allocate pinned CPU memory to copy the kept data")
        _compat.route_everything_to(region.pool, device)
        _active[device.index] = self
        return self

    def __exit__(self, exc_type, exc, tb):
        region, device = self.region, self.region.device
        torch.cuda.synchronize(device)  # the borrower's kernels may still use the memory
        _compat.stop_routing(region.pool, device)
        _compat.clear_cublas_workspaces()
        live = _compat.live_blocks(region.pool)
        if live:  # garbage cycles may still hold borrower tensors
            gc.collect()
            live = _compat.live_blocks(region.pool)
        del _active[device.index]
        _dropped[:] = [item for item in _dropped if item[0] != device.index]
        # Restores the kept data and, unless allocations leaked into it, gives back the memory taken beyond the tensor.
        self.copied_bytes = _lib.memlend_end(region.handle, torch.cuda.current_stream(device).cuda_stream, not live)
        self.extra_bytes = _lib.memlend_extra_peak(region.handle)
        self.region = None  # a finished loan does not keep the tensor alive
        if live and exc_type is None:
            raise LeakError(_leak_message(live))
        return False


def _kept_pages(region, keep):
    if keep is None:
        return torch.ones(region.npages, dtype=torch.bool)
    edges = torch.zeros(region.npages + 1, dtype=torch.int64)  # +1 where a kept range starts, -1 after it ends
    for view in [keep] if isinstance(keep, torch.Tensor) else keep:
        if view.untyped_storage().data_ptr() != region.base:
            raise ValueError(
                "keep must hold views of the lent tensor, such as tensor[3] or tensor[:, 2:5] (indexing with a list "
                "or a mask makes a copy)"
            )
        if view.numel():
            starts, length = _byte_runs(view)
            edges.index_add_(0, starts // PAGE, torch.ones_like(starts))
            edges.index_add_(0, (starts + length - 1) // PAGE + 1, -torch.ones_like(starts))
    return edges.cumsum(0)[:-1] > 0


def _byte_runs(view):
    """Offsets in its storage of the contiguous runs of `view`, and their length, in bytes."""
    shape, stride, run = list(view.shape), list(view.stride()), 1
    while shape and (shape[-1] == 1 or stride[-1] == run):
        run *= shape.pop()
        stride.pop()
    starts = torch.tensor([view.storage_offset()])
    for size, step in zip(shape, stride):
        starts = (starts[:, None] + torch.arange(size) * step).flatten()
    return starts * view.element_size(), run * view.element_size()


def _leak_message(live):
    lines = [
        f"{len(live)} allocation(s) made during the loan are still alive ({sum(b['size'] for b in live)} bytes). Free "
        "them before the loan ends (`del` them, or run the borrower in a function), and create long-lived state before "
        "lending."
    ]
    for block in live[:10]:
        lines.append(f"  - {block['requested_size']} bytes")
        lines += [f"      {f['filename']}:{f['line']} {f['name']}" for f in block.get("frames", []) if f["filename"].endswith(".py")][:6]
    if not any(block.get("frames") for block in live):
        lines.append("Call torch.cuda.memory._record_memory_history() before the loan to see where they were allocated.")
    return "\n".join(lines)
