import ctypes
import gc
import importlib.util

import torch

from . import _compat

PAGE = 2 << 20  # the caching allocator requests segments in multiples of 2 MiB

_lib = None
_allocator = None
_active = {}  # device index -> Loan


def _load():
    """Load the compiled allocator and register it with PyTorch (once per process)."""
    global _lib, _allocator
    if _lib is not None:
        return _lib
    spec = importlib.util.find_spec("memlend._allocator")
    if spec is None:
        raise ImportError("memlend's allocator library is missing: install memlend with pip so that it gets compiled")
    lib = ctypes.CDLL(spec.origin)
    if lib.memlend_init() != 0:
        raise RuntimeError("memlend could not load the CUDA driver (libcuda.so.1)")
    p, i64 = ctypes.c_void_p, ctypes.c_int64
    lib.memlend_create.argtypes, lib.memlend_create.restype = [ctypes.c_int, ctypes.c_size_t, i64], p
    lib.memlend_destroy.argtypes = [p]
    lib.memlend_base.argtypes, lib.memlend_base.restype = [p], ctypes.c_uint64
    lib.memlend_set_order.argtypes = [p, p]
    lib.memlend_begin.argtypes = [p, p, p, ctypes.c_int, i64]
    lib.memlend_end.argtypes = [p]
    lib.memlend_touched.argtypes = [p, p]
    lib.memlend_saved.argtypes = [p, p]
    lib.memlend_fallback_peak.argtypes, lib.memlend_fallback_peak.restype = [ctypes.c_int], i64
    _allocator = _compat.pluggable_allocator(spec.origin)
    _lib = lib
    return lib


class _DeviceMemory:
    """Raw cuMemAlloc'd memory exposed through `__cuda_array_interface__`; freed when the last tensor view dies.

    The region is allocated outside PyTorch's caching allocator: the borrower's segments are sub-ranges of it, and the
    caching allocator indexes allocations by pointer, so the region's own pointer must not be one of its allocations.
    """

    def __init__(self, handle, nbytes):
        self.handle = handle
        self._destroy = _lib.memlend_destroy  # module globals may already be gone at interpreter exit
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (_lib.memlend_base(handle), False),
            "version": 3,
        }

    def __del__(self):
        self._destroy(self.handle)


class LeakError(RuntimeError):
    """Allocations made during a loan are still alive in lent memory when the loan ends."""


class LendableRegion:
    """GPU memory owned by one party that can be lent to arbitrary PyTorch code.

    Args:
        nbytes (`int`):
            Size of the region, a multiple of `block_size`.
        block_size (`int`, *optional*, defaults to `64 << 20`):
            Granularity of `set_lend_order`, `Loan.touched_blocks` and the backup helpers, a multiple of 2 MiB.
        device (`torch.device` or `int`, *optional*):
            CUDA device, defaults to the current one.
    """

    def __init__(self, nbytes: int, block_size: int = 64 << 20, device=None):
        if block_size % PAGE or nbytes % block_size:
            raise ValueError(f"block_size must be a multiple of {PAGE} and nbytes a multiple of block_size")
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device or "cuda")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        _compat.check_supported(device)
        lib = _load()
        self.device = device
        self.nbytes = nbytes
        self.block_size = block_size
        self.num_blocks = nbytes // block_size
        handle = lib.memlend_create(device.index, nbytes, block_size)
        if not handle:
            raise torch.OutOfMemoryError(f"memlend could not allocate a {nbytes} byte region on {device}")
        self._handle = handle
        self.tensor = torch.as_tensor(_DeviceMemory(handle, nbytes), device=device)
        self._order = list(reversed(range(self.num_blocks)))

    def set_lend_order(self, blocks):
        """Hand out `blocks` first, in this order, then the remaining blocks from the far end of the region."""
        blocks = [int(b) for b in blocks]
        listed = set(blocks)
        order = blocks + [b for b in reversed(range(self.num_blocks)) if b not in listed]
        if len(order) != self.num_blocks or len(listed) != len(blocks):
            raise ValueError("lend order must list distinct block indices of this region")
        self._order = order

    def lend(self, empty=(), backup: torch.Tensor | None = None, _prefetch: int = 0, _sync_copies: bool = False):
        """Context manager: inside it, CUDA allocations on this device may be served from the region.

        Args:
            empty (iterable of `int`, *optional*):
                Blocks that hold nothing the owner needs. They are handed out first and never backed up, so a borrower
                that fits in them moves zero bytes.
            backup (`torch.Tensor`, *optional*):
                Pinned CPU `uint8` tensor of `nbytes`. When given, every other block is copied into it right before it
                is first handed out (lazy backup); the copy is ordered on the stream that asked for the memory, so the
                host never waits. Restore with `region.restore(backup, loan.dirty_blocks)`.

        `_prefetch` (keep that many blocks copied ahead of demand) and `_sync_copies` (the host waits for each copy instead
        of the borrower's stream) are the lazy-backup variants compared in BENCHMARKS.md; neither helps, they exist so
        the comparison can be rerun.
        """
        return Loan(self, empty, backup, _prefetch, _sync_copies)

    def save(self, backup: torch.Tensor, blocks=None):
        """Copy `blocks` (default: all) of the region into `backup` (CPU `uint8` tensor of `nbytes`)."""
        for a, b in self._runs(blocks):
            backup[a:b].copy_(self.tensor[a:b], non_blocking=True)

    def restore(self, backup: torch.Tensor, blocks=None):
        """Copy `blocks` (default: all) from `backup` back into the region.

        Copies of both helpers are issued on the current stream, one per contiguous run of blocks; synchronize before
        relying on them from the host or another stream.
        """
        for a, b in self._runs(blocks):
            self.tensor[a:b].copy_(backup[a:b], non_blocking=True)

    def _runs(self, blocks):
        """Byte ranges of the contiguous runs in `blocks`."""
        blocks = sorted(set(range(self.num_blocks) if blocks is None else (int(b) for b in blocks)))
        i = 0
        while i < len(blocks):
            j = i
            while j + 1 < len(blocks) and blocks[j + 1] == blocks[j] + 1:
                j += 1
            yield blocks[i] * self.block_size, (blocks[j] + 1) * self.block_size
            i = j + 1


class Loan:
    """One lending of a `LendableRegion`; use through `LendableRegion.lend()`.

    After the `with` block:
        touched_blocks (`list[int]`): blocks whose memory was handed out during the loan. Every other block is
            bit-identical to its content before the loan.
        dirty_blocks (`list[int]`): touched blocks not declared `empty`, the ones the owner has to restore.
        backed_up_blocks (`list[int]`): blocks copied into the lazy backup (dirty blocks plus any prefetched ones).
        fallback_peak_bytes (`int`): peak memory served from outside the region (ordinary free GPU memory).
    """

    def __init__(self, region, empty, backup, prefetch, sync_copies):
        self.region = region
        self.empty = list(dict.fromkeys(int(b) for b in empty))
        if not all(0 <= b < region.num_blocks for b in self.empty):
            raise ValueError(f"empty blocks must be indices in [0, {region.num_blocks})")
        self.backup = backup
        self.prefetch = prefetch
        self.sync_copies = sync_copies
        self.touched_blocks = None
        self.dirty_blocks = None
        self.backed_up_blocks = None
        self.fallback_peak_bytes = None

    def __enter__(self):
        region = self.region
        device = region.device
        if device.index in _active:
            raise RuntimeError(f"a loan is already active on {device}; loans on one device cannot be nested")
        # The owner's in-flight kernels may still read or write the region.
        torch.cuda.synchronize(device)
        empty = set(self.empty)
        order = self.empty + [b for b in region._order if b not in empty]
        _lib.memlend_set_order(region._handle, (ctypes.c_int64 * region.num_blocks)(*order))
        skip = (ctypes.c_uint8 * region.num_blocks)(*(b in empty for b in range(region.num_blocks)))
        backup = self.backup.data_ptr() if self.backup is not None else None
        _lib.memlend_begin(region._handle, backup, skip, self.sync_copies, self.prefetch)
        self._pool = _compat.new_pool(_allocator, device)
        _compat.route_everything_to(self._pool, device)
        _active[device.index] = self
        return self

    def __exit__(self, exc_type, exc, tb):
        region = self.region
        device = region.device
        torch.cuda.synchronize(device)
        _compat.stop_routing(self._pool, device)
        _compat.clear_cublas_workspaces()
        base = region.tensor.data_ptr()
        live = _compat.live_blocks(self._pool, base, base + region.nbytes)
        if live:  # garbage reference cycles may still hold borrower tensors
            gc.collect()
            live = _compat.live_blocks(self._pool, base, base + region.nbytes)
        # Last reference to the pool: it becomes freeable and its cached segments go back to memlend_free.
        del self._pool
        _lib.memlend_end(region._handle)
        del _active[device.index]
        touched, saved = (ctypes.c_uint8 * region.num_blocks)(), (ctypes.c_uint8 * region.num_blocks)()
        _lib.memlend_touched(region._handle, touched)
        _lib.memlend_saved(region._handle, saved)
        self.touched_blocks = [i for i, t in enumerate(touched) if t]
        self.backed_up_blocks = [i for i, t in enumerate(saved) if t]
        empty = set(self.empty)
        self.dirty_blocks = [b for b in self.touched_blocks if b not in empty]
        self.fallback_peak_bytes = _lib.memlend_fallback_peak(device.index)
        if live and exc_type is None:
            raise LeakError(_leak_message(live, base))
        return False


def _leak_message(live, base):
    lines = [
        f"{len(live)} allocation(s) made during the loan are still alive in lent memory "
        f"({sum(b['size'] for b in live)} bytes). The owner's data there is overwritten and the memory stays "
        "unavailable until they are freed. Free them before the loan ends (`del` the variables, run the borrower in a "
        "function, `zero_grad(set_to_none=True)`), or create long-lived state such as optimizer states before lending."
    ]
    for block in live[:10]:
        lines.append(f"  - {block['requested_size']} bytes at region offset {block['address'] - base}")
        python_frames = [f for f in block.get("frames", []) if f["filename"].endswith(".py")]
        for frame in python_frames[:6]:
            lines.append(f"      {frame['filename']}:{frame['line']} {frame['name']}")
    if not any(block.get("frames") for block in live):
        lines.append("Call torch.cuda.memory._record_memory_history() before the loan to see where they were allocated.")
    return "\n".join(lines)
