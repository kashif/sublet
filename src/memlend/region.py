import ctypes
import gc
import importlib.util

import torch

from . import _compat

PAGE = 2 * 1024**2  # the caching allocator requests segments in multiples of 2 MiB

_lib = None
_allocator = None
_active = {}  # device index -> Loan


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
        lib.memlend_create.argtypes, lib.memlend_create.restype = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int64], p
        lib.memlend_destroy.argtypes = [p]
        lib.memlend_base.argtypes, lib.memlend_base.restype = [p], ctypes.c_uint64
        lib.memlend_begin.argtypes = [p, p, p]
        lib.memlend_end.argtypes = [p]
        lib.memlend_touched.argtypes = [p, p]
        lib.memlend_fallback_peak.argtypes, lib.memlend_fallback_peak.restype = [ctypes.c_int], ctypes.c_int64
        lib.memlend_lends.argtypes, lib.memlend_lends.restype = [p, ctypes.c_uint64], ctypes.c_int
        _allocator = _compat.pluggable_allocator(spec.origin)
        _lib = lib
    return _lib


class _DeviceMemory:
    """The region's memory exposed through `__cuda_array_interface__`, freed when the last tensor view dies.

    memlend allocates it page by page itself, so that it can map the same pages into the borrower's segments.
    """

    def __init__(self, handle, nbytes):
        self.handle = handle
        self._destroy = _lib.memlend_destroy  # module globals may be gone at interpreter exit
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (_lib.memlend_base(handle), False), "version": 3}

    def __del__(self):
        self._destroy(self.handle)


class LeakError(RuntimeError):
    """Allocations made during a loan are still alive in lent memory when the loan ends."""


class LendableRegion:
    """GPU memory owned by one party that can be lent to arbitrary PyTorch code.

    Args:
        nbytes (`int`):
            Size of the region, a multiple of `block_size`.
        block_size (`int`, *optional*, defaults to `64 * 1024**2`):
            Unit of `empty`, `Loan.touched_blocks` and the backup helpers, a multiple of 2 MiB.
        device (`torch.device` or `int`, *optional*):
            CUDA device, defaults to the current one.
    """

    def __init__(self, nbytes: int, block_size: int = 64 * 1024**2, device=None):
        if block_size % PAGE or nbytes % block_size:
            raise ValueError(f"block_size must be a multiple of {PAGE} and nbytes a multiple of block_size")
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device or "cuda")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        _compat.check_supported(device)
        lib = _load()
        self.device, self.nbytes, self.block_size, self.num_blocks = device, nbytes, block_size, nbytes // block_size
        self._handle = lib.memlend_create(device.index, nbytes, block_size)
        if not self._handle:
            raise torch.OutOfMemoryError(f"memlend could not allocate a {nbytes} byte region on {device}")
        self.tensor = torch.as_tensor(_DeviceMemory(self._handle, nbytes), device=device)

    def lend(self, empty=(), backup: torch.Tensor | None = None) -> "Loan":
        """Context manager: inside it, every CUDA allocation on this device may be served from the region.

        Each segment the caching allocator asks for gets its own addresses, backed by unlent pages of the region wherever
        they are: first pages of empty blocks and of blocks already lent, then pages of blocks holding data, then new
        device memory once the region is used up.

        Args:
            empty (iterable of `int`, *optional*):
                Blocks that hold nothing the owner needs: free to lend, never backed up.
            backup (`torch.Tensor`, *optional*):
                Pinned CPU `uint8` tensor of `nbytes`. Each other block is copied into it right before it is first lent,
                ordered on the stream that asked for the memory. Restore with `restore(backup, loan.dirty_blocks)`.
        """
        return Loan(self, empty, backup)

    def save(self, backup: torch.Tensor, blocks=None):
        """Copy `blocks` (default: all) into `backup` (CPU `uint8` tensor of `nbytes`), on the current stream."""
        for a, b in self._runs(blocks):
            backup[a:b].copy_(self.tensor[a:b], non_blocking=True)

    def restore(self, backup: torch.Tensor, blocks=None):
        """Copy `blocks` (default: all) from `backup` into the region, on the current stream."""
        for a, b in self._runs(blocks):
            self.tensor[a:b].copy_(backup[a:b], non_blocking=True)

    def _lends(self, ptr):
        """Whether address `ptr` is in a page of the region lent to a segment."""
        return bool(_lib.memlend_lends(self._handle, ptr))

    def _runs(self, blocks):
        blocks = sorted(set(range(self.num_blocks) if blocks is None else map(int, blocks)))
        i = 0
        while i < len(blocks):
            j = i
            while j + 1 < len(blocks) and blocks[j + 1] == blocks[j] + 1:
                j += 1
            yield blocks[i] * self.block_size, (blocks[j] + 1) * self.block_size
            i = j + 1


class Loan:
    """One lending of a `LendableRegion`. After the `with` block:

    - `touched_blocks`: blocks handed out; every other block is bit-identical to before the loan.
    - `dirty_blocks`: touched blocks not declared empty, the ones to restore.
    - `fallback_peak_bytes`: peak memory served from outside the region.
    """

    def __init__(self, region, empty, backup):
        self.region, self.backup = region, backup
        self.empty = set(map(int, empty))
        if not all(0 <= b < region.num_blocks for b in self.empty):
            raise ValueError(f"empty blocks must be indices in [0, {region.num_blocks})")
        self.touched_blocks = self.dirty_blocks = self.fallback_peak_bytes = None

    def __enter__(self):
        region, device = self.region, self.region.device
        if device.index in _active:
            raise RuntimeError(f"a loan is already active on {device}; loans on one device cannot be nested")
        torch.cuda.synchronize(device)  # the owner's last kernels may still use the region
        skip = (ctypes.c_uint8 * region.num_blocks)(*(b in self.empty for b in range(region.num_blocks)))
        _lib.memlend_begin(region._handle, self.backup.data_ptr() if self.backup is not None else None, skip)
        self._pool = _compat.new_pool(_allocator, device)
        _compat.route_everything_to(self._pool, device)
        _active[device.index] = self
        return self

    def __exit__(self, exc_type, exc, tb):
        region, device = self.region, self.region.device
        torch.cuda.synchronize(device)  # the borrower's kernels may still use the region
        _compat.stop_routing(self._pool, device)
        _compat.clear_cublas_workspaces()
        live = _compat.live_blocks(self._pool, region._lends)
        if live:  # garbage cycles may still hold borrower tensors
            gc.collect()
            live = _compat.live_blocks(self._pool, region._lends)
        del self._pool  # last reference: its free segments go back to the allocator
        _lib.memlend_end(region._handle)
        del _active[device.index]
        touched = (ctypes.c_uint8 * region.num_blocks)()
        _lib.memlend_touched(region._handle, touched)
        self.touched_blocks = [i for i, t in enumerate(touched) if t]
        self.dirty_blocks = [b for b in self.touched_blocks if b not in self.empty]
        self.fallback_peak_bytes = _lib.memlend_fallback_peak(device.index)
        if live and exc_type is None:
            raise LeakError(_leak_message(live))
        return False


def _leak_message(live):
    lines = [
        f"{len(live)} allocation(s) made during the loan are still alive in lent memory ({sum(b['size'] for b in live)} "
        "bytes). Free them before the loan ends (`del` them, or run the borrower in a function), and create "
        "long-lived state before lending."
    ]
    for block in live[:10]:
        lines.append(f"  - {block['requested_size']} bytes")
        lines += [f"      {f['filename']}:{f['line']} {f['name']}" for f in block.get("frames", []) if f["filename"].endswith(".py")][:6]
    if not any(block.get("frames") for block in live):
        lines.append("Call torch.cuda.memory._record_memory_history() before the loan to see where they were allocated.")
    return "\n".join(lines)
