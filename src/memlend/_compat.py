"""Every PyTorch internal memlend relies on, behind a version check.

memlend steers PyTorch's CUDA caching allocator through private functions (`torch._C._cuda_*`). Their behavior is
verified by the test suite on the versions listed in `TESTED`, and nowhere else: on any other version the library
refuses to run instead of failing silently. Set `MEMLEND_SKIP_TORCH_CHECK=1` to try an untested version anyway.
"""

import os

import torch

# Inclusive range of torch minor versions the test suite passed on (see DESIGN.md). Before 2.8 a pool's custom allocator
# is only used on threads where a thread-local MemPoolContext is set, so autograd's backward thread would bypass it.
TESTED = ((2, 8), (2, 15))


def torch_version():
    return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def check_supported(device):
    """Raise a clear error if this torch version or allocator configuration is not supported."""
    if not os.environ.get("MEMLEND_SKIP_TORCH_CHECK"):
        version = torch_version()
        if not TESTED[0] <= version <= TESTED[1]:
            (lo_major, lo_minor), (hi_major, hi_minor) = TESTED
            raise RuntimeError(
                f"memlend is tested with torch {lo_major}.{lo_minor} to {hi_major}.{hi_minor}, found {torch.__version__}. "
                "It relies on private allocator internals that change between releases. "
                "Set MEMLEND_SKIP_TORCH_CHECK=1 to try anyway, and run its test suite first."
            )
    backend = torch.cuda.get_allocator_backend()
    if backend != "native":
        raise RuntimeError(
            f"memlend needs PyTorch's native CUDA caching allocator, but the backend is {backend!r} "
            "(PYTORCH_CUDA_ALLOC_CONF=backend:...). Memory pools with a custom allocator only exist in the native one."
        )
    torch.cuda.init()
    with torch.cuda.device(device):
        torch.empty(0, device=device)  # creates the device's primary context if PyTorch has not yet


def pluggable_allocator(path):
    allocator = torch.cuda.memory.CUDAPluggableAllocator(path, "memlend_malloc", "memlend_free")
    return allocator.allocator()


def new_pool(allocator, device):
    with torch.cuda.device(device):
        return torch.cuda.MemPool(allocator)


def route_everything_to(pool, device):
    """Send every allocation on `device`, from every thread and stream, to `pool`.

    `torch.cuda.use_mem_pool` only routes the calling thread, which misses autograd's backward thread.
    """
    torch._C._cuda_beginAllocateToPool(device.index, pool.id)


def stop_routing(pool, device):
    torch._C._cuda_endAllocateToPool(device.index, pool.id)
    torch._C._cuda_releasePool(device.index, pool.id)


def clear_cublas_workspaces():
    """cuBLAS keeps one workspace per (handle, stream) for the life of the process; drop them all."""
    torch._C._cuda_clearCublasWorkspaces()


def live_blocks(pool, lo, hi):
    """Blocks of `pool` still allocated whose address is in [lo, hi)."""
    # Traces (memory history) are not needed here and can be large; the argument exists from 2.11.
    segments = pool.snapshot(include_traces=False) if torch_version() >= (2, 11) else pool.snapshot()
    return [
        block
        for segment in segments
        if lo <= segment["address"] < hi
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    ]
