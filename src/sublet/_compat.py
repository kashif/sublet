"""The private PyTorch allocator internals sublet relies on, behind a version check.

They change between releases, so sublet refuses torch versions its test suite has not passed on.
`SUBLET_SKIP_TORCH_CHECK=1` bypasses the check.
"""

import os

import torch

# Before 2.8 a pool's custom allocator only applies on threads with a MemPoolContext: autograd's backward thread bypasses it.
TESTED = ((2, 8), (2, 15))


def torch_version():
    return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def check_supported(device):
    if not os.environ.get("SUBLET_SKIP_TORCH_CHECK") and not TESTED[0] <= torch_version() <= TESTED[1]:
        raise RuntimeError(
            f"sublet is tested with torch {'.'.join(map(str, TESTED[0]))} to {'.'.join(map(str, TESTED[1]))}, found "
            f"{torch.__version__}. Set SUBLET_SKIP_TORCH_CHECK=1 to try anyway, after running its test suite."
        )
    backend = torch.cuda.get_allocator_backend()
    if backend != "native":
        raise RuntimeError(f"sublet needs PyTorch's native CUDA caching allocator, found backend {backend!r}")
    with torch.cuda.device(device):
        torch.empty(0, device=device)  # creates the device's primary context


def pluggable_allocator(path):
    return torch.cuda.memory.CUDAPluggableAllocator(path, "sublet_malloc", "sublet_free").allocator()


def new_pool(allocator, device):
    with torch.cuda.device(device):
        return torch.cuda.MemPool(allocator)


def route_everything_to(pool, device):
    # Every thread and stream: torch.cuda.use_mem_pool only routes the calling thread, missing autograd's backward.
    torch._C._cuda_beginAllocateToPool(device.index, pool.id)


def stop_routing(pool, device):
    torch._C._cuda_endAllocateToPool(device.index, pool.id)
    torch._C._cuda_releasePool(device.index, pool.id)


def clear_cublas_workspaces():
    # cuBLAS keeps one workspace per (handle, stream) for the life of the process; drop those made during the loan.
    torch._C._cuda_clearCublasWorkspaces()


def segments(pool):
    return pool.snapshot(include_traces=False) if torch_version() >= (2, 11) else pool.snapshot()


def live_blocks(pool, in_segment):
    """Blocks of `pool` still allocated, in the segments whose address satisfies `in_segment`."""
    return [b for s in segments(pool) if in_segment(s["address"]) for b in s["blocks"] if b["state"] == "active_allocated"]
