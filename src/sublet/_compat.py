"""What sublet needs from PyTorch, behind a version check.

sublet points the cudaMalloc and cudaFree calls of PyTorch's caching allocator (in libc10_cuda.so) at its own, see
_allocator.c. It refuses PyTorch versions its test suite has not passed on; `SUBLET_SKIP_TORCH_CHECK=1` bypasses the
check.
"""

import os

import torch

TESTED = ((2, 6), (2, 16))


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


def check_allocator_settings():
    # With expandable segments, the caching allocator maps memory itself instead of calling cudaMalloc
    if torch_version() >= (2, 12):
        settings = torch._C._accelerator_getAllocatorSettings()
    else:
        settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + "," + os.environ.get("PYTORCH_ALLOC_CONF", "")
    if "expandable_segments:true" in settings.replace(" ", "").lower():
        raise RuntimeError("sublet needs PyTorch's caching allocator without expandable_segments")


def clear_cublas_workspaces():
    # cuBLAS keeps one workspace per (handle, stream) for the life of the process; drop those made during the loan.
    torch._C._cuda_clearCublasWorkspaces()
