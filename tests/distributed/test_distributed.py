import os
import subprocess
import sys

import pytest
import torch

import memlend

MiB = 1024**2
needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")


@needs_two_gpus
@pytest.mark.parametrize("communicator", ["lazy", "eager"])
def test_nccl_collectives_inside_loans(communicator):
    script = os.path.join(os.path.dirname(__file__), "nccl_loan.py")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", script, communicator]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


@needs_two_gpus
def test_concurrent_loans_on_two_devices():
    caches = [memlend.empty(8, 64 * MiB, dtype=torch.uint8, device=d) for d in (0, 1)]
    for c in caches:
        c.view(torch.int64).random_()
    before = [c.clone() for c in caches]
    with memlend.lend(caches[0]), memlend.lend(caches[1]):
        for d in (0, 1):
            torch.empty(100 * MiB, dtype=torch.uint8, device=d).fill_(d + 1)
    assert all(torch.equal(c, b) for c, b in zip(caches, before))
