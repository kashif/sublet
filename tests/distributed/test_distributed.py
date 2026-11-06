import os
import subprocess
import sys

import pytest
import torch

from memlend import LendableRegion

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
    regions = [LendableRegion(512 * MiB, 64 * MiB, device=d) for d in (0, 1)]
    for r in regions:
        r.tensor.view(torch.int64).random_()
    before = [r.tensor.clone() for r in regions]
    with regions[0].lend() as loan0, regions[1].lend() as loan1:
        for d, r in enumerate(regions):
            t = torch.empty(100 * MiB, dtype=torch.uint8, device=d).fill_(d + 1)
            assert r.tensor.data_ptr() <= t.data_ptr() < r.tensor.data_ptr() + r.nbytes
            del t
    for r, b, loan in zip(regions, before, (loan0, loan1)):
        assert loan.touched_blocks == [6, 7]
        view, ref = r.tensor.view(r.num_blocks, -1), b.view(r.num_blocks, -1)
        assert all(torch.equal(view[i], ref[i]) for i in range(r.num_blocks) if i not in loan.touched_blocks)
