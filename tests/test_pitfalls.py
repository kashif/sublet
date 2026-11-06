"""Known pitfalls of lending GPU memory (stream ordering at the loan boundaries, interpreter shutdown)."""

import subprocess
import sys

import pytest
import torch

from helpers import BLOCK, MiB, dev, make_region

SLEEP_CYCLES = int(5e8)  # torch.cuda._sleep: a few hundred ms of GPU time


def test_borrower_kernels_still_running_finish_before_the_loan_ends():
    region, before = make_region(1024 * MiB)
    side = torch.cuda.Stream()
    with region.lend() as loan:
        with torch.cuda.stream(side):
            t = torch.empty(100 * MiB, dtype=torch.uint8, device=dev)
            torch.cuda._sleep(SLEEP_CYCLES)
            t.fill_(7)  # still queued behind the sleep when the loan ends
        del t  # freed on the host while its kernel is pending
    # The owner takes its memory back immediately, on another stream.
    region.tensor[: region.nbytes].copy_(before)
    torch.cuda.synchronize()
    assert loan.touched_blocks and torch.equal(region.tensor, before)


def test_owner_kernels_still_running_finish_before_the_loan_starts():
    region, _ = make_region(1024 * MiB)
    owner = torch.cuda.Stream()
    with torch.cuda.stream(owner):
        torch.cuda._sleep(SLEEP_CYCLES)
        region.tensor[15 * BLOCK :].fill_(42)  # the owner's last write, still queued when it lends
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    with region.lend(backup=backup) as loan:
        torch.empty(64 * MiB, dtype=torch.uint8, device=dev).fill_(1)  # block 15, lent first
    assert loan.touched_blocks == [15]
    assert bool((backup[15 * BLOCK :] == 42).all())


SHUTDOWN = {
    "clean": "with r.lend():\n    torch.empty(10 << 20, device='cuda')",
    "leaked tensor alive at exit": "try:\n    with r.lend():\n        kept.append(torch.empty(10 << 20, device='cuda'))\nexcept LeakError:\n    pass",
    "region dropped before its leaked tensor": "try:\n    with r.lend():\n        kept.append(torch.empty(10 << 20, device='cuda'))\nexcept LeakError:\n    pass\ndel r",
    "exception inside the loan": "try:\n    with r.lend():\n        x = torch.empty(10 << 20, device='cuda')\n        raise ValueError\nexcept ValueError as e:\n    kept.append(e)",
}


@pytest.mark.parametrize("scenario", list(SHUTDOWN))
def test_interpreter_shutdown(scenario):
    # MemPool / pluggable allocator destruction order has crashed at exit in some torch versions
    # ("pure virtual method called", pytorch#145168, #146431).
    code = "import torch\nfrom memlend import LendableRegion, LeakError\nr = LendableRegion(256 << 20)\nkept = []\n" + SHUTDOWN[scenario] + "\nprint('done')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "done" in result.stdout, result.stderr[-3000:]
    assert "pure virtual" not in result.stderr and "Segmentation" not in result.stderr
