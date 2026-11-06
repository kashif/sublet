import subprocess
import sys

import pytest
import torch

from conftest import BLOCK, MiB, assert_untouched_identical, autograd_step, dev, in_region, make_model, make_region
from memlend import LeakError, LendableRegion


def test_untouched_blocks_bit_identical_and_touched_reported():
    region, before = make_region(1024 * MiB)  # 16 blocks of 32 pages

    def borrower():
        # Segments requested: 100 MiB, 20 MiB (for 3 MiB), 2 MiB (small pool), 40 MiB: 81 pages from the far end.
        kept = [torch.empty(n, dtype=torch.uint8, device=dev) for n in (100 * MiB, 3 * MiB, 500_000, 40 * MiB)]
        for i, t in enumerate(kept):
            assert in_region(region, t)
            t.fill_(i + 1)

    with region.lend() as loan:
        borrower()
    assert loan.touched_blocks == [13, 14, 15] and loan.fallback_peak_bytes == 0
    assert assert_untouched_identical(region, before, loan) == [13, 14, 15]


def test_one_allocation_takes_scattered_empty_blocks():
    # Every other block holds data: one allocation of all the empty blocks lends no data block.
    region, before = make_region(1024 * MiB)
    empty = range(0, 16, 2)
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    with region.lend(empty=empty, backup=backup) as loan:
        a = torch.empty(8 * BLOCK, dtype=torch.uint8, device=dev).fill_(1)
        assert in_region(region, a)
        b = torch.empty(BLOCK, dtype=torch.uint8, device=dev).fill_(2)  # the data block at the far end
        del a, b
    assert loan.touched_blocks == list(empty) + [15] and loan.dirty_blocks == [15] and loan.fallback_peak_bytes == 0
    region.restore(backup, loan.dirty_blocks)
    torch.cuda.synchronize()
    blocks, ref = region.tensor.view(16, -1), before.view(16, -1)
    assert all(torch.equal(blocks[b], ref[b]) for b in range(16) if b not in empty)


def test_single_block_request():
    region, _ = make_region(1024 * MiB)
    with region.lend() as loan:  # nothing declared empty: the far end
        torch.empty(BLOCK, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.touched_blocks == [15]
    with region.lend(empty=[5]) as loan:
        torch.empty(BLOCK, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.touched_blocks == [5] and loan.dirty_blocks == []


def test_borrower_frees_and_reallocates():
    region, before = make_region(1024 * MiB)
    gen = torch.Generator().manual_seed(0)

    def borrower():
        live = []
        for _ in range(300):
            live.append(torch.empty(int(torch.randint(1, 64 * MiB, (1,), generator=gen)), dtype=torch.uint8, device=dev).fill_(7))
            if len(live) > 6:
                live.pop(int(torch.randint(0, len(live), (1,), generator=gen)))

    with region.lend() as loan:
        borrower()
    assert 0 < len(loan.touched_blocks) < region.num_blocks  # freed memory is reused within the loan
    assert_untouched_identical(region, before, loan)


def test_borrower_using_autograd_matches_reference():
    tokens = torch.randint(0, 1000, (8, 129), device=dev, generator=torch.Generator(dev).manual_seed(0))
    ref_model, ref_update = make_model()
    ref_losses = [autograd_step(ref_model, ref_update, tokens) for _ in range(3)]
    model, update = make_model()
    losses = [autograd_step(model, update, tokens)]  # outside a loan: creates the AdamW state
    region, before = make_region(1024 * MiB)
    for _ in range(2):
        with region.lend() as loan:
            losses.append(autograd_step(model, update, tokens, region))
    assert losses == ref_losses
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), ref_model.parameters()))
    assert loan.touched_blocks and loan.fallback_peak_bytes == 0
    assert_untouched_identical(region, before, loan)


def test_side_streams_and_cublas_workspaces():
    region, before = make_region(1024 * MiB)
    a = torch.randn(1024, 1024, device=dev)
    out = torch.empty(2, 1024, 1024, device=dev)
    with region.lend() as loan:  # no LeakError: cuBLAS workspaces made for the new streams are released
        for i, s in enumerate([torch.cuda.Stream(), torch.cuda.Stream()]):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                tmp = a @ a
                assert in_region(region, tmp)
                out[i].copy_(tmp)
            torch.cuda.current_stream().wait_stream(s)
        del tmp
    assert torch.equal(out[0], a @ a) and torch.equal(out[1], a @ a)
    assert_untouched_identical(region, before, loan)


def test_leak_is_detected_and_memory_recovers():
    region, _ = make_region(256 * MiB)
    kept = []
    with pytest.raises(LeakError, match=r"1 allocation\(s\) made during the loan are still alive"):
        with region.lend():
            kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev))
    assert in_region(region, kept[0])
    kept.clear()
    torch.cuda.empty_cache()
    with region.lend() as loan:  # the whole region can be lent again
        t = torch.empty(region.nbytes, dtype=torch.uint8, device=dev)
        assert in_region(region, t)
        del t
    assert loan.fallback_peak_bytes == 0


def test_borrower_exceeding_the_region_uses_free_memory():
    region, before = make_region(1024 * MiB)
    with region.lend() as loan:
        # The fourth takes the region's last 124 MiB and 176 MiB of free device memory.
        tensors = [torch.full((300 * MiB,), i + 1, dtype=torch.uint8, device=dev) for i in range(4)]
        assert all(in_region(region, t) for t in tensors)
        assert all(bool((t.cpu() == i + 1).all()) for i, t in enumerate(tensors))  # checked on the CPU: no allocation
        del tensors
    assert loan.fallback_peak_bytes == 4 * 300 * MiB - region.nbytes and loan.touched_blocks == list(range(16))
    assert_untouched_identical(region, before, loan)


def test_cuda_graph_captured_on_region_stays_valid():
    region, before = make_region(1024 * MiB)
    head, out = region.tensor[: 16 * MiB], torch.empty(16 * MiB, dtype=torch.uint8, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(head)
        head.add_(1)
    with region.lend() as loan:
        torch.empty(200 * MiB, dtype=torch.uint8, device=dev).fill_(0)
    assert 0 not in loan.touched_blocks
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, before[: 16 * MiB]) and torch.equal(head, before[: 16 * MiB] + 1)


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_backup_and_restore_give_back_everything(mode):
    region, before = make_region(1024 * MiB)
    empty = [15, 14, 3]
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    if mode == "eager":
        region.save(backup, [b for b in range(16) if b not in empty])
    with region.lend(empty=empty, backup=backup if mode == "lazy" else None) as loan:
        # Blocks 15, 14 and 3 (empty), then the data blocks nearest the far end: 13, 12, 11 and 4 pages of 10.
        kept = [torch.empty(n, dtype=torch.uint8, device=dev).fill_(1) for n in (64 * MiB, 64 * MiB, 64 * MiB, 200 * MiB)]
        del kept
    assert loan.touched_blocks == [3, 10, 11, 12, 13, 14, 15] and loan.dirty_blocks == [10, 11, 12, 13]
    if mode == "lazy":
        assert all(not backup.view(16, -1)[b].any() for b in empty)  # never copied
    region.restore(backup, loan.dirty_blocks)
    torch.cuda.synchronize()
    assert all(torch.equal(region.tensor.view(16, -1)[b], before.view(16, -1)[b]) for b in range(16) if b not in empty)


def test_lazy_backup_with_allocations_on_side_streams():
    region, before = make_region(1024 * MiB)
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    with region.lend(backup=backup) as loan:
        for i in range(2):
            with torch.cuda.stream(torch.cuda.Stream()):
                torch.empty(150 * MiB, dtype=torch.uint8, device=dev).fill_(i + 1)
        torch.cuda.synchronize()
    region.restore(backup, loan.dirty_blocks)
    torch.cuda.synchronize()
    assert loan.dirty_blocks and torch.equal(region.tensor, before)


def test_owner_kernels_still_running_finish_before_the_loan_starts():
    region, _ = make_region(1024 * MiB)
    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda._sleep(int(5e8))
        region.tensor[15 * BLOCK :].fill_(42)  # the owner's last write, still queued when it lends
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    with region.lend(backup=backup) as loan:
        torch.empty(BLOCK, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.touched_blocks == [15] and bool((backup[15 * BLOCK :] == 42).all())


def test_region_memory_lives_as_long_as_its_tensor_views():
    free_before = torch.cuda.mem_get_info()[0]
    region = LendableRegion(1024 * MiB)
    view = region.tensor[:10]
    del region
    assert torch.cuda.mem_get_info()[0] <= free_before - 1024 * MiB
    del view
    assert torch.cuda.mem_get_info()[0] >= free_before - 64 * MiB


@pytest.mark.parametrize(
    "script",
    [
        "try:\n    with r.lend():\n        kept.append(torch.empty(10 * 1024**2, device='cuda'))\nexcept LeakError:\n    pass",
        "try:\n    with r.lend():\n        kept.append(torch.empty(10 * 1024**2, device='cuda'))\nexcept LeakError:\n    pass\ndel r",
    ],
    ids=["leaked tensor alive at exit", "region dropped before its leaked tensor"],
)
def test_interpreter_shutdown(script):
    # Pool and pluggable-allocator destruction order has crashed at exit in some torch versions (pytorch#146431).
    code = "import torch\nfrom memlend import LendableRegion, LeakError\nr = LendableRegion(256 * 1024**2)\nkept = []\n" + script + "\nprint('done')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "done" in result.stdout, result.stderr[-3000:]
