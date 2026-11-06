import pytest
import torch

from helpers import MiB, dev, make_region


def fill_new(nbytes, value):
    t = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    t.fill_(value)  # written right after allocation: the backup copy of its blocks must already be ordered before it
    return t


@pytest.mark.parametrize("sync_copies,prefetch", [(True, 0), (False, 0), (False, 4)], ids=["host-sync", "stream", "stream+prefetch"])
def test_lazy_backup_copies_only_dirty_blocks(sync_copies, prefetch):
    region, before = make_region(1024 * MiB)  # 16 blocks of 32 pages
    empty = [15, 14, 3]
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)

    def borrower():
        tensors = [fill_new(64 * MiB, 1) for _ in range(3)]  # blocks 15, 14, 3: the empty ones, lent first
        assert [t.data_ptr() - region.tensor.data_ptr() for t in tensors] == [15 * 64 * MiB, 14 * 64 * MiB, 3 * 64 * MiB]
        tensors.append(fill_new(200 * MiB, 2))  # then the far end: pages 320..419, blocks 10 to 13

    with region.lend(empty=empty, backup=backup, _prefetch=prefetch, _sync_copies=sync_copies) as loan:
        borrower()
    assert loan.touched_blocks == [3, 10, 11, 12, 13, 14, 15]
    assert loan.dirty_blocks == [10, 11, 12, 13]

    blocks, ref, saved = (t.view(16, -1) for t in (region.tensor, before, backup))
    for b in loan.dirty_blocks:
        assert torch.equal(saved[b].to(dev), ref[b])
    for b in empty:
        assert not saved[b].any()  # never copied
    region.restore(backup, loan.dirty_blocks)
    torch.cuda.synchronize()
    for b in range(16):
        if b not in empty:
            assert torch.equal(blocks[b], ref[b])


def test_borrower_fitting_in_empty_blocks_moves_zero_bytes():
    region, before = make_region(1024 * MiB)
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    with region.lend(empty=[0, 1, 2, 3], backup=backup) as loan:
        fill_new(200 * MiB, 1)
    assert loan.touched_blocks == [0, 1, 2, 3] and loan.dirty_blocks == []
    assert not backup.any()
    assert torch.equal(region.tensor[4 * 64 * MiB :], before[4 * 64 * MiB :])


def test_lazy_backup_with_allocations_on_side_streams():
    region, before = make_region(1024 * MiB)
    backup = torch.zeros(region.nbytes, dtype=torch.uint8, pin_memory=True)
    with region.lend(backup=backup) as loan:
        streams = [torch.cuda.Stream() for _ in range(2)]
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                fill_new(150 * MiB, i + 1)
        torch.cuda.synchronize()
    region.restore(backup, loan.dirty_blocks)
    torch.cuda.synchronize()
    assert loan.dirty_blocks and torch.equal(region.tensor, before)
