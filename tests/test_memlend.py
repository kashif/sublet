import pytest
import torch

from memlend import PAGE, LeakError, LendableRegion

from helpers import (
    BLOCK,
    MiB,
    assert_untouched_identical,
    changed_blocks,
    dev,
    in_region,
    make_model_and_optimizer,
    make_region,
    train_step,
)


def test_untouched_blocks_bit_identical_and_touched_reported():
    region, before = make_region(1024 * MiB)  # 16 blocks of 32 pages

    def borrower():
        # Segments the caching allocator asks for: 100 MiB, 20 MiB (for 3 MiB), 2 MiB (small pool), 40 MiB.
        # Served from the far end: pages 431..511 => blocks 13, 14, 15.
        tensors = [torch.empty(n, dtype=torch.uint8, device=dev) for n in (100 * MiB, 3 * MiB, 500_000, 40 * MiB)]
        for i, t in enumerate(tensors):
            assert in_region(region, t)
            t.fill_(i + 1)

    with region.lend() as loan:
        borrower()
    assert loan.touched_blocks == [13, 14, 15]
    assert loan.fallback_peak_bytes == 0
    assert assert_untouched_identical(region, before, loan) == [13, 14, 15]


def test_lend_order():
    region, before = make_region(1024 * MiB)
    region.set_lend_order([5, 9, 10, 2])

    def borrower():
        a = torch.empty(64 * MiB, dtype=torch.uint8, device=dev).fill_(1)  # 32 pages: block 5
        b = torch.empty(128 * MiB, dtype=torch.uint8, device=dev).fill_(2)  # 64 contiguous pages: blocks 9 and 10
        c = torch.empty(100, dtype=torch.uint8, device=dev).fill_(3)  # one small-pool page: first page of block 2
        assert a.data_ptr() - region.tensor.data_ptr() == 5 * BLOCK
        assert b.data_ptr() - region.tensor.data_ptr() == 9 * BLOCK
        assert c.data_ptr() - region.tensor.data_ptr() == 2 * BLOCK

    with region.lend() as loan:
        borrower()
    assert loan.touched_blocks == [2, 5, 9, 10]
    assert assert_untouched_identical(region, before, loan) == [2, 5, 9, 10]


def test_borrower_frees_and_reallocates():
    region, before = make_region(1024 * MiB)
    gen = torch.Generator().manual_seed(0)

    def borrower():
        live = []
        for _ in range(300):
            n = int(torch.randint(1, 64 * MiB, (1,), generator=gen))
            live.append(torch.empty(n, dtype=torch.uint8, device=dev).fill_(7))
            if len(live) > 6:  # at most ~400 MiB alive at once
                live.pop(int(torch.randint(0, len(live), (1,), generator=gen)))

    with region.lend() as loan:
        borrower()
    # Freed memory is reused inside the loan: far fewer blocks than the 300 allocations would need.
    assert 0 < len(loan.touched_blocks) < region.num_blocks
    assert loan.fallback_peak_bytes == 0
    assert_untouched_identical(region, before, loan)


def test_autograd_training_step_matches_reference():
    tokens = torch.randint(0, 1000, (8, 129), device=dev, generator=torch.Generator(dev).manual_seed(0))
    ref_model, ref_opt = make_model_and_optimizer()
    ref_losses = [train_step(ref_model, ref_opt, tokens) for _ in range(3)]

    model, opt = make_model_and_optimizer()
    losses = [train_step(model, opt, tokens)]  # first step outside: creates the optimizer states
    region, before = make_region(1024 * MiB)
    with region.lend() as loan:
        losses.append(train_step(model, opt, tokens, region))
    with region.lend() as loan2:  # a second loan on the same region
        losses.append(train_step(model, opt, tokens, region))

    assert losses == ref_losses
    for p, ref_p in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(p, ref_p)
    assert loan.touched_blocks and loan2.touched_blocks
    assert loan.fallback_peak_bytes == 0 and loan2.fallback_peak_bytes == 0
    assert_untouched_identical(region, before, loan2)


def test_side_streams_and_cublas_workspaces():
    region, before = make_region(1024 * MiB)
    a = torch.randn(1024, 1024, device=dev)
    expected = a @ a
    out = torch.empty(2, 1024, 1024, device=dev)

    def borrower():
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]  # new streams: cuBLAS creates a workspace for each
        for i, s in enumerate(streams):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                tmp = a @ a
                assert in_region(region, tmp)
                out[i].copy_(tmp)
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)

    with region.lend() as loan:  # no LeakError: the workspaces made during the loan are released
        borrower()
    assert torch.equal(out[0], expected) and torch.equal(out[1], expected)
    assert loan.touched_blocks
    assert_untouched_identical(region, before, loan)


def test_leak_is_detected_and_memory_recovers():
    region, before = make_region(256 * MiB)
    kept = []
    with pytest.raises(LeakError, match=r"1 allocation\(s\) made during the loan are still alive"):
        with region.lend():
            kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev))
    assert in_region(region, kept[0])

    # Once the tensor is gone and the cache emptied, the whole region can be lent again.
    kept.clear()
    torch.cuda.empty_cache()
    with region.lend() as loan:
        t = torch.empty(region.nbytes, dtype=torch.uint8, device=dev)
        assert t.data_ptr() == region.tensor.data_ptr()
        del t
    assert loan.fallback_peak_bytes == 0


def test_optimizer_state_created_during_loan_is_a_leak():
    model, opt = make_model_and_optimizer()
    tokens = torch.randint(0, 1000, (2, 33), device=dev)
    region, _ = make_region(512 * MiB)
    with pytest.raises(LeakError, match="optimizer states"):
        with region.lend():
            train_step(model, opt, tokens)  # AdamW creates exp_avg/exp_avg_sq on its first step


def test_borrower_exceeds_region():
    region, before = make_region(1024 * MiB)

    def borrower():
        tensors = [torch.full((300 * MiB,), i + 1, dtype=torch.uint8, device=dev) for i in range(4)]  # 1200 MiB
        assert [in_region(region, t) for t in tensors] == [True, True, True, False]
        assert all(bool((t == i + 1).all()) for i, t in enumerate(tensors))

    with region.lend() as loan:
        borrower()
    assert loan.fallback_peak_bytes >= 300 * MiB
    assert loan.touched_blocks == list(range(1, 16))  # 450 pages from the far end; block 0 keeps 32 - 30 free pages
    assert_untouched_identical(region, before, loan)


def test_cuda_graph_captured_on_region_stays_valid():
    region, before = make_region(1024 * MiB)
    head = region.tensor[: 16 * MiB]
    out = torch.empty_like(head)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(head)
        head.add_(1)

    ptr = region.tensor.data_ptr()
    with region.lend() as loan:
        torch.empty(200 * MiB, dtype=torch.uint8, device=dev).fill_(0)  # far end: block 0 untouched
    assert region.tensor.data_ptr() == ptr and 0 not in loan.touched_blocks
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, before[: 16 * MiB])
    assert torch.equal(head, before[: 16 * MiB] + 1)


@pytest.mark.parametrize("mode", ["upfront", "lazy"])
def test_restore_touched_blocks_gives_back_everything(mode):
    region, before = make_region(1024 * MiB)
    backup = torch.empty(region.nbytes, dtype=torch.uint8, pin_memory=True)
    if mode == "upfront":
        backup.copy_(region.tensor)
    else:
        backup.zero_()  # lazy: only the touched blocks get written into it

    with region.lend(backup=backup if mode == "lazy" else None) as loan:
        torch.empty(300 * MiB, dtype=torch.uint8, device=dev).fill_(0)
        torch.empty(5 * MiB, dtype=torch.uint8, device=dev).fill_(0)
    assert changed_blocks(region, before) == loan.touched_blocks
    region.restore(backup, loan.touched_blocks)
    torch.cuda.synchronize()
    assert torch.equal(region.tensor, before)


def test_region_memory_lives_as_long_as_its_tensor_views():
    free_before = torch.cuda.mem_get_info()[0]
    region = LendableRegion(1024 * MiB)
    view = region.tensor[:10]
    del region
    assert torch.cuda.mem_get_info()[0] <= free_before - 1024 * MiB
    del view
    assert torch.cuda.mem_get_info()[0] >= free_before - 64 * MiB


def test_block_size_validation():
    with pytest.raises(ValueError):
        LendableRegion(64 * MiB, block_size=PAGE + 1)
    with pytest.raises(ValueError):
        LendableRegion(65 * MiB, block_size=64 * MiB)
