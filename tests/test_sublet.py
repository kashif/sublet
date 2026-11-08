import subprocess
import sys

import pytest
import torch

import sublet
from conftest import ROW, MiB, autograd_step, dev, from_cache, gpu_filled, make_cache, make_model


def test_free_rows_are_lent_before_kept_ones():
    cache, before = make_cache()
    odd = range(1, 16, 2)
    with sublet.lend(cache, keep=[cache[i] for i in odd]) as loan:
        rows = [torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1) for _ in range(8)]  # one per free row
        assert all(from_cache(cache, t) for t in rows)
        del rows
    assert loan.copied_bytes == 0 and loan.extra_bytes == 0
    assert bool((cache[0::2] == 1).all())  # the borrower wrote in the free rows
    assert all(torch.equal(cache[i], before[i]) for i in odd)


def test_an_allocation_longer_than_the_free_stretches_takes_free_gpu_memory():
    cache, before = make_cache()
    with sublet.lend(cache, keep=[cache[i] for i in range(1, 16, 2)]) as loan:
        x = torch.empty(8 * ROW, dtype=torch.uint8, device=dev).fill_(1)  # the free rows are one row long
        assert not from_cache(cache, x)
        del x
    assert loan.copied_bytes == 0 and loan.extra_bytes == 8 * ROW
    assert torch.equal(cache, before)


def test_kept_memory_is_copied_when_the_gpu_is_full():
    cache, before = make_cache()  # all of it kept
    with gpu_filled(2 * ROW):
        with sublet.lend(cache) as loan:
            x = torch.empty(8 * ROW, dtype=torch.uint8, device=dev).fill_(1)
            assert from_cache(cache, x)
            del x
    assert loan.copied_bytes == 8 * ROW and loan.extra_bytes == 0
    assert torch.equal(cache, before)


def test_strided_views_keep_only_their_pages():
    torch.cuda.empty_cache()
    cache = sublet.empty(4, 8, 2 * MiB, dtype=torch.uint8)  # 4 layers of 8 blocks of one page
    cache.view(torch.int64).random_()
    before = cache.clone()
    torch.cuda.empty_cache()
    with sublet.lend(cache, keep=cache[:, 3]) as loan:  # block 3 of every layer
        # Between the kept blocks: stretches of 3, 7, 7, 7 and 4 pages
        xs = [torch.empty(14 * MiB, dtype=torch.uint8, device=dev).fill_(1) for _ in range(3)]
        assert all(from_cache(cache, x) for x in xs)
        del xs
    assert loan.copied_bytes == 0 and loan.extra_bytes == 0
    assert torch.equal(cache[:, 3], before[:, 3])


def test_lend_and_keep_need_views_of_a_lendable_tensor():
    cache, _ = make_cache(2)
    with pytest.raises(ValueError, match="made by sublet.empty"):
        sublet.lend(torch.empty(10, device=dev))
    with pytest.raises(ValueError, match="views of the lent tensor"):
        sublet.lend(cache, keep=[cache[[0, 1]]])  # indexing with a list copies


def test_each_loan_copies_only_what_it_needs():
    cache, before = make_cache()
    with gpu_filled(32 * MiB):
        with sublet.lend(cache, keep=[]):
            torch.empty(12 * ROW, dtype=torch.uint8, device=dev).fill_(1)
        cache.copy_(before)
        with sublet.lend(cache) as loan:  # all of it kept: what the borrower takes is copied, and nothing else
            torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(2)
    assert loan.copied_bytes == ROW
    assert torch.equal(cache, before)


def test_memory_freed_on_one_stream_serves_another():
    # PyTorch keeps the memory freed on a stream for that stream, until another one runs out of memory
    cache, before = make_cache()
    side = torch.cuda.Stream()
    with gpu_filled(32 * MiB):
        with sublet.lend(cache) as loan:
            torch.empty(12 * ROW, dtype=torch.uint8, device=dev).fill_(1)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                torch.empty(12 * ROW, dtype=torch.uint8, device=dev).fill_(2)
            torch.cuda.current_stream().wait_stream(side)
    assert loan.copied_bytes == 12 * ROW  # the same pages, copied once
    assert torch.equal(cache, before)


def test_memory_beyond_the_tensor_is_given_back():
    cache, _ = make_cache()  # 1 GiB
    free = torch.cuda.mem_get_info()[0]
    with sublet.lend(cache, keep=[]) as loan:
        tensors = [torch.full((300 * MiB,), i + 1, dtype=torch.uint8, device=dev) for i in range(4)]
        # Three stretches of the tensor, then free GPU memory
        assert [from_cache(cache, t) for t in tensors] == [True, True, True, False]
        del tensors
    assert loan.extra_bytes == 300 * MiB
    assert torch.cuda.mem_get_info()[0] >= free - 16 * MiB


def test_allocations_alive_at_the_end_are_moved_out_of_the_tensor():
    cache, _ = make_cache()
    with sublet.lend(cache, keep=[]) as loan:  # the borrower takes the tensor's memory first
        survivor = torch.full((20 * MiB,), 7, dtype=torch.uint8, device=dev)
        assert from_cache(cache, survivor)
    assert loan.moved_bytes == 20 * MiB and not from_cache(cache, survivor)
    with gpu_filled(32 * MiB), sublet.lend(cache, keep=[]):  # the next loan writes over the whole tensor
        torch.empty(cache.numel(), dtype=torch.uint8, device=dev).fill_(1)
    assert bool((cache == 1).all()) and bool((survivor == 7).all())


def test_allocations_that_cannot_be_moved_out_raise_leak_error():
    cache, before = make_cache()
    with gpu_filled(8 * MiB):
        with pytest.raises(sublet.LeakError, match="still alive"):
            with sublet.lend(cache):
                survivor = torch.full((6 * ROW,), 3, dtype=torch.uint8, device=dev)
    assert torch.equal(cache, before)  # the kept memory comes back all the same
    survivor.fill_(1)  # its address stays valid
    torch.cuda.synchronize()


def test_an_exception_comes_out_of_the_loan_and_the_kept_memory_comes_back():
    cache, before = make_cache()
    with gpu_filled(2 * ROW):
        with pytest.warns(UserWarning, match="still alive"), pytest.raises(torch.OutOfMemoryError):
            with sublet.lend(cache):
                x = torch.empty(4 * ROW, dtype=torch.uint8, device=dev).fill_(1)  # alive, and too large to move
                torch.empty(1024**5, dtype=torch.uint8, device=dev)
    del x
    assert torch.equal(cache, before)


def test_end_ends_the_loan_once():
    cache, before = make_cache(4)
    loan = sublet.lend(cache)
    torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1)
    loan.end()
    loan.end()
    assert loan.copied_bytes == 0 and torch.equal(cache, before)
    with sublet.lend(cache):  # a new loan can start
        pass


def test_autograd_and_an_optimizer_state_made_during_a_loan():
    tokens = torch.randint(0, 1000, (8, 129), device=dev, generator=torch.Generator(dev).manual_seed(0))
    ref_model, ref_update = make_model()
    ref_losses = [autograd_step(ref_model, ref_update, tokens) for _ in range(3)]
    model, update = make_model()
    cache, _ = make_cache()
    losses, moved = [], []
    for _ in range(3):
        with sublet.lend(cache, keep=[]) as loan:  # the step takes the tensor's memory first
            losses.append(autograd_step(model, update, tokens))
        moved.append(loan.moved_bytes)
    assert losses == ref_losses
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), ref_model.parameters()))
    assert moved[0] > 0  # AdamW's state, made by the first step, moved out of the tensor


def test_side_streams_and_cublas_workspaces():
    cache, before = make_cache()
    a = torch.randn(1024, 1024, device=dev)
    out = torch.empty(2, 1024, 1024, device=dev)
    with sublet.lend(cache, keep=cache[8:]) as loan:
        for i in range(2):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                tmp = a @ a
                assert from_cache(cache, tmp)
                out[i].copy_(tmp)
            torch.cuda.current_stream().wait_stream(s)
        del tmp
    assert loan.moved_bytes == 0  # the workspaces cuBLAS made for the new streams are dropped
    assert torch.equal(out[0], a @ a) and torch.equal(out[1], a @ a) and torch.equal(cache[8:], before[8:])


def test_tensor_dropped_before_an_allocation_made_during_its_loan():
    cache, _ = make_cache(4)
    assert bool((torch.full((1,), 1, dtype=torch.uint8, device=dev) == 1).all())  # loads the kernels the test runs
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0] + cache.numel()
    with sublet.lend(cache, keep=[]):
        survivor = torch.full((10 * MiB,), 1, dtype=torch.uint8, device=dev)
    del cache
    assert bool((survivor == 1).all())
    del survivor
    torch.cuda.empty_cache()  # frees its memory, then the tensor's
    assert torch.cuda.mem_get_info()[0] >= free_before - 16 * MiB


def test_cuda_graph_captured_on_the_owners_tensor_stays_valid():
    cache, before = make_cache()
    head, out = cache[0, : 16 * MiB], torch.empty(16 * MiB, dtype=torch.uint8, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(head)
        head.add_(1)
    torch.cuda.empty_cache()
    with sublet.lend(cache, keep=[cache[0]]):
        torch.empty(15 * ROW, dtype=torch.uint8, device=dev).fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, before[0, : 16 * MiB]) and torch.equal(head, before[0, : 16 * MiB] + 1)


def test_owner_kernels_still_running_finish_before_the_loan_starts():
    cache, _ = make_cache()
    with gpu_filled(32 * MiB):
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(int(5e8))
            cache[15].fill_(42)  # the owner's last write, still queued when it lends
        with sublet.lend(cache) as loan:
            torch.empty(cache.numel(), dtype=torch.uint8, device=dev).fill_(1)  # all of the tensor, copied first
    assert loan.copied_bytes == cache.numel() and bool((cache[15] == 42).all())


def test_tensor_dropped_during_another_loan():
    a, _ = make_cache(4)
    b, before = make_cache(4)
    with sublet.lend(a, keep=[]):
        torch.empty(ROW, dtype=torch.uint8, device=dev)
    with sublet.lend(b):
        x = torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1)
        del a
        del x
    assert torch.equal(b, before)


def test_memory_lives_as_long_as_its_tensor_views():
    free_before = torch.cuda.mem_get_info()[0]
    cache = sublet.empty(1024 * MiB, dtype=torch.uint8)
    view = cache[:10]
    del cache
    assert torch.cuda.mem_get_info()[0] <= free_before - 1024 * MiB
    del view
    assert torch.cuda.mem_get_info()[0] >= free_before - 64 * MiB


@pytest.mark.parametrize(
    "script",
    [
        "with sublet.lend(c, keep=[]):\n    kept.append(torch.empty(10 * 1024**2, device='cuda'))",
        "with sublet.lend(c, keep=[]):\n    kept.append(torch.empty(10 * 1024**2, device='cuda'))\ndel c",
    ],
    ids=["allocation made during a loan alive at exit", "tensor dropped before an allocation made during its loan"],
)
def test_interpreter_shutdown(script):
    code = "import torch\nimport sublet\nc = sublet.empty(256 * 1024**2, dtype=torch.uint8)\nkept = []\n" + script + "\nprint('done')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "done" in result.stdout, result.stderr[-3000:]
