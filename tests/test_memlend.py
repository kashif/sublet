import subprocess
import sys

import pytest
import torch

import memlend
from conftest import ROW, MiB, autograd_step, dev, from_cache, make_cache, make_model


def test_free_rows_are_lent_before_kept_ones():
    cache, before = make_cache()
    odd = range(1, 16, 2)
    with memlend.lend(cache, keep=[cache[i] for i in odd]) as loan:
        x = torch.empty(8 * ROW, dtype=torch.uint8, device=dev).fill_(1)  # as much as the 8 free rows, scattered
        assert from_cache(cache, x)
        del x
    assert loan.copied_bytes == 0 and loan.extra_bytes == 0
    assert bool((cache[0::2] == 1).all())  # the borrower wrote in the free rows
    assert all(torch.equal(cache[i], before[i]) for i in odd)


@pytest.mark.parametrize("keep, copied", [(None, 300 * MiB), ([], 0)], ids=["everything", "nothing"])
def test_kept_memory_the_borrower_uses_is_copied(keep, copied):
    cache, before = make_cache()
    with memlend.lend(cache, keep=keep) as loan:
        torch.empty(300 * MiB, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.copied_bytes == copied and loan.extra_bytes == 0
    assert torch.equal(cache, before) == (keep is None)


def test_strided_views_keep_only_their_pages():
    cache = memlend.empty(4, 8, 2 * MiB, dtype=torch.uint8)  # 4 layers of 8 blocks of one page
    cache.view(torch.int64).random_()
    before = cache.clone()
    with memlend.lend(cache, keep=cache[:, 3]) as loan:  # block 3 of every layer
        torch.empty(cache.numel(), dtype=torch.uint8, device=dev).fill_(1)
    assert loan.copied_bytes == 4 * 2 * MiB and torch.equal(cache[:, 3], before[:, 3])


def test_lend_and_keep_need_views_of_a_lendable_tensor():
    cache, _ = make_cache(2)
    with pytest.raises(ValueError, match="made by memlend.empty"):
        memlend.lend(torch.empty(10, device=dev))
    with pytest.raises(ValueError, match="views of the lent tensor"):
        memlend.lend(cache, keep=[cache[[0, 1]]])  # indexing with a list copies


def test_later_loans_reuse_the_borrowers_memory():
    cache, before = make_cache()
    with memlend.lend(cache, keep=[]):
        torch.empty(4 * ROW, dtype=torch.uint8, device=dev).fill_(1)
    cache.copy_(before)  # the owner writes everywhere, including where the borrower's memory stays
    with memlend.lend(cache) as loan:
        torch.empty(4 * ROW, dtype=torch.uint8, device=dev).fill_(2)
    assert loan.copied_bytes == 4 * ROW and torch.equal(cache, before)


def test_borrower_exceeding_the_tensor_uses_free_memory():
    cache, _ = make_cache()
    with memlend.lend(cache, keep=[]) as loan:
        # The fourth takes the last 124 MiB of the tensor and 176 MiB of free device memory.
        tensors = [torch.full((300 * MiB,), i + 1, dtype=torch.uint8, device=dev) for i in range(4)]
        assert all(from_cache(cache, t) for t in tensors)
        assert all(bool((t.cpu() == i + 1).all()) for i, t in enumerate(tensors))  # checked on the CPU: no allocation
        del tensors
    assert loan.extra_bytes == 4 * 300 * MiB - cache.numel()
    with memlend.lend(cache, keep=[]) as loan:  # the memory beyond the tensor was given back
        torch.empty(cache.numel(), dtype=torch.uint8, device=dev)
    assert loan.extra_bytes == 0


def test_kept_memory_comes_back_after_an_exception():
    cache, before = make_cache()
    with pytest.raises(torch.OutOfMemoryError):
        with memlend.lend(cache):
            x = torch.empty(4 * ROW, dtype=torch.uint8, device=dev).fill_(1)
            torch.empty(1024**5, dtype=torch.uint8, device=dev)
    del x
    assert torch.equal(cache, before)


def test_borrower_using_autograd_matches_reference():
    tokens = torch.randint(0, 1000, (8, 129), device=dev, generator=torch.Generator(dev).manual_seed(0))
    ref_model, ref_update = make_model()
    ref_losses = [autograd_step(ref_model, ref_update, tokens) for _ in range(3)]
    model, update = make_model()
    losses = [autograd_step(model, update, tokens)]  # outside a loan: creates the AdamW state
    cache, before = make_cache()
    for _ in range(2):
        with memlend.lend(cache) as loan:
            losses.append(autograd_step(model, update, tokens, cache))
    assert losses == ref_losses
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), ref_model.parameters()))
    assert loan.copied_bytes and loan.extra_bytes == 0 and torch.equal(cache, before)


def test_side_streams_and_cublas_workspaces():
    cache, before = make_cache()
    a = torch.randn(1024, 1024, device=dev)
    out = torch.empty(2, 1024, 1024, device=dev)
    with memlend.lend(cache):  # no LeakError: cuBLAS workspaces made for the new streams are released
        for i in range(2):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):  # allocations on this stream wait for the copies of what they overwrite
                tmp = a @ a
                assert from_cache(cache, tmp)
                out[i].copy_(tmp)
            torch.cuda.current_stream().wait_stream(s)
        del tmp
    assert torch.equal(out[0], a @ a) and torch.equal(out[1], a @ a) and torch.equal(cache, before)


def test_leak_is_detected_and_memory_recovers():
    cache, before = make_cache(4)
    kept = []
    with pytest.raises(memlend.LeakError, match=r"1 allocation\(s\) made during the loan are still alive"):
        with memlend.lend(cache):
            kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev))
    assert torch.equal(cache, before)  # restored all the same
    kept.clear()
    with memlend.lend(cache, keep=[]) as loan:  # the leaked memory is lent again
        torch.empty(10 * MiB, dtype=torch.uint8, device=dev)
        torch.empty(cache.numel() - 10 * MiB, dtype=torch.uint8, device=dev)
    assert loan.extra_bytes == 0


def test_cuda_graph_captured_on_the_owners_tensor_stays_valid():
    cache, before = make_cache()
    head, out = cache[0, : 16 * MiB], torch.empty(16 * MiB, dtype=torch.uint8, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(head)
        head.add_(1)
    with memlend.lend(cache, keep=[cache[0]]):
        torch.empty(15 * ROW, dtype=torch.uint8, device=dev).fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, before[0, : 16 * MiB]) and torch.equal(head, before[0, : 16 * MiB] + 1)


def test_owner_kernels_still_running_finish_before_the_loan_starts():
    cache, _ = make_cache()
    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda._sleep(int(5e8))
        cache[15].fill_(42)  # the owner's last write, still queued when it lends
    with memlend.lend(cache) as loan:  # lends the last row first
        torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.copied_bytes == ROW and bool((cache[15] == 42).all())


def test_tensor_dropped_during_another_loan():
    a, _ = make_cache(4)
    b, before = make_cache(4)
    with memlend.lend(a):
        torch.empty(ROW, dtype=torch.uint8, device=dev)
    with memlend.lend(b):
        x = torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1)
        del a  # its pool holds a segment
        del x
    assert torch.equal(b, before)


def test_memory_lives_as_long_as_its_tensor_views():
    free_before = torch.cuda.mem_get_info()[0]
    cache = memlend.empty(1024 * MiB, dtype=torch.uint8)
    view = cache[:10]
    del cache
    assert torch.cuda.mem_get_info()[0] <= free_before - 1024 * MiB
    del view
    assert torch.cuda.mem_get_info()[0] >= free_before - 64 * MiB


@pytest.mark.parametrize(
    "script",
    [
        "try:\n    with memlend.lend(c):\n        kept.append(torch.empty(10 * 1024**2, device='cuda'))\nexcept memlend.LeakError:\n    pass",
        "try:\n    with memlend.lend(c):\n        kept.append(torch.empty(10 * 1024**2, device='cuda'))\nexcept memlend.LeakError:\n    pass\ndel c",
    ],
    ids=["leaked tensor alive at exit", "tensor dropped before its leaked allocation"],
)
def test_interpreter_shutdown(script):
    # Pool and pluggable-allocator destruction order has crashed at exit in some torch versions (pytorch#146431).
    code = "import torch\nimport memlend\nc = memlend.empty(256 * 1024**2, dtype=torch.uint8)\nkept = []\n" + script + "\nprint('done')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "done" in result.stdout, result.stderr[-3000:]
