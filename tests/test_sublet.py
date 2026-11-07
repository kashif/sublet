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
        x = torch.empty(8 * ROW, dtype=torch.uint8, device=dev).fill_(1)  # as much as the 8 free rows, scattered
        assert from_cache(cache, x)
        del x
    assert loan.copied_bytes == 0 and loan.extra_bytes == 0
    assert bool((cache[0::2] == 1).all())  # the borrower wrote in the free rows
    assert all(torch.equal(cache[i], before[i]) for i in odd)


def test_free_gpu_memory_goes_before_kept_memory():
    cache, before = make_cache()  # all of it kept
    with gpu_filled(4 * ROW):
        free = torch.cuda.mem_get_info()[0]
        size = (free + 2 * ROW) // (2 * MiB) * (2 * MiB)  # more than the GPU has left
        with sublet.lend(cache) as loan:
            torch.empty(size, dtype=torch.uint8, device=dev).fill_(1)
        assert torch.cuda.mem_get_info()[0] >= free - 16 * MiB  # the GPU memory was given back
    assert loan.extra_bytes + loan.copied_bytes == size and loan.extra_bytes >= free // 2 and loan.copied_bytes >= 2 * ROW
    assert torch.equal(cache, before)


def test_strided_views_keep_only_their_pages():
    cache = sublet.empty(4, 8, 2 * MiB, dtype=torch.uint8)  # 4 layers of 8 blocks of one page
    cache.view(torch.int64).random_()
    before = cache.clone()
    with sublet.lend(cache, keep=cache[:, 3]) as loan:  # block 3 of every layer
        torch.empty(28 * 2 * MiB, dtype=torch.uint8, device=dev).fill_(1)  # as much as the other blocks
    others = [b for b in range(8) if b != 3]
    assert loan.copied_bytes == 0 and loan.extra_bytes == 0 and bool((cache[:, others] == 1).all())
    assert torch.equal(cache[:, 3], before[:, 3])


def test_lend_and_keep_need_views_of_a_lendable_tensor():
    cache, _ = make_cache(2)
    with pytest.raises(ValueError, match="made by sublet.empty"):
        sublet.lend(torch.empty(10, device=dev))
    with pytest.raises(ValueError, match="views of the lent tensor"):
        sublet.lend(cache, keep=[cache[[0, 1]]])  # indexing with a list copies


def test_later_loans_reuse_the_borrowers_memory():
    cache, before = make_cache()
    with sublet.lend(cache, keep=[]):
        torch.empty(4 * ROW, dtype=torch.uint8, device=dev).fill_(1)
    cache.copy_(before)  # the owner writes everywhere, including where the borrower's memory stays
    with sublet.lend(cache) as loan:
        torch.empty(4 * ROW, dtype=torch.uint8, device=dev).fill_(2)
    assert loan.copied_bytes == 4 * ROW and torch.equal(cache, before)


def test_memory_beyond_the_tensor_is_given_back_between_loans():
    cache, _ = make_cache()
    free = torch.cuda.mem_get_info()[0]
    for _ in range(2):  # the second loan takes that memory again
        with sublet.lend(cache, keep=[]) as loan:
            # The fourth takes the last 124 MiB of the tensor and 176 MiB of free GPU memory.
            tensors = [torch.full((300 * MiB,), i + 1, dtype=torch.uint8, device=dev) for i in range(4)]
            assert all(from_cache(cache, t) for t in tensors)
            assert all(bool((t.cpu() == i + 1).all()) for i, t in enumerate(tensors))  # on the CPU: no allocation
            del tensors
        assert loan.extra_bytes == 4 * 300 * MiB - cache.numel()
        assert torch.cuda.mem_get_info()[0] >= free - 16 * MiB


def test_memory_beyond_the_tensor_taken_meanwhile():
    cache, _ = make_cache()
    with sublet.lend(cache, keep=[]):
        tensors = [torch.empty(300 * MiB, dtype=torch.uint8, device=dev) for _ in range(4)]  # 176 MiB beyond the tensor
        del tensors
    with gpu_filled(128 * MiB):  # someone else takes that memory between loans
        with sublet.lend(cache, keep=[]) as loan:  # the borrower's memory starts over, inside the tensor
            torch.empty(600 * MiB, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.extra_bytes == 0


def test_kept_memory_comes_back_after_an_exception():
    cache, before = make_cache()
    with gpu_filled(2 * ROW), pytest.raises(torch.OutOfMemoryError):  # so that the borrower takes kept memory
        with sublet.lend(cache):
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
        with sublet.lend(cache):
            losses.append(autograd_step(model, update, tokens, cache))
    assert losses == ref_losses
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), ref_model.parameters()))
    assert torch.equal(cache, before)


def test_side_streams_and_cublas_workspaces():
    cache, before = make_cache()
    a = torch.randn(1024, 1024, device=dev)
    out = torch.empty(2, 1024, 1024, device=dev)
    with sublet.lend(cache):  # no LeakError: cuBLAS workspaces made for the new streams are released
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
    with pytest.raises(sublet.LeakError, match=r"1 allocation\(s\) made during the loan are still alive"):
        with sublet.lend(cache):
            kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev))
    assert torch.equal(cache, before)  # restored all the same
    kept.clear()
    with sublet.lend(cache):  # freed now: no LeakError
        torch.empty(10 * MiB, dtype=torch.uint8, device=dev)


def test_tensor_dropped_before_its_leaked_allocation():
    cache, _ = make_cache(4)
    kept = []
    with pytest.raises(sublet.LeakError):
        with sublet.lend(cache):
            kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev).fill_(1))
    del cache
    assert bool((kept[0] == 1).all())  # still valid
    kept.clear()
    torch.cuda.empty_cache()  # frees the last segment, then the tensor's memory


def test_cuda_graph_captured_on_the_owners_tensor_stays_valid():
    cache, before = make_cache()
    head, out = cache[0, : 16 * MiB], torch.empty(16 * MiB, dtype=torch.uint8, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(head)
        head.add_(1)
    with sublet.lend(cache, keep=[cache[0]]):
        torch.empty(15 * ROW, dtype=torch.uint8, device=dev).fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, before[0, : 16 * MiB]) and torch.equal(head, before[0, : 16 * MiB] + 1)


def test_owner_kernels_still_running_finish_before_the_loan_starts():
    cache, _ = make_cache()
    with gpu_filled(32 * MiB):  # so that the borrower takes kept memory, from the last row
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(int(5e8))
            cache[15].fill_(42)  # the owner's last write, still queued when it lends
        with sublet.lend(cache) as loan:
            torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.copied_bytes > 0 and bool((cache[15] == 42).all())


def test_tensor_dropped_during_another_loan():
    a, _ = make_cache(4)
    b, before = make_cache(4)
    with sublet.lend(a):
        torch.empty(ROW, dtype=torch.uint8, device=dev)
    with sublet.lend(b):
        x = torch.empty(ROW, dtype=torch.uint8, device=dev).fill_(1)
        del a  # its pool holds a segment
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
        "try:\n    with sublet.lend(c):\n        kept.append(torch.empty(10 * 1024**2, device='cuda'))\nexcept sublet.LeakError:\n    pass",
        "try:\n    with sublet.lend(c):\n        kept.append(torch.empty(10 * 1024**2, device='cuda'))\nexcept sublet.LeakError:\n    pass\ndel c",
    ],
    ids=["leaked tensor alive at exit", "tensor dropped before its leaked allocation"],
)
def test_interpreter_shutdown(script):
    # Pool and pluggable-allocator destruction order has crashed at exit in some torch versions (pytorch#146431).
    code = "import torch\nimport sublet\nc = sublet.empty(256 * 1024**2, dtype=torch.uint8)\nkept = []\n" + script + "\nprint('done')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "done" in result.stdout, result.stderr[-3000:]
