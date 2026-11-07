"""Allocator settings and workloads inside a loan."""

import gc

import pytest
import torch
import torch.nn as nn

import sublet
from conftest import MiB, dev, from_cache, make_cache
from sublet import _compat


@pytest.mark.skipif(torch.cuda.get_allocator_backend() == "native", reason="only under backend:cudaMallocAsync")
def test_non_native_backend_is_refused():
    with pytest.raises(RuntimeError, match="native CUDA caching allocator"):
        sublet.empty(64 * MiB, dtype=torch.uint8)


def test_per_process_memory_fraction_counts_lent_memory():
    cache, before = make_cache()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction((torch.cuda.memory_reserved() + 256 * MiB) / total)
    try:
        with sublet.lend(cache):
            torch.empty(200 * MiB, dtype=torch.uint8, device=dev).fill_(1)
        with pytest.raises(torch.OutOfMemoryError):
            with sublet.lend(cache):
                torch.empty(400 * MiB, dtype=torch.uint8, device=dev)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert torch.equal(cache, before)
    with sublet.lend(cache):  # usable again after the OOM
        torch.empty(400 * MiB, dtype=torch.uint8, device=dev).fill_(1)
    assert torch.equal(cache, before)


def test_empty_cache_inside_loan():
    cache, before = make_cache()
    with sublet.lend(cache):
        torch.empty(100 * MiB, dtype=torch.uint8, device=dev).fill_(1)
        torch.cuda.empty_cache()
        b = torch.empty(300 * MiB, dtype=torch.uint8, device=dev).fill_(2)
        assert from_cache(cache, b)
        del b
    assert torch.equal(cache, before)


def test_leak_report_names_the_allocation_site():
    cache, _ = make_cache(4)
    kept = []
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    try:
        with pytest.raises(sublet.LeakError, match="test_workloads.py"):
            with sublet.lend(cache):
                kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def test_garbage_cycles_are_collected_before_declaring_a_leak():
    cache, _ = make_cache(4)

    class Node:
        pass

    gc.disable()
    try:
        with sublet.lend(cache):  # no LeakError
            a, b = Node(), Node()
            a.other, b.other = b, a
            a.t = torch.empty(10 * MiB, dtype=torch.uint8, device=dev).fill_(1)
            del a, b
    finally:
        gc.enable()


def test_torch_compile_inside_loan():
    torch._dynamo.reset()
    cache, before = make_cache()
    mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256)).to(dev)
    x = torch.randn(64, 256, device=dev)
    expected, out = mlp(x).detach(), torch.empty(64, 256, device=dev)

    def borrower():  # compile, autotune, forward and backward
        y = torch.compile(mlp)(x)
        y.sum().backward()
        out.copy_(y.detach())
        mlp.zero_grad(set_to_none=True)

    with sublet.lend(cache):
        borrower()
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(cache, before)
    torch._dynamo.reset()


def test_cuda_graph_capture_and_replay_inside_loan():
    cache, before = make_cache()
    x, out = torch.randn(1024, 1024, device=dev), torch.empty(3, 1024, 1024, device=dev)

    def borrower():
        static_in = torch.empty_like(x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            (static_in @ static_in).relu()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = (static_in @ static_in).relu()
        for i in range(3):
            static_in.copy_(x * (i + 1))
            graph.replay()
            out[i].copy_(static_out)

    with sublet.lend(cache):
        borrower()
    for i in range(3):
        torch.testing.assert_close(out[i], ((x * (i + 1)) @ (x * (i + 1))).relu())
    assert torch.equal(cache, before)


def test_cuda_graph_captured_inside_loan_and_kept():
    cache, before = make_cache(4)
    x, graphs = torch.ones(16, device=dev), []

    def borrower():
        with torch.cuda.stream(torch.cuda.Stream()):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                x.mul_(2)
        graphs.append(graph)

    if _compat.torch_version() < (2, 14):
        # The graph's RNG state is allocated outside its pool: it lands in the lent memory and lives as long as the graph.
        with pytest.raises(sublet.LeakError):
            with sublet.lend(cache):
                borrower()
    else:
        with sublet.lend(cache):
            borrower()
        graphs[0].replay()
        torch.cuda.synchronize()
        assert torch.equal(x, torch.full_like(x, 2.0)) and torch.equal(cache, before)


def test_several_tensors_and_loans_in_a_row():
    (c1, before1), (c2, before2) = make_cache(8), make_cache(8)
    for cache in (c1, c2, c1, c2):
        with sublet.lend(cache, keep=cache[:6]) as loan:
            assert from_cache(cache, torch.empty(100 * MiB, dtype=torch.uint8, device=dev).fill_(1))
        assert loan.copied_bytes == 0
    assert torch.equal(c1[:6], before1[:6]) and torch.equal(c2[:6], before2[:6])
    with pytest.raises(RuntimeError, match="already active"):
        with sublet.lend(c1):
            with sublet.lend(c2):
                pass
