"""Allocator settings and workloads inside a loan."""

import gc
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import sublet
from conftest import MiB, ROW, dev, from_cache, gpu_filled, make_cache


@pytest.mark.skipif(torch.cuda.get_allocator_backend() == "native", reason="only under backend:cudaMallocAsync")
def test_non_native_backend_is_refused():
    with pytest.raises(RuntimeError, match="native CUDA caching allocator"):
        sublet.empty(64 * MiB, dtype=torch.uint8)


def test_expandable_segments_are_refused():
    code = (
        "import torch, sublet\nc = sublet.empty(64 * 1024**2, dtype=torch.uint8)\ntry:\n    sublet.lend(c)\n"
        "except RuntimeError as e:\n    print(e)\n"
    )
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert "without expandable_segments" in result.stdout, result.stderr[-3000:]


def test_per_process_memory_fraction_counts_lent_memory():
    cache, before = make_cache()
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
    with sublet.lend(cache, keep=cache[8:]):
        torch.empty(100 * MiB, dtype=torch.uint8, device=dev).fill_(1)
        torch.cuda.empty_cache()
        b = torch.empty(300 * MiB, dtype=torch.uint8, device=dev).fill_(2)
        assert from_cache(cache, b)
        del b
    assert torch.equal(cache[8:], before[8:])


def test_leak_report_names_the_allocation_site():
    cache, _ = make_cache(4)
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    try:
        with gpu_filled(8 * MiB), pytest.raises(sublet.LeakError, match="test_workloads.py"):
            with sublet.lend(cache):
                kept = torch.empty(3 * ROW, dtype=torch.uint8, device=dev)  # noqa: F841
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def test_garbage_cycles_are_collected_before_giving_up():
    cache, _ = make_cache(4)

    class Node:
        pass

    gc.disable()
    try:
        with gpu_filled(8 * MiB), sublet.lend(cache):  # no room to move it out: no LeakError either
            a, b = Node(), Node()
            a.other, b.other = b, a
            a.t = torch.empty(3 * ROW, dtype=torch.uint8, device=dev).fill_(1)
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

    with sublet.lend(cache, keep=[]):  # the borrower, its graph's memory included, takes the tensor's memory first
        borrower()
    for i in range(3):
        torch.testing.assert_close(out[i], ((x * (i + 1)) @ (x * (i + 1))).relu())


def test_cuda_graph_captured_inside_loan_and_replayed_after():
    cache, _ = make_cache(4)
    x, graphs = torch.ones(16, device=dev), []

    def borrower():
        with torch.cuda.stream(torch.cuda.Stream()):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                x.mul_(2)
        graphs.append(graph)

    with sublet.lend(cache, keep=[]):  # the graph's memory comes from the tensor, and moves out when the loan ends
        borrower()
    with gpu_filled(32 * MiB), sublet.lend(cache, keep=[]):  # the next loan writes over the whole tensor
        torch.empty(cache.numel(), dtype=torch.uint8, device=dev).fill_(1)
    graphs[0].replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 2.0))


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
