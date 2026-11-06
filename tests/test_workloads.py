"""Allocator settings and workloads inside a loan."""

import gc

import pytest
import torch
import torch.nn as nn

from conftest import MiB, assert_untouched_identical, dev, in_region, make_region
from memlend import LeakError, LendableRegion, _compat


@pytest.mark.skipif(torch.cuda.get_allocator_backend() == "native", reason="only under backend:cudaMallocAsync")
def test_non_native_backend_is_refused():
    with pytest.raises(RuntimeError, match="native CUDA caching allocator"):
        LendableRegion(64 * MiB)


def test_per_process_memory_fraction_counts_region_segments():
    region, before = make_region(1024 * MiB)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction((torch.cuda.memory_reserved() + 256 * MiB) / total)
    try:
        with region.lend() as loan:
            torch.empty(200 * MiB, dtype=torch.uint8, device=dev).fill_(1)
        with pytest.raises(torch.OutOfMemoryError):
            with region.lend():
                torch.empty(400 * MiB, dtype=torch.uint8, device=dev)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert_untouched_identical(region, before, loan)
    with region.lend() as loan:  # usable again after the OOM
        torch.empty(400 * MiB, dtype=torch.uint8, device=dev).fill_(1)
    assert loan.fallback_peak_bytes == 0


def test_empty_cache_inside_loan():
    region, before = make_region(1024 * MiB)
    with region.lend() as loan:
        torch.empty(100 * MiB, dtype=torch.uint8, device=dev).fill_(1)
        torch.cuda.empty_cache()  # leaves the loan's pool, still in use, alone
        b = torch.empty(300 * MiB, dtype=torch.uint8, device=dev).fill_(2)
        assert in_region(region, b)
        del b
    assert_untouched_identical(region, before, loan)


def test_leak_report_names_the_allocation_site():
    region, _ = make_region(256 * MiB)
    kept = []
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    try:
        with pytest.raises(LeakError, match="test_workloads.py"):
            with region.lend():
                kept.append(torch.empty(10 * MiB, dtype=torch.uint8, device=dev))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def test_garbage_cycles_are_collected_before_declaring_a_leak():
    region, _ = make_region(256 * MiB)

    class Node:
        pass

    gc.disable()
    try:
        with region.lend() as loan:
            a, b = Node(), Node()
            a.other, b.other = b, a
            a.t = torch.empty(10 * MiB, dtype=torch.uint8, device=dev).fill_(1)
            del a, b
    finally:
        gc.enable()
    assert loan.touched_blocks


def test_torch_compile_inside_loan():
    torch._dynamo.reset()
    region, before = make_region(1024 * MiB)
    mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256)).to(dev)
    x = torch.randn(64, 256, device=dev)
    expected, out = mlp(x).detach(), torch.empty(64, 256, device=dev)

    def borrower():  # compile, autotune, forward and backward
        y = torch.compile(mlp)(x)
        y.sum().backward()
        out.copy_(y.detach())
        mlp.zero_grad(set_to_none=True)

    with region.lend() as loan:
        borrower()
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    assert loan.touched_blocks
    assert_untouched_identical(region, before, loan)
    torch._dynamo.reset()


def test_cuda_graph_capture_and_replay_inside_loan():
    region, before = make_region(1024 * MiB)
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

    with region.lend() as loan:
        borrower()
    for i in range(3):
        torch.testing.assert_close(out[i], ((x * (i + 1)) @ (x * (i + 1))).relu())
    assert_untouched_identical(region, before, loan)


def test_cuda_graph_captured_inside_loan_and_kept():
    region, before = make_region(256 * MiB)
    x, graphs = torch.ones(16, device=dev), []

    def borrower():
        with torch.cuda.stream(torch.cuda.Stream()):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                x.mul_(2)
        graphs.append(graph)

    if _compat.torch_version() < (2, 14):
        # The graph's RNG state is allocated outside its pool: it lands in the region and lives as long as the graph.
        with pytest.raises(LeakError):
            with region.lend():
                borrower()
    else:
        with region.lend() as loan:
            borrower()
        graphs[0].replay()
        torch.cuda.synchronize()
        assert torch.equal(x, torch.full_like(x, 2.0))
        assert_untouched_identical(region, before, loan)


def test_several_regions_and_loans_in_a_row():
    (r1, before1), (r2, before2) = make_region(512 * MiB), make_region(512 * MiB)
    loans = []
    for region in (r1, r2, r1, r2):
        with region.lend() as loan:
            assert in_region(region, torch.empty(100 * MiB, dtype=torch.uint8, device=dev).fill_(1))
        loans.append(loan)
    assert [loan.touched_blocks for loan in loans] == [[6, 7]] * 4
    assert_untouched_identical(r1, before1, loans[2])
    assert_untouched_identical(r2, before2, loans[3])
    with pytest.raises(RuntimeError, match="already active"):
        with r1.lend():
            with r2.lend():
                pass
