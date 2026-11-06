# memlend: design

## Problem

A colocated RL trainer shares one GPU between a generation engine and a training step. The engine owns a large
preallocated tensor (its KV cache, tens of GB) whose content must survive the training step. Today the whole tensor
is copied to pinned CPU memory, freed, then reallocated and copied back after training: about 5 s per step for
115 GB at 55 GB/s, and the reallocated tensor can land at a different address, which breaks anything that captured
the old one (CUDA graphs).

Training usually needs only part of that memory. What is missing is a general mechanism: an owner lends a region to
arbitrary PyTorch code (autograd, cuBLAS workspaces, several streams, tensors freed and reallocated), the borrower can
also use ordinary free GPU memory, and afterwards the owner learns exactly which blocks were handed out. Every other
block is bit-identical without any copy, the owner's tensor keeps its address, and anything the borrower leaves alive
in lent memory is reported as an error. The owner is paused during the loan.

## Options considered

Numbers in this section come from phase-1 runs on one H100 80GB HBM3, torch 2.13.0+cu130, driver 595.91.07. Logs are
in `experiments/`. Allocator behavior was read from the v2.13.0 sources (`_ref/`) and then checked by experiment.

### 1. `MemPool` + `CUDAPluggableAllocator` serving sub-ranges of the region (chosen)

During the loan, every allocation on the device goes to a private `MemPool` whose segments come from a small C
allocator. The caching allocator asks that allocator for whole segments (always multiples of 2 MiB: 2 MiB for
requests up to 1 MiB, 20 MiB for 1 to 10 MiB, above that the size rounded up to 2 MiB), then splits, caches and reuses
them inside the pool as usual. The C allocator serves each segment from a contiguous run of free 2 MiB pages of the
region, picked in lend order, and falls back to `cuMemAlloc` when no run is long enough. Touched blocks are recorded
when a run is handed out, so tracking is exact and costs nothing per tensor.

What had to be checked (`experiments/exp1_routing.py`, log `exp1_routing.log`):

- `torch.cuda.use_mem_pool` only routes the current thread. Autograd runs backward on its own device thread, so
  parameter gradients landed in the default pool `(0, 0)`, not in ours. Its own docstring says so.
- `torch._C._cuda_beginAllocateToPool(device, pool_id)` installs a filter that accepts every stream from every
  thread: the forward output, the gradients made by the backward thread and a tensor allocated on a side stream all
  landed in our pool.
- cuBLAS caches one workspace per (handle, stream) for the life of the process, allocated through the caching
  allocator. A matmul on a new stream during the loan left a 32 MiB block alive in the pool after routing ended;
  `torch._C._cuda_clearCublasWorkspaces()` releases it, after which the pool returned every segment (16 mallocs,
  16 frees). Without that call, `test_side_streams_and_cublas_workspaces` fails with a 2 x 32 MiB leak.
- `MemPool(use_on_oom=True)` would give "free memory first, pool on OOM" without any routing, but in 2.13
  `try_mempool_fallback` only reuses free blocks already cached in the pool and never asks the pool's allocator for
  a segment: a 2 GiB allocation on a full GPU raised OOM with zero calls to our allocator.
- Freeing: segments of a private pool are only returned to the allocator when the pool is freeable (use count 0),
  so the loan creates a fresh pool, and at exit drops its last reference; the `MemPool` destructor then calls
  `emptyCache(pool_id)`, which hands every free segment back. A segment that still holds a live block stays, which
  is how leaks are detected.
- Before torch 2.8 (read in the 2.6.0 and 2.7.1 sources, then confirmed by the matrix), a pool's custom allocator is used
  only on threads where a thread-local `MemPoolContext` is active, so allocations routed to the pool from autograd's
  backward thread got plain `cudaMalloc` segments. From 2.8 the allocator belongs to the pool. memlend requires 2.8.
- The region must not come from the caching allocator: it indexes live allocations by pointer, so a borrower block
  starting at the region's first byte would collide with the region tensor's own entry (from reading the allocator
  source, not reproduced). The region is raw `cuMemAlloc` memory exposed to PyTorch through
  `__cuda_array_interface__`; `test_region_memory_lives_as_long_as_its_tensor_views` checks it is freed exactly when
  the last view dies.

### 2. CUDA virtual memory management

- Unmapping the owner's chunks to lend them, then remapping at the same address: the physical memory is released,
  so every lent chunk loses its content and must be restored whether the borrower used it or not, and the amount to
  lend has to be decided before the borrower runs. PyTorch has no hook to unmap more chunks when the default pool runs
  out (`use_on_oom` pools do not allocate, see above).
- Aliasing: keep the owner's mapping, and for each segment the borrower needs, map free physical chunks of the region
  at a new virtual range. This honors the lend order exactly (any chunks, no contiguity), but it is slow
  (`experiments/vmm_timing.c`, `vmm_breakdown.c`): `cuMemSetAccess` costs ~94 us and `cuMemUnmap` ~48 us per 2 MiB
  chunk (2048 chunks: 192.5 ms and 98.9 ms; `cuMemMap` is ~1.3 us). A 1 GiB segment costs 74.5 ms to map, set access
  and unmap, against 2.0 ms for `cuMemAlloc` + `cuMemFree`, and every loan pays it again. With 64 MiB chunks it drops
  to 6.0 + 3.1 ms per 4 GiB, but then every 2 MiB small-pool segment pins a whole 64 MiB chunk. Not worth it for now;
  see the lend-order limitation below.

### 3. Free N trailing chunks, or expandable segments (`experiments/exp3_baselines.py`)

The owner splits the region into 8 x 1 GiB tensors, frees the last 4 and empties the cache, the borrower runs, then
the owner reallocates. With the plain allocator the reallocated chunks kept their address only for [yes, yes, no, no]
(borrower using the same sizes), [yes, yes, no, no] (borrower keeping a 64 MiB tensor) and [yes, yes, yes, no]
(different sizes); with `expandable_segments:True`, [yes, yes, yes, no] in all three. In every case all 4 lent chunks
lost their content, including the one the borrower never needed. So: no stable address, full restore of everything
lent, amount fixed up front.

### 4. Free into PyTorch's cache, reallocate, checksum (`experiments/exp4_cache_and_checksum.py`)

The owner `del`s its 8 GiB tensor without emptying the cache, the borrower reuses the cached memory, the owner
allocates 8 GiB again and checksums 64 MiB blocks to find what changed. In the happy path it works: same address and
50 of 128 blocks changed for 3.1 GiB written. As soon as anything calls `empty_cache()` (trainers do; the allocator
itself does on its OOM retry path), the segment goes back to the driver and 128 of 128 blocks changed. No lend order
(the borrower packs from the start), no guarantee of getting the same block back, no leak detection. Checksumming is
affordable, 17.6 ms for a per-block int64 sum of 50 GiB, but with allocator-level tracking it is not needed; the tests
use exact comparisons instead.

## Chosen design

Package layout (`pip install .`, see README.md):

- `src/memlend/region.py`: `LendableRegion`, `Loan`, `LeakError`.
- `src/memlend/_compat.py`: every private PyTorch call, behind a version check (see Compatibility).
- `src/memlend/_allocator.c` (~260 lines): the segment allocator. It opens the CUDA driver at runtime with `dlopen`, so
  building it needs only a C compiler; `setup.py` compiles it at install time and turns a compiler failure into a
  message that says what is missing.

How a loan works:

- `LendableRegion(nbytes, block_size)` `cuMemAlloc`s the region outside PyTorch's caching allocator and exposes it as
  `region.tensor` (flat `uint8`); the memory is freed when the last view of that tensor dies. The caching allocator
  indexes live allocations by pointer, so the region must not be one of its allocations.
- Pages are 2 MiB (the caching allocator's segment granularity); blocks (`block_size`, a multiple of 2 MiB) are the unit
  of the lend order, of `touched_blocks` and of backups.
- Entering a loan synchronizes the device (the owner's last kernels may still use the region), builds this loan's lend
  order (`empty` blocks first, then `set_lend_order`'s order, by default the far end first), creates a `MemPool` backed
  by the C allocator and routes every allocation on the device to it, from every thread and stream
  (`torch._C._cuda_beginAllocateToPool`).
- The caching allocator then asks the C allocator for segments. Each is served from the shortest prefix of the lend
  order that contains a long enough run of contiguous free pages, or from `cuMemAlloc` (ordinary free GPU memory) when
  the region has none. The blocks a run overlaps are marked touched.
- Lazy backup (`backup=`): the first time a block that is not `empty` is handed out, it is copied to the pinned backup
  on a side stream, an event is recorded, and the stream that asked for the memory waits on that event
  (`cuStreamWaitEvent`). The host never blocks and the borrower's kernels on that stream simply follow the copy. Under
  PyTorch's stream semantics, a tensor used on another stream than the one it was allocated on must be synchronized
  with that stream anyway, so ordering on the allocating stream is enough (`test_lazy_backup_with_allocations_on_side_streams`;
  removing the wait makes the four stream-mode lazy tests fail).
- Exiting synchronizes the device (the borrower's kernels may still be writing), ends the routing, releases cuBLAS's
  per-stream workspaces (cached for the life of the process), lists the pool's blocks still alive in the region,
  collects garbage cycles and lists them again if there were any, drops the pool (its free segments go back to the C
  allocator) and raises `LeakError` if anything is still alive in the region.
- The C allocator keeps one lender per device, so loans on different devices can run at the same time. A region dropped
  by its owner while a leaked borrower tensor still lives in it is freed when that tensor goes.

### Lazy backup: what was kept

Measured in `benchmarks/bench_lazy_backup.py` (BENCHMARKS.md, "Lazy backup"), against copying blocks up front:

- Declaring empty blocks (`empty=`) is the change that matters: a borrower that fits in them moves zero bytes, so the
  backup costs nothing (H100 and B200: the loan takes exactly as long as without any backup). It is the documented
  default path.
- Ordering each copy on the borrower's stream instead of making the host wait saves 1% on H100 (977 vs 988 ms per loan
  at batch 2 with every block valuable; the 7 GB/s link is the bottleneck) and 4% on B200 (136 vs 142 ms). It is kept:
  it is never slower and frees the host.
- Prefetching the next blocks in lend order never pays off: with 8 blocks ahead it is within noise (at best 44 vs 48 ms on
  B200 when a few blocks spill), with 32 it is always slower because it copies blocks the borrower never takes (318 vs
  58 ms on H100 with half the region empty). It is only reachable through the private `_prefetch` argument, kept so the
  comparison can be rerun, as is `_sync_copies`.

## API

```python
from memlend import LendableRegion, LeakError

region = LendableRegion(nbytes, block_size=64 << 20, device=None)  # region.tensor: the owner's memory, fixed address
region.set_lend_order(blocks)            # optional: these first, then the rest from the far end
with region.lend(empty=(), backup=None) as loan:  # empty: blocks holding nothing; backup: pinned CPU buffer
    run_anything()
loan.touched_blocks      # lent: everything else is bit-identical, with no copy
loan.dirty_blocks        # touched and not declared empty: the blocks to restore
loan.backed_up_blocks    # copied into the lazy backup
loan.fallback_peak_bytes # peak memory taken outside the region
region.save(backup, blocks=None)      # eager backup helper
region.restore(backup, blocks=None)   # restore helper
```

Owners with several buffers carve them out of one region (one loan per device at a time); owners on several devices
create one region per device, and loans on different devices can be active together.

## Compatibility

| torch | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
|---|---|---|---|---|
| 2.15.0+cu132 | no wheel | 35 passed | 35 passed | 35 passed |
| 2.14.1+cu130 | 35 passed | 35 passed | 35 passed | 35 passed |
| 2.13.0+cu130 | 35 passed | 35 passed | 35 passed | 35 passed |
| 2.12.1+cu130 | 35 passed | 35 passed | 35 passed | 35 passed |
| 2.11.0+cu130 | 35 passed | 35 passed | 35 passed | 35 passed |
| 2.10.0+cu128 | 35 passed | 35 passed | 35 passed | 35 passed |
| 2.9.1+cu128 | 35 passed | 35 passed | 35 passed | 35 passed |
| 2.8.0+cu128 | 35 passed | 35 passed | 35 passed | 35 passed |

H100, `matrix/results.tsv`, logs in `matrix/logs/`. Every supported version passes the full suite on every Python build
it ships; torch 2.15 has no Python 3.10 wheel. Outside 2.8 to 2.15 the library refuses to run with a clear error:

- torch 2.6 and 2.7: refused. Forced with `MEMLEND_SKIP_TORCH_CHECK=1`, 33 of 35 tests fail, because there the pool's
  custom allocator only applies on threads with a thread-local context, so autograd's backward thread bypasses the
  region. Unsupported by design.
- torch 2.16 nightly (2.16.0.dev20261102): refused by the version check; forced, 35 passed.

## Allocator settings and workloads inside a loan

| torch (Python 3.12) | expandable_segments:True | max_split_size_mb:128 | backend:cudaMallocAsync |
|---|---|---|---|
| 2.15 | 35 passed | 35 passed | refused (1 passed) |
| 2.14 | 35 passed | 35 passed | refused (1 passed) |
| 2.13 | 35 passed | 35 passed | refused (1 passed) |
| 2.12 | 35 passed | 35 passed | refused (1 passed) |
| 2.11 | 35 passed | 35 passed | refused (1 passed) |
| 2.10 | 35 passed | 35 passed | refused (1 passed) |
| 2.9 | not supported by PyTorch | 35 passed | refused (1 passed) |
| 2.8 | not supported by PyTorch | 35 passed | refused (1 passed) |

On torch 2.8 and 2.9, PyTorch itself rejects a MemPool under expandable segments ("torch.cuda.MemPool doesn't currently
support expandable_segments"), so every loan fails with that error; from 2.10 on it works. Under `cudaMallocAsync` the
only test that runs is the one checking that memlend refuses that backend with a clear error. The workload tests
(torch.compile, CUDA graphs captured and replayed inside a loan, autocast bf16, side streams, a DataLoader with pinned
memory, gc mid-loan, several loans and regions in a row) are part of the 35 and pass in every passing cell.

## Multi-GPU and NCCL

On 2 B200 GPUs (`tests/distributed/`, logs `benchmarks/results/b200_nccl_{lazy,eager}.log`, `b200_distributed.log`):
all_reduce, all_gather_into_tensor, broadcast and an async all_reduce on 64 MiB tensors allocated in the region, over
three loans each, give correct results with untouched blocks identical, both when the communicator already exists
(`eager`: `init_process_group(device_id=...)` plus a warm-up collective) and when the first collective of the
communicator happens inside the loan (`lazy`). Collectives captured in a CUDA graph also work: a graph captured before
any loan replays correctly inside and after loans, and a graph captured and replayed inside a loan works.
`test_concurrent_loans_on_two_devices` runs loans on both GPUs of one process at once.

Behaviour of the first collective inside a loan, decided and documented: NCCL allocates its communicator buffers
itself, through the driver, not through PyTorch's caching allocator, so they never come from the region and are not
leaks: they took 1094 MiB of ordinary free GPU memory per GPU here (the "free memory outside the region" delta of the
first lazy loan; 4 MiB in eager mode, whose buffers already existed). The only risk is running out of memory outside
the region at that moment, so the recommendation is to create communicators before lending
(`init_process_group(device_id=...)` and one warm-up collective).

Not tested: more than 2 GPUs, multi-node, NCCL user-buffer registration (`ncclMemAlloc` pools,
`TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK`), and the 2-GPU tests on H100 (the H100 jobs had one GPU).

## Known pitfalls and how they are covered

1. Memory still in use when it changes hands (the caching allocator frees a block on the host before its kernels
   finish). Entering a loan synchronizes the device; leaving it synchronizes the device before anything is released.
   `test_owner_kernels_still_running_finish_before_the_loan_starts` (the owner's last write is still queued behind a
   250 ms kernel when it lends; the lazy backup must hold that write) fails when the entry synchronization is removed.
   `test_borrower_kernels_still_running_finish_before_the_loan_ends` (a borrower write still queued when the loan ends,
   then the owner overwrites the region) passes, but it also passed with the exit synchronization removed: another call
   on the exit path already waits, which I did not pin down, so that test does not prove the explicit synchronization
   is needed. memlend never unmaps or remaps memory, so the `cuMemUnmap` race vLLM hit cannot happen.
2. NCCL buffer registration bound to physical pages (torch_memory_saver#88). memlend never replaces the physical pages
   behind an address, so registrations stay valid; graph-captured collectives before and inside loans pass (above).
3. PyTorch bugs around pluggable-allocator pools: expandable segments (pytorch#147851) is a column of the settings
   matrix on every supported version; `empty_cache()` with a pluggable allocator (pytorch#145168) is exercised inside a
   loan (`test_empty_cache_inside_loan`) and on a freeable pool still holding leaked segments
   (`test_leak_is_detected_and_memory_recovers`); destruction order at exit ("pure virtual method called",
   pytorch#145168, #146431) is `test_interpreter_shutdown` (clean exit, a leaked tensor alive at exit, the region dropped
   before its leaked tensor, an exception inside a loan), each in a fresh interpreter, on every version.
4. ROCm returning stale memory on remap: out of scope. memlend never remaps, and it only supports CUDA (the allocator
   calls the CUDA driver API).
5. Discard and recompute: both examples in BENCHMARKS.md include it, and the KV-cache example is also run with short
   contexts (5% of the cache live), where recompute is cheapest.

## Limitations

- Contiguity. Each segment the caching allocator asks for needs contiguous pages. The lend order is honored per segment
  ("the shortest lend-order prefix with a long enough free run"), so a large segment over scattered empty blocks has to
  include valuable ones (phase 1: with 60% of a 50 GiB region empty but scattered, 17 of 186 touched blocks held data),
  and a large request can fail although enough pages are free: in the KV-cache example on H100, a Qwen3-0.6B step at
  batch 8 (58.6 GiB peak) OOMs inside a 64 GiB region + 8 GiB headroom because its 9.27 GiB fp32 logits found no
  contiguous run. Aliasing chunks at new addresses (CUDA VMM) would lift this but costs ~75 ms per GiB per loan.
- No defragmentation inside a loan: the caching allocator does not release cached free segments of a pool in use to
  satisfy a larger request.
- Region first: the borrower takes region memory before ordinary free memory.
- Everything allocated on the device by any thread during a loan goes to the loan: the owner must really be paused.
- Long-lived state created during a loan raises `LeakError`: optimizer states, a `GradScaler` scale, `.grad` not set to
  `None`, tensors still referenced when the `with` block ends, and CUDA graphs captured during the loan that outlive it
  (PyTorch allocates a graph's RNG state outside the graph's pool, so it lands in the region; this includes
  `torch.compile(mode="reduce-overhead")` graphs first recorded inside a loan). Graphs captured before lending can be
  replayed inside a loan.
- cuBLAS workspaces are cleared at every loan exit, process-wide; they are recreated on the next matmul (32 MiB each).
- Region segments count against `torch.cuda.set_per_process_memory_fraction`, like any PyTorch allocation.
- One loan per device at a time. To lend several buffers together, carve them out of one region.
- Each loan starts with an empty pool, so the borrower's segments are requested again every loan (free for region pages,
  a `cudaMalloc` for fallback ones).
- PyTorch 2.8 or newer: before 2.8 a pool's custom allocator is used only on threads where a thread-local
  `MemPoolContext` is set, so autograd's backward thread would bypass the region.
- CUDA only; Linux only.
