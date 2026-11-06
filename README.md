# memlend

Lend a GPU memory region to arbitrary PyTorch code, then learn exactly which blocks it used.

An owner (an inference engine's KV cache, an embedding table, any big buffer that must survive) keeps its tensor at a
fixed address. While it is paused, any PyTorch code (a training step, autograd, cuBLAS, several streams, NCCL) can
allocate from that memory. Afterwards the owner learns which blocks were handed out: every other block is bit-identical
without a single copy, and only the blocks that held data and were used need to be restored.

```python
import torch
from memlend import LendableRegion

region = LendableRegion(40 << 30)                       # the owner's memory: region.tensor, 40 GiB, fixed address
backup = torch.empty(region.nbytes, dtype=torch.uint8, pin_memory=True)
with region.lend(empty=free_blocks, backup=backup) as loan:  # blocks holding nothing go first and are never copied
    train_step()                                         # any PyTorch code; its CUDA allocations may use the region
region.restore(backup, loan.dirty_blocks)               # copy back only what held data and was used
print(len(loan.touched_blocks), "blocks lent,", len(loan.dirty_blocks), "restored")
```

## Install

```bash
pip install .   # from this directory
```

Requirements: Linux, an NVIDIA GPU and driver, PyTorch 2.8 to 2.15 with its default ("native") CUDA allocator (with `expandable_segments`, PyTorch 2.10 or newer), and a
C compiler (gcc or clang) at install time to build the small allocator in `src/memlend/_allocator.c`. No CUDA toolkit
is needed: the CUDA driver is opened at runtime. No dependency beyond torch.

The library refuses to run (with a clear error) on torch versions outside the tested range and under
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`. `MEMLEND_SKIP_TORCH_CHECK=1` bypasses the version check, for trying
a new release after running the test suite.

## API

- `LendableRegion(nbytes, block_size=64 << 20, device=None)`: allocates the region. `region.tensor` is a flat `uint8`
  CUDA tensor over it, valid for as long as any view of it lives. `block_size` (a multiple of 2 MiB) is the unit of
  everything below.
- `region.set_lend_order(blocks)`: these blocks are lent first, in this order, then the rest from the far end.
- `region.lend(empty=(), backup=None)`: context manager. Inside it, every CUDA allocation on the region's device, from
  any thread and stream, may be served from the region, then from ordinary free GPU memory once the region is full.
  - `empty`: blocks that hold nothing the owner needs. They are lent before anything else and never backed up.
  - `backup`: a pinned CPU `uint8` tensor of `nbytes`. Every other block is copied into it right before it is first
    lent (lazy backup), ordered on the borrower's stream so the host never waits.
- After the loan: `loan.touched_blocks` (lent, so overwritten), `loan.dirty_blocks` (touched and not declared empty: the
  ones to restore), `loan.backed_up_blocks`, `loan.fallback_peak_bytes` (memory taken outside the region).
- `region.save(backup, blocks=None)` and `region.restore(backup, blocks=None)`: copy blocks out to / back from a CPU
  buffer, one copy per contiguous run.
- `memlend.LeakError`: raised at the end of a loan if anything the borrower allocated is still alive in lent memory.

## Rules for borrowers

- Everything allocated during the loan must be freed before it ends: run the borrower in a function (variables of a
  `with` block outlive it), `zero_grad(set_to_none=True)`, and create long-lived state (optimizer states, CUDA graphs,
  NCCL communicators) before lending. Anything left behind raises `LeakError` with its size and, if
  `torch.cuda.memory._record_memory_history()` is on, where it was allocated.
- The owner must not touch the region during a loan; the loan waits for the owner's pending kernels when it starts and
  for the borrower's when it ends.
- One loan per device at a time. Several regions per device are fine; carve several buffers out of one region to lend
  them together.

See `DESIGN.md` for how it works, the compatibility matrix and the limitations, and `BENCHMARKS.md` for when it pays
off.

## License

No license has been chosen yet: that is the author's decision.
