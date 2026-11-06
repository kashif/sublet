# memlend

Lend a GPU memory region to arbitrary PyTorch code, then learn exactly which blocks it used.
The owner keeps its tensor at a fixed address; while it is paused, any PyTorch code (autograd, several streams, NCCL)
allocates from the region; afterwards every block it did not take is bit-identical, with no copy.

## Install

```bash
pip install git+https://github.com/qgallouedec/memlend.git
pip install .   # from a clone, for development
```

A C compiler (gcc or clang) is needed at install time; no CUDA toolkit.

## Example

```python
import torch
from memlend import LendableRegion

region = LendableRegion(16 * 1024**3)                    # the owner's memory, in 64 MiB blocks
blocks = region.tensor.view(torch.int64).view(region.num_blocks, -1)
blocks.random_()                                         # the owner's data
empty = range(0, region.num_blocks, 2)                   # every other block holds nothing
data = list(range(1, region.num_blocks, 2))
checksums = blocks.sum(1)
backup = torch.empty(region.nbytes, dtype=torch.uint8, pin_memory=True)

def run_anything():                                      # any PyTorch code
    x = torch.randn(1024**3, device="cuda")              # 4 GiB
    return (x * x).mean().item()

with region.lend(empty=empty, backup=backup) as loan:    # data blocks are copied out just before being lent
    run_anything()
region.restore(backup, loan.dirty_blocks)                # copy back only the data blocks that were lent
print(len(loan.touched_blocks), "blocks lent,", len(loan.dirty_blocks), "restored, data intact:",
      torch.equal(blocks.sum(1)[data], checksums[data]))
```

Each allocation goes to the contiguous run of the region that overwrites the fewest data blocks (empty blocks and
blocks already lent in this loan are free), nearest the far end on ties, and to ordinary free GPU memory when the region
has no run long enough. `loan.touched_blocks` lists the lent blocks, `loan.dirty_blocks` those holding data. Anything the
borrower leaves alive in the region raises `LeakError` when the loan ends. `region.save` and `region.restore` copy
blocks to and from a CPU buffer.

## Benchmark

`benchmark.py`: a region taking all free GPU memory but 8 GiB, with a share of its 64 MiB blocks holding data (in 1 GiB
runs), lent to a job that needs most of the GPU (a forward, backward and update pass of a 0.6B-parameter model). Time per
cycle (make room, run the job, get the data back) and GiB copied between GPU and CPU.

| GPU, data in the region | job needs | copy all out and back | discard and recompute | memlend, lend order (2899ec9) | memlend |
|---|---|---|---|---|---|
| H100, 20 of 66 GiB | 15 GiB | 15.4 s (132 GiB) | 1.80 s | 0.12 s (0 GiB) | 0.13 s (0 GiB) |
| H100, 20 of 66 GiB | 59 GiB | 15.9 s (132 GiB) | 1.87 s | OOM | OOM |
| H100, 3 of 66 GiB | 59 GiB | 15.8 s (132 GiB) | 0.91 s | OOM | 0.53 s (2.0 GiB) |
| B200, 49 of 165 GiB | 15 GiB | 6.7 s (330 GiB) | 2.54 s | 0.08 s (0 GiB) | 0.08 s (0 GiB) |
| B200, 49 of 165 GiB | 59 GiB | 6.8 s (330 GiB) | 2.25 s | 0.53 s (18.6 GiB) | 0.28 s (4.6 GiB) |
| B200, 49 of 165 GiB | 117 GiB | 7.1 s (330 GiB) | 2.50 s | 1.13 s (41.4 GiB) | OOM |
| B200, 8 of 165 GiB | 117 GiB | 7.1 s (330 GiB) | 1.01 s | 0.48 s (2.0 GiB) | 0.48 s (1.4 GiB) |

Doing nothing (running the job next to the full region) OOMs in every row. The H100 host link is slow here (7 GB/s out,
14 GB/s in; B200: 57 and 55 GB/s). The "lend order" column is the previous placement, measured with the previous
version of the script on a slightly smaller region. The OOM rows are the contiguity limitation below.

## Compatibility

Tested with torch 2.8 to 2.15 on Python 3.10 to 3.13 (H100; B200 on Python 3.12), with the default allocator settings,
`max_split_size_mb`, and `expandable_segments:True` from torch 2.10 (2.8 and 2.9 reject memory pools under it). Other
torch versions and `backend:cudaMallocAsync` are refused; `MEMLEND_SKIP_TORCH_CHECK=1` bypasses the version check.

## Limitations

- Each allocation needs contiguous pages: a large one can fail (falling back to free GPU memory, or OOM) when the region
  is fragmented, even with enough free pages in total.
- The owner must not touch the region during a loan, and one loan per device runs at a time.
- Memory the borrower keeps after the loan (state created during it, variables still referenced, CUDA graphs captured in
  it on torch before 2.14) raises `LeakError`: create long-lived state before lending.
- Linux and CUDA only. License: not chosen yet.
