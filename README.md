# memlend

Lend the memory of a GPU tensor to any PyTorch code, and get back the parts you keep.
While the tensor's owner is paused, any PyTorch code (autograd, several streams, NCCL, `torch.compile`) allocates from
the tensor's memory. When the loan ends, the parts you keep are back, bit-identical: only the kept memory the borrower
actually used is copied to the CPU and back.

## Install

```bash
pip install git+https://github.com/qgallouedec/memlend.git
```

A C compiler (gcc or clang) is needed at install time; no CUDA toolkit.

## Example

```python
import torch
import memlend

cache = memlend.empty(16, 1024**3, dtype=torch.uint8)  # 16 GiB, used like any tensor
cache.random_(0, 256)
before = cache[1::2].clone()

def run_anything():                                    # any PyTorch code
    x = torch.randn(512 * 1024**2, device="cuda")      # 2 GiB
    return (x * x).mean().item()

with memlend.lend(cache, keep=cache[1::2]) as loan:   # odd rows hold data, even rows are free
    run_anything()                                     # allocates from the even rows
print(f"copied {loan.copied_bytes} bytes, data intact: {torch.equal(cache[1::2], before)}")
```

`keep` takes views of the tensor (`cache[3]`, `cache[:, 2:5]`, a list of them) and defaults to all of it; `keep=[]` gets
nothing back. Each allocation of the borrower gets its own addresses, backed by pages of the tensor wherever they are:
free pages first, then kept pages, copied to the CPU just before. Beyond the tensor, it uses free GPU memory, given back
when the loan ends. The borrower's memory stays mapped from one loan to the next, so mapping is paid once. Anything the
borrower leaves alive raises `LeakError` when the loan ends.

## Benchmark

`benchmark.py`: a tensor taking all free GPU memory but 8 GiB, with a share of its 64 MiB blocks holding data (in 1 GiB
runs), lent to a job that needs most of the GPU (a forward, backward and update pass of a 0.6B-parameter model). Time per
cycle (make room, run the job, get the data back) and GiB copied between GPU and CPU.

| GPU, data in the tensor | Job needs | Copy all out and back | Recompute | memlend, first loan | memlend, later loans |
|---|---|---|---|---|---|
| H100, 20 of 66 GiB | 15 GiB | 15.5 s (132 GiB) | 1.57 s | 1.11 s | 0.13 s (0 GiB) |
| H100, 20 of 66 GiB | 59 GiB | 15.9 s (132 GiB) | 1.90 s | 16.6 s | 4.52 s (35.4 GiB) |
| H100, 3 of 66 GiB | 59 GiB | 19.3 s (132 GiB) | 0.92 s | 4.25 s | 0.55 s (1.4 GiB) |
| B200, 49 of 165 GiB | 15 GiB | 6.7 s (330 GiB) | 2.12 s | 0.77 s | 0.09 s (0 GiB) |
| B200, 49 of 165 GiB | 59 GiB | 6.9 s (330 GiB) | 2.27 s | 2.40 s | 0.26 s (0 GiB) |
| B200, 49 of 165 GiB | 117 GiB | 7.1 s (330 GiB) | 2.52 s | 8.28 s | 0.91 s (22.3 GiB) |
| B200, 8 of 165 GiB | 117 GiB | 7.1 s (330 GiB) | 1.02 s | 4.32 s | 0.47 s (0 GiB) |

Doing nothing (running the job next to the full tensor) runs out of memory in every row. The first loan maps the job's
memory and pins CPU memory for the kept data it copies; later loans reuse both. The H100 host link is slow here (7 GB/s
out, 14 GB/s in; B200: 57 and 55 GB/s), which is why copying 35 GiB loses to recomputing it in the second row.

## Compatibility

Tested with torch 2.8 to 2.15 on Python 3.10 to 3.13 (H100; B200 on Python 3.12), with the default allocator settings,
`max_split_size_mb`, and `expandable_segments:True` from torch 2.10 (2.8 and 2.9 reject memory pools under it). Other
torch versions and `backend:cudaMallocAsync` are refused; `MEMLEND_SKIP_TORCH_CHECK=1` bypasses the version check.

## Limitations

- The first loan maps the borrower's memory page by page (about 0.05 s per GiB) and pins CPU memory for the kept data
  it copies (about 0.6 s per GiB on the H100 nodes here).
- Kept data the owner writes where the borrower's memory stays is copied out and back at every loan.
- The owner must not use the tensor during a loan, and one loan per device runs at a time.
- Memory the borrower keeps after the loan (state created during it, variables still referenced, CUDA graphs captured in
  it on torch before 2.14) raises `LeakError`: create long-lived state before lending.
- Linux and CUDA only. License: not chosen yet.
