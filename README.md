# memlend

Lend the memory of a GPU tensor to any PyTorch code, and get back the parts you keep.
While the tensor's owner is paused, any PyTorch code (autograd, several streams, NCCL, `torch.compile`) allocates from
the tensor's memory. When the loan ends, the parts you keep are back, bit-identical: only the kept memory the borrower
actually used is copied to the CPU and back.

![GPU memory over time: without memlend the job runs out of memory; with memlend it borrows the tensor's memory](assets/lending.svg)

Each column is the GPU memory at one moment. Without memlend, the job fills the free memory above the owner's tensor
and runs out. With memlend, it takes the free memory, then the tensor's pages, whose data is copied to the CPU just
before and back when the loan ends.

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
nothing back. Each allocation of the borrower gets its own addresses, backed by the cheapest memory first: pages of the
tensor that are not kept (wherever they are), then free GPU memory, then kept pages, copied to the CPU just before. The
borrower's memory stays mapped from one loan to the next, so mapping is paid once; the free GPU memory it took is given
back between loans. Anything the borrower leaves alive raises `LeakError` when the loan ends.

## Benchmark

`benchmark.py`: a tensor takes all free GPU memory but 8 GiB, and 30% of it holds data to keep. A job fills one tensor
of 25%, 75% or 100% of that size with random numbers. Next to the full tensor, the job only finds the 8 GiB left, so it
runs out of memory. Time per cycle (make room, run the job, get the data back):

| GPU, tensor | Job | Copy all out and back | memlend, first loan | memlend, later loans |
|---|---|---|---|---|
| H100, 70 GiB | 17 GiB: fits in the free part | 16.3 s | 0.90 s | 0.02 s (0 GiB copied) |
| H100, 70 GiB | 52 GiB: needs the free GPU memory too | 16.5 s | 3.38 s | 0.06 s (0 GiB) |
| H100, 70 GiB | 70 GiB: needs kept data too | 16.6 s | 13.2 s | 3.53 s (29.5 GiB) |
| B200, 169 GiB | 42 GiB | 6.8 s | 1.33 s | 0.03 s (0 GiB) |
| B200, 169 GiB | 126 GiB | 7.0 s | 3.73 s | 0.09 s (0 GiB) |
| B200, 169 GiB | 169 GiB | 7.2 s | 21.4 s | 2.69 s (85.8 GiB) |

Copy all moves the whole tensor out and back (140 and 338 GiB). The first loan maps the job's memory and pins CPU
memory for the kept data it copies; later loans reuse both. The H100 host link is slow here (7 GB/s out, 14 GB/s in;
B200: 57 and 55 GB/s).

## Compatibility

Tested with torch 2.8 to 2.15 on Python 3.10 to 3.13 (B200, two of them for NCCL; H100 on torch 2.8, 2.12 and 2.13),
with the default allocator settings, and with `max_split_size_mb` and `expandable_segments:True` on torch 2.15. Other
torch versions and `backend:cudaMallocAsync` are refused; `MEMLEND_SKIP_TORCH_CHECK=1` bypasses the version check.

## Limitations

- The first loan maps the borrower's memory page by page (about 0.05 s per GiB) and pins CPU memory for the kept data
  it copies (about 0.6 s per GiB on the H100 nodes here).
- Kept data the owner writes where the borrower's memory stays is copied out and back at every loan.
- The owner must not use the tensor during a loan, and one loan per device runs at a time.
- Memory the borrower keeps after the loan (state created during it, variables still referenced, CUDA graphs captured in
  it on torch before 2.14) raises `LeakError`: create long-lived state before lending.
- Linux and CUDA only. License: not chosen yet.
