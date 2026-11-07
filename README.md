# sublet

**Lend GPU memory, offload only what's needed.**

sublet lends the memory of a GPU tensor to any PyTorch code (autograd, several streams, NCCL, `torch.compile`) while
the tensor's owner is paused. When the loan ends, the parts you keep are back, bit-identical: only the kept memory the
borrower actually used went to the CPU and back.

![GPU memory over time: copying the tensor out and back, or lending it with sublet](assets/lending.svg)

Each column is the GPU memory at one moment. Without sublet, the owner copies its whole tensor to the CPU to make
room, and back after the job. With sublet, the job takes the free memory, then the tensor's pages it needs, and only
those are copied out and back.

## Install

```bash
pip install git+https://github.com/qgallouedec/sublet.git
```

A C compiler (gcc or clang) is needed at install time; no CUDA toolkit.

## Example

```python
import torch
import sublet

cache = sublet.empty(16, 1024**3, dtype=torch.uint8)   # 16 GiB, used like any tensor
cache.random_(0, 256)
before = cache[1::2].clone()

def run_anything():                                    # any PyTorch code
    x = torch.randn(512 * 1024**2, device="cuda")      # 2 GiB
    return (x * x).mean().item()

with sublet.lend(cache, keep=cache[1::2]) as loan:    # odd rows hold data, even rows are free
    run_anything()                                    # allocates from the even rows
print(f"copied {loan.copied_bytes} bytes, data intact: {torch.equal(cache[1::2], before)}")
```

`keep` takes views of the tensor (`cache[3]`, `cache[:, 2:5]`, a list of them) and defaults to all of it; `keep=[]` gets
nothing back. Each allocation of the borrower gets its own addresses, backed by the cheapest memory first: pages of the
tensor that are not kept (wherever they are), then free GPU memory, then kept pages, copied to the CPU just before. The
borrower's memory stays mapped from one loan to the next, so mapping is paid once; the free GPU memory it took is given
back between loans. Anything the borrower leaves alive raises `LeakError` when the loan ends.

## Benchmark

`benchmark.py`: a tensor takes all free GPU memory but 8 GiB, and 30% of it holds data to keep. A job fills one tensor
of 25%, 75% or 100% of that size with random numbers: the first fits in the free part of the tensor, the second also
needs the free GPU memory, the third also needs kept data. Next to the full tensor, the job only finds the 8 GiB left
and runs out of memory. Time per cycle (make room, run the job, get the data back):

![Seconds per cycle on H100 and B200, first and later: copy all out and back, and sublet](assets/benchmark.svg)

Copy all moves the whole tensor out and back (140 and 338 GiB); its first cycle also pins that much CPU memory. sublet's
first loan maps the job's memory and pins CPU memory only for the kept data it copies; later loans reuse both. The H100
host link is slow here (7 GB/s out, 14 GB/s in; B200: 57 and 55 GB/s).

## Compatibility

Tested with torch 2.8 to 2.15 on Python 3.10 to 3.13 (torch 2.15 has no Python 3.10 build), on T4, A10G, L4, L40S,
A100, H100, H200, B200 and RTX PRO 6000, on two GPUs for NCCL (A10G, H200, B200), and with `max_split_size_mb`. Other
torch versions and `backend:cudaMallocAsync` are refused; `SUBLET_SKIP_TORCH_CHECK=1` bypasses the version check.

## Limitations

- The first loan maps the borrower's memory page by page (about 0.05 s per GiB) and pins CPU memory for the kept data
  it copies (about 0.6 s per GiB on the H100 nodes here).
- Kept data the owner writes where the borrower's memory stays is copied out and back at every loan.
- The owner must not use the tensor during a loan, and one loan per device runs at a time.
- Memory the borrower keeps after the loan (state created during it, variables still referenced, CUDA graphs captured in
  it on torch before 2.14) raises `LeakError`: create long-lived state before lending.
- Linux and CUDA only.

## License

[Apache 2.0](LICENSE).
