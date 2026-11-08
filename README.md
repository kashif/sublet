# sublet

**Lend GPU memory, offload only what's needed.**

sublet lends the memory of a GPU tensor to any PyTorch code (autograd, several streams, NCCL, `torch.compile`, CUDA
graphs) while the tensor's owner is paused. When the loan ends, the parts you keep are back, bit-identical: only the
kept memory the borrower actually used went to the CPU and back.

![GPU memory over time: copying the tensor out and back, or lending it with sublet](assets/lending.svg)

Each column is the GPU memory at one moment; here the whole tensor holds data to keep. Without sublet, the owner
copies all of it to the CPU to make room, and back after the job. With sublet, the job takes the free GPU memory,
then the tensor's memory it still needs, and only that is copied out and back.

## Install

```bash
pip install sublet
```

A C compiler (gcc or clang) is needed at install time; no CUDA toolkit.

## Example

```python
import torch
import sublet

cache = sublet.empty(16, 1024**3, dtype=torch.uint8)  # 16 GiB, used like any tensor
cache.random_(0, 256)
before = cache[1::2].clone()

def run_anything():                                   # any PyTorch code
    x = torch.randn(256 * 1024**2, device="cuda")     # 1 GiB
    return (x * x).mean().item()

with sublet.lend(cache, keep=cache[1::2]) as loan:    # odd rows hold data, even rows are free
    run_anything()                                    # allocates from the even rows
print(f"copied {loan.copied_bytes} bytes, data intact: {torch.equal(cache[1::2], before)}")
```

`keep` takes views of the tensor (`cache[3]`, `cache[:, 2:5]`, a list of them) and defaults to all of it; `keep=[]` gets
nothing back. When the loan spans functions, `loan = sublet.lend(cache, keep=...)` starts it and `loan.end()` ends it.

During a loan, PyTorch's caching allocator gets the new memory it asks for from sublet, cheapest first: a stretch of
the tensor that holds nothing to keep, free GPU memory, then the stretch with the least kept data, which is copied to
the CPU just before. Before it copies, sublet has PyTorch free the memory it caches, as PyTorch does when the GPU runs
out. Each kept byte is copied at most once per loan. When the loan ends, the memory the borrower freed goes back to the tensor, its allocations still alive (an
optimizer state made by the first step, a loss kept for logging) are moved out of the tensor's memory, and the kept
data is copied back.

## Benchmark

`benchmark.py`: a tensor takes all free GPU memory but 8 GiB, in rows of 1 GiB, and 30% of the rows, at random, hold
data to keep. A job fills tensors of 1 GiB with random numbers, 25%, 75% or 100% of the tensor's size in all: the
first fits in the free rows, the second also needs the free GPU memory, the third also needs kept rows. Next to the
full tensor, the job only finds the 8 GiB left and runs out of memory. Time per cycle (make room, run the job, get the
data back):

![Seconds per cycle on H100 and B200, first and later: copy all out and back, and sublet](assets/benchmark.svg)

Copy all moves the whole tensor out and back; its first cycle also pins that much CPU memory. Until the job needs kept
rows, sublet copies nothing and a cycle is the job itself; at 100%, it copies the kept rows the job lands on (13 GiB on
the H100, 43 GiB on the B200), and its first loan pins that much CPU memory. A GiB to the CPU and back takes 0.23 s
on this H100 node and 0.04 s on the B200.

## Compatibility

Tested with torch 2.6 to 2.16 on Python 3.10 to 3.13 (torch 2.15 has no Python 3.10 build), on T4, A10G, L4, L40S,
A100, H100, H200, B200 and RTX PRO 6000, on two GPUs for NCCL (A10G, H200, B200), and with `max_split_size_mb`. sublet
refuses `expandable_segments` (the caching allocator then maps memory itself, without `cudaMalloc`) and
`backend:cudaMallocAsync`, and torch versions outside the tested range unless `SUBLET_SKIP_TORCH_CHECK=1`.

## Limitations

- An allocation takes one stretch of the tensor's memory: when the tensor's free memory is scattered in stretches
  shorter than an allocation, it takes free GPU memory, or copies kept memory, before it gathers scattered pages.
- sublet refuses `expandable_segments`, and without it PyTorch's caching allocator can strand free memory inside its
  segments: a step that needs nearly all of the GPU can run out of memory where it fits with `expandable_segments`.
- The borrower's allocations still alive when the loan ends need free GPU memory to move to. Without it, the loan ends
  with `LeakError`: the kept data is back all the same, but those allocations no longer hold data.
- The owner must not use the tensor during a loan, and one loan per device runs at a time.
- Linux and CUDA only, with PyTorch's caching allocator calling `cudaMalloc` from `libc10_cuda.so`, as the PyTorch
  wheels do.

## License

[Apache 2.0](LICENSE).
