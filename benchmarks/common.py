"""Shared helpers for the benchmarks: timing, a hand-built transformer, a synthetic borrower, and a pinned buffer."""

import statistics
import time

import torch
import torch.nn as nn

GiB, MiB = 1 << 30, 1 << 20
dev = torch.device("cuda")


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t, out


def median_time(fn, reps):
    return statistics.median(sync_time(fn)[0] for _ in range(reps))


def pinned(nbytes):
    """Pinned host buffer of exactly nbytes (pin_memory=True would round the allocation up to a power of two)."""
    buf = torch.empty(nbytes, dtype=torch.uint8)
    buf.view(torch.int64).zero_()  # fault the pages in before pinning
    torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
    return buf


def link_bandwidth(nbytes=4 * GiB):
    """Host<->device copy bandwidth in GB/s with a pinned buffer."""
    host, device = pinned(nbytes), torch.empty(nbytes, dtype=torch.uint8, device=dev)
    d2h = median_time(lambda: host.copy_(device, non_blocking=True), 3)
    h2d = median_time(lambda: device.copy_(host, non_blocking=True), 3)
    torch.cuda.cudart().cudaHostUnregister(host.data_ptr())
    return nbytes / d2h / 1e9, nbytes / h2d / 1e9


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, s, d))
        return x + self.fc2(nn.functional.gelu(self.fc1(self.norm2(x))))


VOCAB, SEQ = 32000, 2048


def make_model():
    """0.94B-parameter bf16 transformer (d=2048, 16 layers) with AdamW, the phase-1 benchmark model."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(VOCAB, 2048), *[Block(2048, 16) for _ in range(16)], nn.Linear(2048, VOCAB, bias=False))
    model = model.to(dev, torch.bfloat16)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-5)


def train_step(model, opt, batch):
    tokens = torch.randint(0, VOCAB, (batch, SEQ + 1), device=dev)
    logits = model(tokens[:, :-1])
    loss = nn.functional.cross_entropy(logits.float().view(-1, VOCAB), tokens[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def block_checksums(region):
    return region.tensor.view(torch.int64).view(region.num_blocks, -1).sum(dim=1)
