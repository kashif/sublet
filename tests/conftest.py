import contextlib

import pytest
import torch
import torch.nn as nn

import sublet
from sublet import _compat
from sublet.lend import _regions

MiB = 1024**2
ROW = 64 * MiB
dev = torch.device("cuda")


def pytest_collection_modifyitems(config, items):
    # Under a non-native allocator backend sublet refuses to run: only the test of that refusal applies.
    if torch.cuda.get_allocator_backend() != "native":
        for item in items:
            if "backend" not in item.name:
                item.add_marker(pytest.mark.skip(reason="sublet needs the native allocator backend"))


def make_cache(rows=16, device=dev):
    """A lendable tensor of `rows` rows of 64 MiB of random bytes, and a copy of it."""
    cache = sublet.empty(rows, ROW, dtype=torch.uint8, device=device)
    cache.view(torch.int64).random_()
    return cache, cache.clone()


@contextlib.contextmanager
def gpu_filled(leave):
    """Inside, all free GPU memory but about `leave` bytes is taken."""
    blocker = torch.empty(torch.cuda.mem_get_info()[0] - leave, dtype=torch.uint8, device=dev)
    try:
        yield
    finally:
        del blocker
        torch.cuda.empty_cache()


def from_cache(cache, t):
    """Whether `t` was allocated from the memory `cache` lends."""
    pool = _regions[cache.untyped_storage().data_ptr()].pool
    return any(s["address"] <= t.data_ptr() < s["address"] + s["total_size"] for s in _compat.segments(pool))


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, 8, d // 8).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-1, -2)) / (d // 8) ** 0.5  # plain matmuls: deterministic
        scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        x = x + self.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(self.norm2(x))


def make_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(1000, 256), *[Block(256) for _ in range(4)], nn.Linear(256, 1000)).to(dev)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def autograd_step(model, update, tokens, cache=None):
    """Forward, backward and an AdamW update; returns the loss."""
    logits = model(tokens[:, :-1])
    loss = nn.functional.cross_entropy(logits.reshape(-1, 1000), tokens[:, 1:].reshape(-1))
    loss.backward()
    if cache is not None:  # gradients are allocated by autograd's backward thread
        assert all(from_cache(cache, p.grad) for p in model.parameters())
    update.step()
    update.zero_grad(set_to_none=True)
    return loss.item()
