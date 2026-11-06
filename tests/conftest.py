import pytest
import torch
import torch.nn as nn

from memlend import LendableRegion

MiB = 1024**2
BLOCK = 64 * MiB
dev = torch.device("cuda")


def pytest_collection_modifyitems(config, items):
    # Under a non-native allocator backend memlend refuses to run: only the test of that refusal applies.
    if torch.cuda.get_allocator_backend() != "native":
        for item in items:
            if "backend" not in item.name:
                item.add_marker(pytest.mark.skip(reason="memlend needs the native allocator backend"))


def make_region(nbytes, block_size=BLOCK):
    """Region filled with random bytes, and a copy of that content."""
    region = LendableRegion(nbytes, block_size)
    region.tensor.view(torch.int64).random_()
    return region, region.tensor.clone()


def changed_blocks(region, before):
    diff = (region.tensor.view(region.num_blocks, -1) != before.view(region.num_blocks, -1)).any(dim=1)
    return diff.nonzero().flatten().tolist()


def assert_untouched_identical(region, before, loan):
    """Every block not reported as touched is bit-identical; returns the blocks that changed."""
    changed = changed_blocks(region, before)
    assert set(changed) <= set(loan.touched_blocks), f"blocks {changed} changed but touched={loan.touched_blocks}"
    return changed


def in_region(region, t):
    return region._lends(t.data_ptr())


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


def autograd_step(model, update, tokens, region=None):
    """Forward, backward and an AdamW update; returns the loss."""
    logits = model(tokens[:, :-1])
    loss = nn.functional.cross_entropy(logits.reshape(-1, 1000), tokens[:, 1:].reshape(-1))
    loss.backward()
    if region is not None:  # gradients are allocated by autograd's backward thread
        assert all(in_region(region, p.grad) for p in model.parameters())
    update.step()
    update.zero_grad(set_to_none=True)
    return loss.item()
