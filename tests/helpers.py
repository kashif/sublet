import torch
import torch.nn as nn

from memlend import LendableRegion

MiB = 1 << 20
BLOCK = 64 * MiB
dev = torch.device("cuda")


def make_region(nbytes, block_size=BLOCK):
    """Region filled with random bytes, plus a copy of that content taken before any loan."""
    region = LendableRegion(nbytes, block_size)
    region.tensor.view(torch.int64).random_()
    return region, region.tensor.clone()


def changed_blocks(region, before):
    """Blocks whose bytes differ from `before` (exact comparison of every byte)."""
    diff = (region.tensor.view(region.num_blocks, -1) != before.view(region.num_blocks, -1)).any(dim=1)
    return diff.nonzero().flatten().tolist()


def assert_untouched_identical(region, before, loan):
    changed = changed_blocks(region, before)
    assert set(changed) <= set(loan.touched_blocks), f"blocks {changed} changed but touched={loan.touched_blocks}"
    untouched = [b for b in range(region.num_blocks) if b not in loan.touched_blocks]
    view, ref = region.tensor.view(region.num_blocks, -1), before.view(region.num_blocks, -1)
    for b in untouched:
        assert torch.equal(view[b], ref[b])
    return changed


def in_region(region, t):
    return region.tensor.data_ptr() <= t.data_ptr() < region.tensor.data_ptr() + region.nbytes


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, 8, d // 8).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-1, -2)) / (d // 8) ** 0.5  # plain matmuls: deterministic, unlike fused attention backward
        scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        attn = scores.softmax(-1) @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(self.norm2(x))


def make_model_and_optimizer():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(1000, 256), *[Block(256) for _ in range(4)], nn.Linear(256, 1000)).to(dev)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def train_step(model, optimizer, tokens, region=None):
    logits = model(tokens[:, :-1])
    loss = nn.functional.cross_entropy(logits.reshape(-1, 1000), tokens[:, 1:].reshape(-1))
    loss.backward()
    if region is not None:  # gradients are allocated by autograd's backward thread
        assert all(in_region(region, p.grad) for p in model.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()
