"""Example 1: an inference engine's KV cache lent to a training step (colocated RL).

Owner: a KV-cache-like region taking most of the GPU (free memory minus --headroom-gib), in 64 MiB blocks; a share of
it (--live) holds live sequences, in 1 GiB runs spread over the region, the rest is empty. Borrower: one training step
(forward, backward, AdamW) of Qwen3-0.6B on batch x 2048 tokens. One cycle = whatever the owner does to make room +
the training step + getting its KV cache back.

Strategies:
- do-nothing: train next to the full KV cache; OOMs when the step needs more than the headroom.
- copy-all: copy the whole cache to pinned CPU memory, free it, train, reallocate it, copy it back (today's approach).
- recompute: free the cache, train, reallocate it, and recompute the live sequences with a prefill forward pass over
  the same number of tokens (the KV writes themselves are not included, so this is a lower bound).
- memlend-eager: copy the live blocks out, lend with `empty=` the empty blocks, restore the dirty live blocks.
- memlend-lazy: lend with `empty=` and a lazy backup (a live block is copied out only right before it is lent).

    python benchmarks/bench_kv_cache.py --out benchmarks/results/kv_cache_<gpu>.jsonl > benchmarks/results/kv_cache_<gpu>.log
"""

import argparse
import json
import random
import statistics

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from common import GiB, MiB, block_checksums, dev, link_bandwidth, pinned, sync_time
from memlend import LendableRegion

MODEL = "Qwen/Qwen3-0.6B"
SEQ = 2048
BLOCK = 64 * MiB


def train_step(model, opt, batch):
    ids = torch.randint(0, model.config.vocab_size, (batch, SEQ), device=dev)
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


@torch.inference_mode()
def prefill(model, tokens):
    """Forward passes over `tokens` tokens in chunks of 8 x 2048, the compute needed to rebuild their KV cache."""
    for _ in range(max(1, tokens // (8 * SEQ))):
        model.model(input_ids=torch.randint(0, model.config.vocab_size, (8, SEQ), device=dev), use_cache=False)  # no LM head: prefill needs no logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom-gib", type=int, default=8)
    parser.add_argument("--live", type=float, default=0.3)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 8, 16])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    d2h_bw, h2d_bw = link_bandwidth()
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | pinned copies: D2H {d2h_bw:.1f} GB/s, H2D {h2d_bw:.1f} GB/s", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    train_step(model, opt, 1)  # optimizer states and lazy CUDA state before any loan
    prefill(model, 8 * SEQ)
    cfg = AutoConfig.from_pretrained(MODEL)
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
    peaks = {}
    for b in list(args.batches):  # memory each step needs, measured with nothing else on the GPU
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            train_step(model, opt, b)
            peaks[b] = (torch.cuda.max_memory_allocated() - base) / GiB
        except torch.OutOfMemoryError:
            print(f"batch {b} does not fit on this GPU even alone: skipped", flush=True)
            args.batches.remove(b)
    torch.cuda.empty_cache()

    nbytes = (torch.cuda.mem_get_info()[0] - args.headroom_gib * GiB) // BLOCK * BLOCK
    n = nbytes // BLOCK
    rng = random.Random(0)
    runs = list(range(n // 16))
    live_runs = set(rng.sample(runs, round(len(runs) * args.live)))
    live = [b for b in range(n) if b // 16 in live_runs]
    empty = [b for b in range(n) if b // 16 not in live_runs]
    live_tokens = len(live) * BLOCK // kv_bytes_per_token
    print(
        f"KV cache {nbytes / GiB:.1f} GiB ({n} blocks), live {len(live) * BLOCK / GiB:.1f} GiB = {live_tokens} tokens "
        f"at {kv_bytes_per_token} B/token; headroom {args.headroom_gib} GiB; step peak memory: "
        + ", ".join(f"batch {b}: {p:.1f} GiB" for b, p in peaks.items()),
        flush=True,
    )
    host = pinned(nbytes)

    def record(**row):
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Baselines on a plain tensor (what an engine has today).
    kv = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    kv.view(torch.int64).random_()
    for b in args.batches:
        try:
            t = statistics.median(sync_time(lambda: train_step(model, opt, b))[0] for _ in range(args.reps))
            record(batch=b, strategy="do-nothing", cycle_s=t, d2h_gib=0, h2d_gib=0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            record(batch=b, strategy="do-nothing", cycle_s=None, oom=True, d2h_gib=0, h2d_gib=0)

        def copy_all():
            nonlocal kv
            host.copy_(kv, non_blocking=True)
            torch.cuda.synchronize()
            addr = kv.data_ptr()
            del kv
            torch.cuda.empty_cache()
            train_step(model, opt, b)
            torch.cuda.empty_cache()
            kv = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            kv.copy_(host, non_blocking=True)
            return kv.data_ptr() != addr

        times, moved = zip(*(sync_time(copy_all) for _ in range(args.reps)))
        record(batch=b, strategy="copy-all", cycle_s=statistics.median(times), d2h_gib=nbytes / GiB, h2d_gib=nbytes / GiB, address_changed=any(moved))

        def recompute():
            nonlocal kv
            del kv
            torch.cuda.empty_cache()
            train_step(model, opt, b)
            torch.cuda.empty_cache()
            kv = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            prefill(model, live_tokens)

        times = [sync_time(recompute)[0] for _ in range(args.reps)]
        record(batch=b, strategy="recompute", cycle_s=statistics.median(times), d2h_gib=0, h2d_gib=0, recomputed_tokens=live_tokens)
    del kv
    torch.cuda.empty_cache()

    region = LendableRegion(nbytes, BLOCK)
    region.tensor.view(torch.int64).random_()
    original = block_checksums(region)
    for b in args.batches:
        loan = region.lend(empty=empty, backup=host)  # warm-up, restoring what it overwrites
        try:
            with loan:
                train_step(model, opt, b)
        except torch.OutOfMemoryError as e:
            # Each segment needs contiguous pages: a large one can find no free run even when enough pages are free.
            torch.cuda.empty_cache()
            region.restore(host, loan.dirty_blocks)
            for strategy in ["memlend-eager", "memlend-lazy"]:
                record(batch=b, strategy=strategy, cycle_s=None, oom=True, d2h_gib=0, h2d_gib=0, error=str(e).split(".")[1].strip())
            continue
        region.restore(host, loan.dirty_blocks)
        for strategy in ["memlend-eager", "memlend-lazy"]:
            rows = []
            for _ in range(args.reps):
                def cycle():
                    if strategy == "memlend-eager":
                        region.save(host, live)
                        kwargs = dict(empty=empty)
                    else:
                        kwargs = dict(empty=empty, backup=host)
                    with region.lend(**kwargs) as loan:
                        train_step(model, opt, b)
                    region.restore(host, loan.dirty_blocks)
                    return loan

                t, loan = sync_time(cycle)
                assert torch.equal(block_checksums(region)[live], original[live]), "live KV blocks not restored"
                copied = len(live) if strategy == "memlend-eager" else len(loan.backed_up_blocks)
                rows.append((t, copied, len(loan.dirty_blocks), len(loan.touched_blocks), loan.fallback_peak_bytes))
            t = statistics.median(r[0] for r in rows)
            copied, dirty, touched, fallback = rows[-1][1:]
            record(
                batch=b, strategy=strategy, cycle_s=t, d2h_gib=copied * BLOCK / GiB, h2d_gib=dirty * BLOCK / GiB,
                touched_gib=touched * BLOCK / GiB, fallback_gib=fallback / GiB, address_changed=False,
            )


if __name__ == "__main__":
    main()
