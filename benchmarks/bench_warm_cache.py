"""Example 2: a large embedding table kept warm on the GPU between jobs.

Owner: an embedding table (the last hidden states of Qwen3-0.6B over a corpus, bf16, 1024 values per token) taking most
of the GPU (free memory minus --headroom-gib). Every block of it is valuable: there is nothing empty to lend first.
Borrower: an unrelated job that needs X GiB of scratch memory for a short time (allocations of mixed sizes, all written,
freed and refilled). One cycle = make room + the job + get the table back.

Strategies:
- do-nothing: run the job next to the table; OOMs when the job needs more than the headroom.
- copy-all: copy the table to pinned CPU memory, free it, run the job, reallocate it, copy it back.
- recompute: free the table, run the job, reallocate it and recompute it by running the encoder over the corpus again.
  It is timed once, as the full rebuild, since it does not depend on the job.
- memlend-eager: copy the whole table out, lend, restore only the dirty blocks.
- memlend-lazy: lend with a lazy backup: only the blocks the job takes are copied out, and only they are restored.

    python benchmarks/bench_warm_cache.py --out benchmarks/results/warm_cache_<gpu>.jsonl > benchmarks/results/warm_cache_<gpu>.log
"""

import argparse
import json
import statistics

import torch
from transformers import AutoModel

from common import GiB, MiB, block_checksums, dev, link_bandwidth, pinned, sync_time
from memlend import LendableRegion

BLOCK = 64 * MiB
CHUNK = (16, 2048)  # encoder batch: 16 sequences of 2048 tokens


def job(total):
    """X GiB of scratch memory: mixed tensor sizes, written, half freed and refilled with other sizes."""
    mix = [(512 * MiB, 0.60), (64 * MiB, 0.25), (8 * MiB, 0.10), (1 * MiB, 0.05)]
    tensors = []
    for size, share in mix:
        tensors += [torch.empty(size, dtype=torch.uint8, device=dev).fill_(1) for _ in range(int(total * share) // size)]
    freed = sum(t.numel() for t in tensors[::2])
    tensors = tensors[1::2]
    for size, share in reversed(mix):
        tensors += [torch.empty(size, dtype=torch.uint8, device=dev).fill_(2) for _ in range(int(freed * share) // size)]


@torch.inference_mode()
def encode_into(encoder, table):
    """Fill `table` (flat uint8) with bf16 hidden states of the encoder, chunk by chunk."""
    rows = table.view(torch.bfloat16).view(-1, encoder.config.hidden_size)
    per_chunk = CHUNK[0] * CHUNK[1]
    gen = torch.Generator(dev).manual_seed(0)
    for start in range(0, rows.shape[0], per_chunk):
        ids = torch.randint(0, encoder.config.vocab_size, CHUNK, device=dev, generator=gen)
        h = encoder(input_ids=ids, use_cache=False).last_hidden_state.reshape(-1, encoder.config.hidden_size)
        stop = min(start + per_chunk, rows.shape[0])
        rows[start:stop] = h[: stop - start]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom-gib", type=int, default=8)
    parser.add_argument("--job-gib", type=int, nargs="+", default=[4, 16, 32])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    d2h_bw, h2d_bw = link_bandwidth()
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | pinned copies: D2H {d2h_bw:.1f} GB/s, H2D {h2d_bw:.1f} GB/s", flush=True)
    encoder = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    torch.cuda.empty_cache()
    nbytes = (torch.cuda.mem_get_info()[0] - args.headroom_gib * GiB - 6 * GiB) // BLOCK * BLOCK  # 6 GiB for encoding
    host = pinned(nbytes)

    def record(**row):
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    table = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    rebuild_s, _ = sync_time(lambda: encode_into(encoder, table))
    tokens = nbytes // (2 * encoder.config.hidden_size)
    print(f"table {nbytes / GiB:.1f} GiB = {tokens} tokens of hidden states, built in {rebuild_s:.1f} s", flush=True)
    torch.cuda.empty_cache()

    for x in args.job_gib:
        try:
            t = statistics.median(sync_time(lambda: job(x * GiB))[0] for _ in range(args.reps))
            record(job_gib=x, strategy="do-nothing", cycle_s=t, d2h_gib=0, h2d_gib=0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            record(job_gib=x, strategy="do-nothing", cycle_s=None, oom=True, d2h_gib=0, h2d_gib=0)

        def copy_all():
            nonlocal table
            host.copy_(table, non_blocking=True)
            torch.cuda.synchronize()
            del table
            torch.cuda.empty_cache()
            job(x * GiB)
            torch.cuda.empty_cache()
            table = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            table.copy_(host, non_blocking=True)

        t = statistics.median(sync_time(copy_all)[0] for _ in range(args.reps))
        record(job_gib=x, strategy="copy-all", cycle_s=t, d2h_gib=nbytes / GiB, h2d_gib=nbytes / GiB)

        def recompute():
            nonlocal table
            del table
            torch.cuda.empty_cache()
            job(x * GiB)
            torch.cuda.empty_cache()
            table = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            encode_into(encoder, table)

        if x == args.job_gib[0]:
            t, _ = sync_time(recompute)
            record(job_gib=x, strategy="recompute", cycle_s=t, d2h_gib=0, h2d_gib=0, recomputed_tokens=tokens)
    del table
    torch.cuda.empty_cache()

    region = LendableRegion(nbytes, BLOCK)
    region.restore(host)  # the table, as copied out by the last copy-all cycle
    torch.cuda.synchronize()
    original = block_checksums(region)
    for x in args.job_gib:
        with region.lend(backup=host) as loan:  # warm-up
            job(x * GiB)
        region.restore(host, loan.dirty_blocks)
        for strategy in ["memlend-eager", "memlend-lazy"]:
            rows = []
            for _ in range(args.reps):
                def cycle():
                    if strategy == "memlend-eager":
                        region.save(host)
                    with region.lend(backup=host if strategy == "memlend-lazy" else None) as loan:
                        job(x * GiB)
                    region.restore(host, loan.dirty_blocks)
                    return loan

                t, loan = sync_time(cycle)
                assert torch.equal(block_checksums(region), original), "table not restored"
                copied = region.num_blocks if strategy == "memlend-eager" else len(loan.backed_up_blocks)
                rows.append((t, copied, len(loan.dirty_blocks), loan.fallback_peak_bytes))
            t = statistics.median(r[0] for r in rows)
            copied, dirty, fallback = rows[-1][1:]
            record(job_gib=x, strategy=strategy, cycle_s=t, d2h_gib=copied * BLOCK / GiB, h2d_gib=dirty * BLOCK / GiB, fallback_gib=fallback / GiB)


if __name__ == "__main__":
    main()
