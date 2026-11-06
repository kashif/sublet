"""The README benchmark: a region partly holding data, lent to a job that needs most of the GPU.

The region takes all free GPU memory but 8 GiB, in 64 MiB blocks; a share of it (--data) holds data, in 1 GiB runs
spread over the region, the rest is empty. The job is a forward, backward and AdamW update of Qwen3-0.6B on batch x 2048
tokens (needs `transformers`). One cycle = what the owner does to make room + the job + getting its data back:

- do-nothing: run the job next to the region.
- copy-all: copy the region to pinned CPU memory, free it, run the job, reallocate it, copy it back.
- recompute: free the region, run the job, reallocate it and rebuild the data with forward passes of the model, one
  token per per-token state size of data (a lower bound: the state itself is not written).
- memlend: lend the region, keeping the blocks that hold data. The first cycle maps the job's memory and is reported
  apart (first_s); later cycles reuse it.

    python benchmark.py --data 0.3 --batches 2 8 --out results.jsonl
"""

import argparse
import json
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

import memlend

GiB, BLOCK, SEQ = 1024**3, 64 * 1024**2, 2048
dev = torch.device("cuda")


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t, out


def job(model, update, batch):
    ids = torch.randint(0, model.config.vocab_size, (batch, SEQ), device=dev)
    model(input_ids=ids, labels=ids).loss.backward()
    update.step()
    update.zero_grad(set_to_none=True)


@torch.no_grad()
def rebuild(model, tokens):
    for _ in range(max(1, tokens // (8 * SEQ))):
        model.model(input_ids=torch.randint(0, model.config.vocab_size, (8, SEQ), device=dev), use_cache=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=float, default=0.3, help="share of the region holding data")
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--memlend-only", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(dev)
    update = torch.optim.AdamW(model.parameters(), lr=1e-6)
    job(model, update, 1)  # AdamW state before any loan
    torch.cuda.empty_cache()
    cfg = model.config
    state_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
    nbytes = (torch.cuda.mem_get_info()[0] - 8 * GiB) // BLOCK * BLOCK
    n = nbytes // BLOCK
    runs = list(range(n // 16))
    data_runs = set(random.Random(0).sample(runs, round(len(runs) * args.data)))
    data = [b for b in range(n) if b // 16 in data_runs]
    tokens = len(data) * BLOCK // state_bytes_per_token

    def record(**row):
        row.update(gpu=torch.cuda.get_device_name(), region_gib=nbytes / GiB, data_gib=len(data) * BLOCK / GiB)
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    if not args.memlend_only:
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        owner = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        for b in args.batches:
            try:
                record(batch=b, strategy="do-nothing", cycle_s=statistics.median(timed(lambda: job(model, update, b))[0] for _ in range(args.reps)))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                record(batch=b, strategy="do-nothing", oom=True)

            def cycle(rebuild_data):
                nonlocal owner
                if not rebuild_data:
                    host.copy_(owner, non_blocking=True)
                    torch.cuda.synchronize()
                del owner
                torch.cuda.empty_cache()
                job(model, update, b)
                torch.cuda.empty_cache()
                owner = torch.empty(nbytes, dtype=torch.uint8, device=dev)
                rebuild(model, tokens) if rebuild_data else owner.copy_(host, non_blocking=True)

            record(batch=b, strategy="copy-all", cycle_s=statistics.median(timed(lambda: cycle(False))[0] for _ in range(args.reps)), moved_gib=2 * nbytes / GiB)
            record(batch=b, strategy="recompute", cycle_s=statistics.median(timed(lambda: cycle(True))[0] for _ in range(args.reps)))
        del owner, host
        torch.cuda.empty_cache()

    for b in args.batches:
        region = memlend.empty(n, BLOCK, dtype=torch.uint8)  # a new one per job size: each its own steady state
        blocks = region.view(torch.int64).view(n, -1)
        blocks.random_()
        sums = blocks.sum(1)
        keep = [region[i] for i in data]
        rows = []
        try:
            for _ in range(args.reps + 1):

                def cycle():
                    with memlend.lend(region, keep=keep) as loan:
                        job(model, update, b)
                    return loan

                t, loan = timed(cycle)
                assert torch.equal(blocks.sum(1)[data], sums[data]), "data not restored"
                rows.append((t, loan.copied_bytes, loan.extra_bytes))
        except torch.OutOfMemoryError:
            record(batch=b, strategy="memlend", oom=True)
            continue
        finally:
            del region, blocks, keep
            torch.cuda.empty_cache()
        cycle_s = statistics.median(r[0] for r in rows[1:])
        record(batch=b, strategy="memlend", first_s=rows[0][0], cycle_s=cycle_s, moved_gib=2 * rows[-1][1] / GiB, extra_gib=rows[-1][2] / GiB)


if __name__ == "__main__":
    main()
