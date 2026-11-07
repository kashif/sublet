"""The README benchmark: a job that needs most of the GPU, while a tensor holding data takes it all.

The tensor takes all free GPU memory but 8 GiB, in rows of 1 GiB; a share of the rows (--data) holds data to keep. The
job fills one tensor of a given share of that size (--job) with random numbers and sums it. Time per cycle:

- copy-all: copy the whole tensor to pinned CPU memory, free it, run the job, allocate it again and copy it back.
- memlend: lend the tensor keeping the data rows. The first loan is timed apart; later ones are the median of --reps.

    python benchmark.py --data 0.3 --job 0.25 0.75 1.0 --out results.jsonl
"""

import argparse
import functools
import json
import random
import statistics
import time

import torch

import memlend

GiB = 1024**3
dev = torch.device("cuda")


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, out


def job(nbytes):
    x = torch.empty(nbytes // 4, device=dev).normal_()
    return x.sum().item()


def lend_and_run(cache, keep, size):
    with memlend.lend(cache, keep=keep) as loan:
        job(size)
    return loan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=float, default=0.3, help="share of the tensor's rows holding data to keep")
    parser.add_argument("--job", type=float, nargs="+", default=[0.25, 0.75, 1.0], help="job size, as a share of the tensor")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = (torch.cuda.mem_get_info()[0] - 8 * GiB) // GiB
    data = sorted(random.Random(0).sample(range(rows), round(rows * args.data)))
    host = torch.empty(rows * GiB, dtype=torch.uint8, pin_memory=True)

    for share in args.job:
        size = int(share * rows) * GiB
        owner = torch.empty(rows * GiB, dtype=torch.uint8, device=dev)

        def copy_all():
            nonlocal owner
            host.copy_(owner, non_blocking=True)
            torch.cuda.synchronize()
            del owner
            torch.cuda.empty_cache()
            job(size)
            torch.cuda.empty_cache()
            owner = torch.empty(rows * GiB, dtype=torch.uint8, device=dev)
            owner.copy_(host, non_blocking=True)

        copy_all_s = statistics.median(timed(copy_all)[0] for _ in range(args.reps))
        del owner
        torch.cuda.empty_cache()

        cache = memlend.empty(rows, GiB, dtype=torch.uint8)
        cache.view(torch.int64).random_()
        sums = [cache[i].view(torch.int64).sum() for i in data]
        keep = [cache[i] for i in data]

        loans = [timed(functools.partial(lend_and_run, cache, keep, size)) for _ in range(args.reps + 1)]
        assert all(torch.equal(cache[i].view(torch.int64).sum(), s) for i, s in zip(data, sums)), "data not restored"
        row = {
            "gpu": torch.cuda.get_device_name(),
            "tensor_gib": rows,
            "data_gib": len(data),
            "job_gib": size / GiB,
            "copy_all_s": copy_all_s,
            "memlend_first_s": loans[0][0],
            "memlend_s": statistics.median(t for t, _ in loans[1:]),
            "memlend_copied_gib": 2 * loans[-1][1].copied_bytes / GiB,
        }
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        del cache, keep, sums
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
