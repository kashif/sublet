"""Backup strategies for a lent region: eager (copy before the loan) vs lazy (copy a block just before it is lent).

Owner: a region of most of the GPU (default: free memory minus 24 GiB), split in 64 MiB blocks, of which a share holds
nothing ("empty"). Borrower: one training step of the 0.94B-parameter benchmark transformer (batch x 2048 tokens).

Scenarios (which blocks are empty, and whether the owner tells memlend):
- all-valuable: every block holds data (the phase-1 setting).
- half-empty: the first half of the region is empty and passed as `empty=`; the borrower fits in it.
- few-empty: 5 GiB at the start is empty; larger borrowers spill into valuable blocks.

Strategies: none (no backup, the loan alone), eager-all (copy the whole region out first), eager-live (copy only the
non-empty blocks out first), lazy-host (copy each block just before it is lent, host waits), lazy-stream (same, ordered on
the borrower's stream), lazy-stream+prefetch-N (and keep N blocks copied ahead). Each cycle = backup + loan + restore of
the dirty blocks, checked against per-block checksums.

    python benchmarks/bench_lazy_backup.py --out benchmarks/results/lazy_backup_<gpu>.json > benchmarks/results/lazy_backup_<gpu>.log
"""

import argparse
import json
import statistics
import time

import torch

from common import GiB, MiB, block_checksums, link_bandwidth, make_model, pinned, sync_time, train_step
from memlend import LendableRegion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom-gib", type=int, default=24, help="GPU memory left outside the region")
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    d2h, h2d = link_bandwidth()
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | pinned copies: D2H {d2h:.1f} GB/s, H2D {h2d:.1f} GB/s", flush=True)

    model, opt = make_model()
    train_step(model, opt, 1)  # optimizer states outside any loan
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0]
    nbytes = (free - args.headroom_gib * GiB) // (64 * MiB) * (64 * MiB)
    region = LendableRegion(nbytes, 64 * MiB)
    n = region.num_blocks
    region.tensor.view(torch.int64).random_()
    original = block_checksums(region)
    t = time.perf_counter()
    backup, pristine = pinned(nbytes), pinned(nbytes)
    region.save(pristine)  # untimed reference copy, used to repair the region between runs
    torch.cuda.synchronize()
    print(f"region {nbytes / GiB:.1f} GiB ({n} blocks), two pinned buffers ready in {time.perf_counter() - t:.1f} s", flush=True)

    scenarios = {"all-valuable": [], "half-empty": list(range(n // 2)), "few-empty": list(range(min(80, n // 4)))}
    strategies = ["none", "eager-all", "eager-live", "lazy-host", "lazy-stream", "lazy-stream+prefetch-8", "lazy-stream+prefetch-32"]
    for batch in args.batches:
        for _ in range(2):  # warm-up: kernels, cuBLAS handles
            with region.lend() as loan:
                train_step(model, opt, batch)
            region.restore(pristine, loan.touched_blocks)
        for scenario, empty in scenarios.items():
            live = [b for b in range(n) if b not in set(empty)]
            for strategy in strategies:
                runs = []
                for _ in range(args.reps):
                    kwargs = dict(empty=empty)
                    backup_s = 0.0
                    if strategy == "eager-all":
                        backup_s, _ = sync_time(lambda: region.save(backup))
                    elif strategy == "eager-live":
                        backup_s, _ = sync_time(lambda: region.save(backup, live))
                    elif strategy.startswith("lazy"):
                        kwargs.update(backup=backup, _sync_copies=strategy == "lazy-host")
                        if "prefetch" in strategy:
                            kwargs["_prefetch"] = int(strategy.rsplit("-", 1)[1])
                    loan_s, loan = sync_time(lambda: _loan(region, kwargs, model, opt, batch))
                    restore_s = 0.0
                    if strategy != "none":
                        restore_s, _ = sync_time(lambda: region.restore(backup, loan.dirty_blocks))
                        assert torch.equal(block_checksums(region)[live], original[live]), f"{strategy}: owner data not restored"
                    if strategy.startswith("lazy"):
                        copied = len(loan.backed_up_blocks)
                    else:
                        copied = {"none": 0, "eager-all": n, "eager-live": len(live)}[strategy]
                    runs.append((backup_s + loan_s + restore_s, backup_s, loan_s, restore_s, copied, len(loan.dirty_blocks), len(loan.touched_blocks)))
                    region.restore(pristine, loan.touched_blocks)  # untimed: back to the original content
                    torch.cuda.synchronize()
                med = [statistics.median(r[i] for r in runs) for i in range(4)] + list(runs[-1][4:])
                row = dict(zip(["cycle_s", "backup_s", "loan_s", "restore_s", "copied_out_blocks", "dirty_blocks", "touched_blocks"], med))
                row.update(batch=batch, scenario=scenario, strategy=strategy, copied_out_gib=row["copied_out_blocks"] / 16, restored_gib=row["dirty_blocks"] / 16)
                print(json.dumps(row), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


def _loan(region, kwargs, model, opt, batch):
    with region.lend(**kwargs) as loan:
        train_step(model, opt, batch)
    return loan


if __name__ == "__main__":
    main()
