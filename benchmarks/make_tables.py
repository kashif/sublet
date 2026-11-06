"""Print the markdown tables of BENCHMARKS.md from the benchmark results (benchmarks/results/*.jsonl)."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, "results", f"{name}.jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def fmt_s(x):
    return "OOM" if x is None else f"{x:.2f} s"


def example_table(rows, key, key_label):
    order = ["do-nothing", "copy-all", "recompute", "memlend-eager", "memlend-lazy"]
    keys = sorted({r[key] for r in rows})
    print(f"| {key_label} | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    for k in keys:
        cells = []
        for strategy in order:
            r = next((r for r in rows if r[key] == k and r["strategy"] == strategy), None)
            if r is None:
                cells.append("-")
            elif r.get("oom"):
                cells.append("OOM")
            else:
                moved = r["d2h_gib"] + r["h2d_gib"]
                cells.append(f"{fmt_s(r['cycle_s'])} ({moved:.1f} GiB)" if moved else fmt_s(r["cycle_s"]))
        print(f"| {k} | " + " | ".join(cells) + " |")


def lazy_table(rows):
    """One row per (batch, scenario): cycle time and GiB copied out per strategy."""
    order = ["none", "eager-all", "eager-live", "lazy-host", "lazy-stream", "lazy-stream+prefetch-8", "lazy-stream+prefetch-32"]
    print("| batch | empty blocks | " + " | ".join(order) + " |")
    print("|---|---|" + "---|" * len(order))
    for batch in sorted({r["batch"] for r in rows}):
        for scenario in ["all-valuable", "half-empty", "few-empty"]:
            cells = []
            for strategy in order:
                r = next((r for r in rows if r["batch"] == batch and r["scenario"] == scenario and r["strategy"] == strategy), None)
                cells.append("-" if r is None else f"{r['cycle_s'] * 1e3:.0f} ms ({r['copied_out_gib']:.1f})")
            print(f"| {batch} | {scenario} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    gpu = sys.argv[1]
    print(f"## {gpu}\n\n### lazy backup\n")
    lazy_table(load(f"lazy_backup_{gpu}"))
    for live in ("30", "05"):
        print(f"\n### KV cache, {int(live)}% live\n")
        example_table(load(f"kv_cache_live{live}_{gpu}"), "batch", "batch x 2048")
    print("\n### warm cache\n")
    example_table(load(f"warm_cache_{gpu}"), "job_gib", "job needs (GiB)")
