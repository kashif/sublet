"""Markdown tables of the test matrix from matrix/results*.tsv (the last row of each cell wins)."""

import re
import sys

ROOT = "/fsx/qgallouedec/memlend/matrix"


def load(tag=""):
    cells = {}
    for line in open(f"{ROOT}/results{'_' + tag if tag else ''}.tsv"):
        cell, env, versions, outcome, failures = line.rstrip("\n").split("\t")
        cells[cell.removeprefix(f"{tag}__") if tag else cell] = (versions, outcome, failures)
    return cells


def short(outcome, failures):
    if outcome.startswith("no environment"):
        return "no wheel"
    m = re.search(r"(\d+) failed", outcome)
    p = re.search(r"(\d+) passed", outcome)
    if m:
        names = ", ".join(f.split("::")[-1] for f in failures.split(", "))
        return f"**{m.group(1)} failed** ({names}), {p.group(1) if p else 0} passed"
    if p:
        return f"{p.group(1)} passed"
    return outcome[:120]


def versions_table(cells, torch_versions, pythons):
    print("| torch | " + " | ".join(f"Python {p}" for p in pythons) + " |")
    print("|---|" + "---|" * len(pythons))
    for t in torch_versions:
        row = []
        for p in pythons:
            c = cells.get(f"versions__t{t}-py{p}")
            row.append("-" if c is None else short(c[1], c[2]))
        exact = next((c[0].split()[0] for k, c in cells.items() if k.startswith(f"versions__t{t}-") and c[0] != "-"), t)
        print(f"| {exact} | " + " | ".join(row) + " |")


def settings_table(cells, torch_versions, settings):
    print("| torch (Python 3.12) | " + " | ".join(settings) + " |")
    print("|---|" + "---|" * len(settings))
    for t in torch_versions:
        row = []
        for s in settings:
            c = cells.get(f"settings__t{t}-py3.12__{s}")
            row.append("-" if c is None else short(c[1], c[2]))
        print(f"| {t} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    cells = load(tag)
    torch_versions = ["2.15", "2.14", "2.13", "2.12", "2.11", "2.10", "2.9", "2.8"]
    versions_table(cells, torch_versions, ["3.10", "3.11", "3.12", "3.13"] if not tag else ["3.12"])
    if not tag:
        print()
        settings_table(cells, torch_versions, ["expandable", "max_split", "cudaMallocAsync"])
        print()
        for k in ["unsupported__t2.6-py3.12__refused", "unsupported__t2.6-py3.12__forced", "unsupported__t2.7-py3.12__refused",
                  "unsupported__t2.7-py3.12__forced", "nightly__refused", "nightly__forced"]:
            c = cells.get(k)
            print(f"- {k}: " + ("-" if c is None else f"{c[0]}: {short(c[1], c[2])}"))
