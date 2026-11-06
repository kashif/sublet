"""Run the test suite in every environment of envs/ and under several allocator settings; write matrix/results.tsv.

Cells:
- versions: every torch x Python env (default settings).
- settings: one env per torch version (Python 3.12) x PYTORCH_CUDA_ALLOC_CONF values.
- unsupported: torch 2.6 and 2.7, refused by the version check, then run with MEMLEND_SKIP_TORCH_CHECK=1.
- nightly: the nightly env, same two cells.

Each env gets memlend installed from this checkout (non-editable, so the C allocator is built by pip). Logs go to
matrix/logs/<cell>.log.

    python matrix/run_matrix.py [--only versions|settings|nightly]
"""

import argparse
import os
import re
import subprocess
import time

ROOT = "/fsx/qgallouedec/memlend"
ENVS = f"{ROOT}/envs"
LOGS = f"{ROOT}/matrix/logs"
TORCH = ["2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.14", "2.15"]  # 2.6 and 2.7 are the "unsupported" cells
PYTHON = ["3.10", "3.11", "3.12", "3.13"]
SETTINGS = {
    "expandable": "expandable_segments:True",
    "max_split": "max_split_size_mb:128",
    "cudaMallocAsync": "backend:cudaMallocAsync",
}


def env_ready(env):
    return os.path.exists(f"{ENVS}/{env}/.ok")


installed = set()
ARGS = None


def install(env):
    if env in installed or ARGS.no_install:
        return
    py = f"{ENVS}/{env}/bin/python"
    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", py, "--reinstall-package", "memlend", "--no-deps", ROOT],
        check=True,
        env={**os.environ, "UV_CACHE_DIR": f"{ENVS}/.uv-cache", "UV_LINK_MODE": "hardlink"},
    )
    installed.add(env)


def run(cell, env, extra_env, argv=("-m", "pytest", "tests", "-q", "-rA", "-p", "no:cacheprovider")):
    py = f"{ENVS}/{env}/bin/python"
    if not env_ready(env):
        row = [cell, env, "-", "no environment: no torch wheel for this Python (see envs/build_envs*.log)", "-"]
        with open(f"{ROOT}/matrix/results{'_' + ARGS.tag if ARGS.tag else ''}.tsv", "a") as f:
            f.write("\t".join(row) + "\n")
        return row
    install(env)
    cell = f"{ARGS.tag}__{cell}" if ARGS.tag else cell
    log = f"{LOGS}/{cell}.log"
    environ = {k: v for k, v in os.environ.items() if k not in ("PYTORCH_CUDA_ALLOC_CONF", "MEMLEND_SKIP_TORCH_CHECK")}
    environ.update(extra_env)
    with open(log, "w") as f:
        f.write(f"# env={env} extra_env={extra_env}\n")
        f.flush()
        # cwd=tests' parent so the installed package (not src/) is imported: src/ is not on sys.path.
        subprocess.run([py, *argv], cwd=ROOT, env=environ, stdout=f, stderr=subprocess.STDOUT)
    text = open(log).read()
    versions = subprocess.run([py, "-c", "import torch, sys; print(torch.__version__, sys.version.split()[0])"], capture_output=True, text=True).stdout.strip()
    summary = re.findall(r"=+ (.*(?:passed|failed|error|skipped).*) in [\d.]+s", text)
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, flags=re.M)
    outcome = summary[-1] if summary else text.strip().splitlines()[-1][:300]
    row = [cell, env, versions, outcome, ", ".join(failures) or "-"]
    with open(f"{ROOT}/matrix/results{'_' + ARGS.tag if ARGS.tag else ''}.tsv", "a") as f:
        f.write("\t".join(row) + "\n")
    print("\t".join(row), flush=True)
    return row


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--tag", default="", help="prefix for cell names and the results file (e.g. b200)")
    parser.add_argument("--python", nargs="+", default=PYTHON)
    parser.add_argument("--no-install", action="store_true", help="use the memlend already installed in each env")
    parser.add_argument("--cells", nargs="+", help="rerun only these cells (versions__<env> or settings__<env>__<setting>)")
    args = ARGS = parser.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if args.cells:
        for cell in args.cells:
            kind, env, *setting = cell.split("__")
            run(cell, env, {"PYTORCH_CUDA_ALLOC_CONF": SETTINGS[setting[0]]} if kind == "settings" else {})
        return
    os.makedirs(LOGS, exist_ok=True)
    if args.only in (None, "versions"):
        for t in reversed(TORCH):
            for py in args.python:
                run(f"versions__t{t}-py{py}", f"t{t}-py{py}", {})
    if args.only in (None, "settings"):
        for t in reversed(TORCH):
            for name, conf in SETTINGS.items():
                run(f"settings__t{t}-py3.12__{name}", f"t{t}-py3.12", {"PYTORCH_CUDA_ALLOC_CONF": conf})
    if args.only in (None, "unsupported"):
        for t in ["2.6", "2.7"]:
            run(f"unsupported__t{t}-py3.12__refused", f"t{t}-py3.12", {}, ("-c", "import memlend; memlend.LendableRegion(64 << 20)"))
            run(f"unsupported__t{t}-py3.12__forced", f"t{t}-py3.12", {"MEMLEND_SKIP_TORCH_CHECK": "1"})
    if args.only in (None, "nightly"):
        run("nightly__refused", "nightly-py3.13", {}, ("-c", "import memlend; memlend.LendableRegion(64 << 20)"))
        run("nightly__forced", "nightly-py3.13", {"MEMLEND_SKIP_TORCH_CHECK": "1"})


if __name__ == "__main__":
    main()
