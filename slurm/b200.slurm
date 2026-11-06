#!/bin/bash
#SBATCH --job-name=memlend-b200
#SBATCH --partition=hopper-b200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=03:00:00
#SBATCH --output=/fsx/qgallouedec/memlend/benchmarks/results/b200-slurm-%j.out
# B200 run: single-GPU test suite, multi-GPU tests (NCCL inside loans, two devices in one process), the three
# benchmarks, then the torch-version matrix (Python 3.12) on this architecture.
set -u
cd /fsx/qgallouedec/memlend
PY=/fsx/qgallouedec/memlend/envs/dev/bin/python
R=benchmarks/results
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi topo -m
echo "== single-GPU suite $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $PY -m pytest tests --ignore=tests/distributed -q -rA -p no:cacheprovider > $R/b200_tests.log 2>&1; tail -3 $R/b200_tests.log
echo "== multi-GPU tests $(date +%T)"
$PY -m pytest tests/distributed -q -rA -s -p no:cacheprovider > $R/b200_distributed.log 2>&1; tail -3 $R/b200_distributed.log
cd benchmarks
run() {  # name, script, extra args
  echo "== bench $1 $(date +%T)"
  rm -f results/$1_b200.jsonl
  CUDA_VISIBLE_DEVICES=0 $PY $2 --out results/$1_b200.jsonl "${@:3}" > results/$1_b200.log 2>&1; echo "exit=$?" >> results/$1_b200.log
  tail -2 results/$1_b200.log | cut -c1-200
}
run lazy_backup bench_lazy_backup.py
run kv_cache_live30 bench_kv_cache.py --live 0.3
run kv_cache_live05 bench_kv_cache.py --live 0.05
run warm_cache bench_warm_cache.py
cd ..
echo "== matrix $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $PY matrix/run_matrix.py --only versions --python 3.12 --tag b200 --no-install
echo "== done $(date +%T)"
