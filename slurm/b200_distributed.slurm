#!/bin/bash
#SBATCH --job-name=memlend-dist
#SBATCH --partition=hopper-b200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/fsx/qgallouedec/memlend/benchmarks/results/b200-dist-slurm-%j.out
# Multi-GPU tests on B200: NCCL collectives inside loans (lazy and eager communicator), two devices in one process.
cd /fsx/qgallouedec/memlend
PY=/fsx/qgallouedec/memlend/envs/dev/bin/python
nvidia-smi --query-gpu=name --format=csv
for mode in lazy eager; do
  echo "== nccl_loan $mode $(date +%T)"
  timeout 600 $PY -m torch.distributed.run --nproc-per-node 2 tests/distributed/nccl_loan.py $mode > benchmarks/results/b200_nccl_$mode.log 2>&1; echo "exit=$?"
done
echo "== pytest tests/distributed $(date +%T)"
timeout 1200 $PY -m pytest tests/distributed -q -rA -p no:cacheprovider > benchmarks/results/b200_distributed.log 2>&1; tail -5 benchmarks/results/b200_distributed.log
echo "== done $(date +%T)"
