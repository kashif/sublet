#!/bin/bash
#SBATCH --job-name=memlend-matrix
#SBATCH --partition=hopper-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/fsx/qgallouedec/memlend/matrix/slurm-%j.out
# Test matrix on one H100: every torch x Python env, allocator settings, nightly.
set -u
cd /fsx/qgallouedec/memlend
nvidia-smi --query-gpu=name,driver_version --format=csv
/fsx/qgallouedec/memlend/envs/dev/bin/python matrix/run_matrix.py
