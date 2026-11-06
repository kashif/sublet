#!/bin/bash
# KV-cache example on this H100, after the warm-cache benchmark (pid given as $1) is done.
cd /fsx/qgallouedec/memlend/benchmarks
PY=/fsx/qgallouedec/memlend/envs/dev/bin/python
while kill -0 $1 2>/dev/null; do sleep 5; done
for live in 30 05; do
  rm -f results/kv_cache_live${live}_h100.jsonl
  $PY bench_kv_cache.py --live 0.$live --batches 2 4 8 --out results/kv_cache_live${live}_h100.jsonl > results/kv_cache_live${live}_h100.log 2>&1
  echo "exit=$?" >> results/kv_cache_live${live}_h100.log
done
