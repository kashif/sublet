# memlend benchmarks

Every number here comes from the scripts in `benchmarks/` (raw rows in `benchmarks/results/*.jsonl`, logs next to
them; `python benchmarks/make_tables.py h100|b200` prints the tables). Two machines:

- H100 80GB HBM3 (hopper-dev node). Its host link is slow: pinned copies run at 7.1 to 7.7 GB/s device-to-host and
  14.2 GB/s host-to-device (PCIe Gen5 x16, NUMA-local memory). Expect copy-bound numbers to shrink about 8x on a
  healthy link.
- B200 (hopper-b200 node): 57.0 GB/s device-to-host, 55.4 GB/s host-to-device.

Both run torch 2.13.0+cu130 and the same scripts. "Cycle" is everything one round costs: making room, the borrower's
work, and getting the owner's data back. Sizes in parentheses are bytes moved between GPU and CPU (out + back).

## When memlend wins, in plain words

memlend wins whenever the owner's data must survive and the borrower needs less than all of it: nothing that the
borrower does not take is ever copied, and with `empty=` (the blocks that hold nothing) a borrower that fits in them
moves zero bytes. Its cycle is then the borrower's own time plus a few milliseconds. Copying the whole region out and
back costs the same no matter how little the borrower needs (15 s per cycle on the H100, 7 to 10 s on the B200, for a
64 and 163 GiB region). Recomputing is a real competitor only when the data is cheap to rebuild (a few hundred thousand
tokens of KV cache, 0.5 to 2.5 s here), and hopeless when it is not (181 s to rebuild a 61 GiB embedding table).

memlend loses or does not apply when the borrower needs almost the whole region (everything gets copied anyway, and the
lazy copy-out sits inside the loan, so the borrower waits for it), and when a large allocation finds no contiguous free
run in the region: each segment the borrower's allocator asks for must be contiguous, so at batch 8 on the H100 the
Qwen3-0.6B step (58.6 GiB peak, of which one 9.27 GiB fp32 logits tensor) does not fit in a 64 GiB region with 30% of
it in use, while copy-all and recompute, which free the whole region, do. Leaving more ordinary free memory next to the
region, or splitting the largest allocation (chunked losses), avoids it.

## Lazy backup: what each variant buys

`benchmarks/bench_lazy_backup.py`. A region of free memory minus 24 GiB (47.6 GiB on H100, 146.7 GiB on B200) in 64 MiB
blocks, borrowed by one training step (forward, backward, AdamW) of a 0.94B-parameter bf16 transformer at batch x 2048
tokens. "all-valuable": every block holds data; "half-empty": the first half is empty and passed as `empty=`;
"few-empty": only 5 GiB is. Cells: cycle time (GiB copied out). "none" is the loan alone without any backup (the owner's
overwritten data is lost): the floor. eager-all copies the region out before the loan, eager-live only the non-empty
blocks; lazy-host copies each block right before it is lent with the host waiting, lazy-stream orders that copy on the
borrower's stream instead (the library's behaviour), prefetch-N also keeps N blocks copied ahead. Every cycle ends by
restoring the dirty blocks, checked against per-block checksums.

H100:

| batch | empty blocks | none | eager-all | eager-live | lazy-host | lazy-stream | lazy-stream+prefetch-8 | lazy-stream+prefetch-32 |
|---|---|---|---|---|---|---|---|---|
| 2 | all-valuable | 59 ms (0.0) | 7997 ms (47.6) | 7993 ms (47.6) | 1432 ms (5.9) | 1421 ms (5.9) | 1456 ms (6.4) | 1694 ms (7.9) |
| 2 | half-empty | 58 ms (0.0) | 7550 ms (47.6) | 3800 ms (23.8) | 58 ms (0.0) | 58 ms (0.0) | 80 ms (0.5) | 318 ms (2.0) |
| 2 | few-empty | 58 ms (0.0) | 7623 ms (47.6) | 6843 ms (42.6) | 286 ms (1.0) | 288 ms (1.0) | 315 ms (1.5) | 554 ms (3.0) |
| 8 | all-valuable | 186 ms (0.0) | 9448 ms (47.6) | 9440 ms (47.6) | 5590 ms (23.2) | 5568 ms (23.2) | 5566 ms (23.7) | 5777 ms (25.2) |
| 8 | half-empty | 187 ms (0.0) | 7702 ms (47.6) | 3949 ms (23.8) | 186 ms (0.0) | 185 ms (0.0) | 190 ms (0.5) | 320 ms (2.0) |
| 8 | few-empty | 187 ms (0.0) | 9121 ms (47.6) | 8318 ms (42.6) | 4407 ms (18.2) | 4402 ms (18.2) | 4401 ms (18.7) | 4579 ms (20.2) |

B200:

| batch | empty blocks | none | eager-all | eager-live | lazy-host | lazy-stream | lazy-stream+prefetch-8 | lazy-stream+prefetch-32 |
|---|---|---|---|---|---|---|---|---|
| 2 | all-valuable | 32 ms (0.0) | 2914 ms (146.7) | 2911 ms (146.7) | 256 ms (5.9) | 250 ms (5.9) | 250 ms (6.4) | 264 ms (7.9) |
| 2 | half-empty | 33 ms (0.0) | 2798 ms (146.7) | 1415 ms (73.4) | 33 ms (0.0) | 33 ms (0.0) | 33 ms (0.5) | 39 ms (2.0) |
| 2 | few-empty | 32 ms (0.0) | 2821 ms (146.7) | 2723 ms (141.7) | 70 ms (1.0) | 67 ms (1.0) | 64 ms (1.5) | 78 ms (3.0) |
| 8 | all-valuable | 108 ms (0.0) | 3323 ms (146.7) | 3322 ms (146.7) | 975 ms (23.2) | 966 ms (23.2) | 967 ms (23.7) | 966 ms (25.2) |
| 8 | half-empty | 109 ms (0.0) | 2872 ms (146.7) | 1491 ms (73.4) | 108 ms (0.0) | 108 ms (0.0) | 109 ms (0.5) | 109 ms (2.0) |
| 8 | few-empty | 108 ms (0.0) | 3225 ms (146.7) | 3130 ms (141.7) | 782 ms (18.2) | 776 ms (18.2) | 774 ms (18.7) | 774 ms (20.2) |

What each change removes, starting from the phase-1 lazy mode (host waits, no knowledge of empty blocks: 988 ms per loan
at batch 2 on H100, 142 ms on B200, against 59 and 33 ms without backup):

- Declaring empty blocks removes all of it when the borrower fits in them: 58 ms on H100 and 33 ms on B200 at batch 2,
  identical to no backup, with 0 bytes copied. When it spills (few-empty), only the spilled blocks are copied: 1 GiB
  instead of 5.9 at batch 2.
- Ordering the copies on the borrower's stream removes 1% on H100 (977 vs 988 ms, the link is the bottleneck) and 2 to
  4% on B200 (136 vs 142 ms at batch 2, 515 vs 524 ms at batch 8).
- Prefetching removes nothing measurable: 8 blocks ahead is within a few ms either way (44 vs 48 ms at best, B200 batch
  2 few-empty), 32 ahead is slower because it copies blocks the borrower never takes (318 vs 58 ms on H100 at batch 2,
  half-empty).
- Against eager backup the gap is the copy-out: eager copies every valuable block whether or not it is lent (7.5 s on
  H100, 2.8 s on B200 for the whole region), lazy only the ones that are.

## Example 1: an inference engine's KV cache lent to a training step

`benchmarks/bench_kv_cache.py`. The owner is a KV-cache-like region taking all free memory but 8 GiB (64.1 GiB on H100,
163.2 GiB on B200), with 30% (or 5%, short contexts) of it holding live sequences in 1 GiB runs spread over the region.
The borrower is a training step of Qwen3-0.6B (transformers, bf16, AdamW) on batch x 2048 tokens; its peak memory is
14.7 GiB at batch 2, 29.3 GiB at batch 4, 58.6 GiB at batch 8 and 117.3 GiB at batch 16 (batch 16 does not fit on the
H100 at all). Strategies: do-nothing (train next to the full cache), copy-all (copy the cache to pinned CPU memory,
free it, train, reallocate, copy back: what colocated RL trainers do today), recompute (free, train, reallocate, and
re-run the prefill of the live tokens: forward passes of the base model, without writing the KV, so a lower bound),
memlend-eager (copy the live blocks out, lend with `empty=`, restore the dirty ones), memlend-lazy (lend with `empty=`
and a lazy backup).

H100, 30% live (19.0 GiB = 177,883 tokens):

| batch x 2048 | do-nothing | copy-all | recompute | memlend-eager | memlend-lazy |
|---|---|---|---|---|---|
| 2 | OOM | 15.46 s (128.2 GiB) | 1.39 s | 3.11 s (19.0 GiB) | 0.12 s |
| 4 | OOM | 15.18 s (128.2 GiB) | 1.53 s | 3.18 s (19.0 GiB) | 0.21 s |
| 8 | OOM | 15.40 s (128.2 GiB) | 1.76 s | OOM | OOM |

H100, 5% live (3.0 GiB = 28,086 tokens):

| batch x 2048 | do-nothing | copy-all | recompute | memlend-eager | memlend-lazy |
|---|---|---|---|---|---|
| 2 | OOM | 15.33 s (128.2 GiB) | 0.52 s | 0.59 s (3.0 GiB) | 0.12 s |
| 4 | OOM | 15.16 s (128.2 GiB) | 0.66 s | 0.67 s (3.0 GiB) | 0.21 s |
| 8 | OOM | 15.57 s (128.2 GiB) | 0.89 s | OOM | OOM |

B200, 30% live (49.0 GiB = 458,752 tokens):

| batch x 2048 | do-nothing | copy-all | recompute | memlend-eager | memlend-lazy |
|---|---|---|---|---|---|
| 2 | OOM | 6.55 s (326.4 GiB) | 2.02 s | 1.00 s (49.0 GiB) | 0.08 s |
| 8 | OOM | 6.75 s (326.4 GiB) | 2.23 s | 1.34 s (58.3 GiB) | 0.53 s (18.6 GiB) |
| 16 | OOM | 9.58 s (326.4 GiB) | 2.50 s | 1.79 s (69.7 GiB) | 1.13 s (41.4 GiB) |

B200, 5% live (8.0 GiB):

| batch x 2048 | do-nothing | copy-all | recompute | memlend-eager | memlend-lazy |
|---|---|---|---|---|---|
| 2 | OOM | 8.47 s (326.4 GiB) | 0.53 s | 0.23 s (8.0 GiB) | 0.08 s |
| 8 | OOM | 6.74 s (326.4 GiB) | 0.74 s | 0.39 s (8.0 GiB) | 0.24 s |
| 16 | OOM | 7.00 s (326.4 GiB) | 1.00 s | 0.63 s (9.0 GiB) | 0.48 s (2.0 GiB) |

Reading it: memlend-lazy is the fastest everywhere it fits, 2.1x to 25x faster than recompute and 8.5x to 130x faster than
copy-all. At batches 2 and 4 the step fits in the empty blocks and nothing moves. On the B200 at batches 8 and 16 some
live blocks are lent anyway (18.6 and 41.4 GiB moved at 30% live) although there is far more empty space than the step
needs: the step's large tensors need contiguous runs longer than the 1 GiB empty runs. The H100 batch 8 OOM is the same
effect, with no free run long enough at all (see above). copy-all's reallocated cache landed at the same address in
these runs, but nothing guarantees it (phase 1 showed it moving).

## Example 2: an embedding table kept warm between jobs

`benchmarks/bench_warm_cache.py`. The owner keeps a large table of Qwen3-0.6B hidden states (bf16, 1024 values per
token) over a corpus, taking all free memory but 14 GiB (60.6 GiB = 31.8M tokens on H100). Every block is valuable:
nothing is empty. Between uses, an unrelated job needs X GiB of scratch memory (mixed tensor sizes, written, half freed
and refilled). Rebuilding the table means running the encoder over the corpus again; it is timed once, since it does
not depend on the job.

H100:

| job needs (GiB) | do-nothing | copy-all | recompute | memlend-eager | memlend-lazy |
|---|---|---|---|---|---|
| 4 | 0.02 s | 14.33 s (121.2 GiB) | 180.75 s | 9.80 s (64.4 GiB) | 0.88 s (7.5 GiB) |
| 16 | 0.08 s | 14.55 s (121.2 GiB) | - | 10.76 s (76.9 GiB) | 3.82 s (32.6 GiB) |
| 32 | OOM | 14.87 s (121.2 GiB) | - | 12.01 s (93.2 GiB) | 7.64 s (65.2 GiB) |

B200:

| job needs (GiB) | do-nothing | copy-all | recompute | memlend-eager | memlend-lazy |
|---|---|---|---|---|---|
| 4 | 0.01 s | 6.30 s (319.4 GiB) | 318.69 s | 3.09 s (163.4 GiB) | 0.15 s (7.5 GiB) |
| 16 | 0.05 s | 6.43 s (319.4 GiB) | - | 3.34 s (176.0 GiB) | 0.63 s (32.6 GiB) |
| 32 | OOM | 6.55 s (319.4 GiB) | - | 3.68 s (192.3 GiB) | 1.26 s (65.2 GiB) |

Reading it: with nothing empty, memlend-lazy moves exactly what the job takes (out and back), so its cycle grows with
the job (0.88 s for 4 GiB, 7.6 s for 32 GiB on H100) while copy-all stays at the whole table (14.3 to 14.9 s). Rebuilding
takes 181 s on H100. On B200 the same holds at the faster link: 0.15 s for 4 GiB and 1.26 s for 32 GiB, against 6.3 to 6.6 s for copy-all and 319 s to rebuild. do-nothing works while the job fits next to the table and OOMs beyond.

## Not measured

- Multi-step training loops where the loan spans several steps with optimizer state created beforehand (each cycle here
  is one step).
- Any engine's real KV cache layout (vLLM, SGLang): the KV cache here is a flat buffer with live data placed in 1 GiB
  runs.
- Other link speeds than the two machines above, and other GPUs.
