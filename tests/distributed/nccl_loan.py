"""NCCL collectives inside loans. Run with `torchrun --nproc-per-node 2 tests/distributed/nccl_loan.py [lazy|eager]`.

`lazy`: the process group is created without `device_id`, so the communicator is created by the first collective,
which happens inside a loan. `eager`: the communicator is created (and warmed up) before any loan.
Prints one line per check and exits non-zero on the first failure.
"""

import os
import sys

import torch
import torch.distributed as dist

import memlend

MiB = 1024**2
mode = sys.argv[1]
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
if mode == "eager":
    dist.init_process_group("nccl", device_id=dev)
    warm = torch.ones(1, device=dev)
    dist.all_reduce(warm)
else:
    dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()


def log(msg):
    print(f"[rank {rank}] {msg}", flush=True)


cache = memlend.empty(2048 * MiB, dtype=torch.uint8, device=dev)
cache.view(torch.int64).random_()
before = cache.clone()
results = torch.zeros(3, device=dev)


def collectives():
    n = 64 * MiB // 4  # 64 MiB of fp32 per tensor
    x = torch.full((n,), float(rank + 1), device=dev)
    log("all_reduce")
    dist.all_reduce(x)
    log("all_gather")
    gathered = torch.empty(world * n, device=dev)
    dist.all_gather_into_tensor(gathered, torch.full((n,), float(rank), device=dev))
    b = torch.full((n,), float(rank + 10), device=dev)
    log("broadcast")
    dist.broadcast(b, src=0)
    work = dist.all_reduce(torch.ones(n, device=dev), async_op=True)
    work.wait()
    results[0] = x[0] - world * (world + 1) / 2
    results[1] = (gathered.view(world, n)[:, 0] - torch.arange(world, device=dev)).abs().sum()
    results[2] = b[0] - 10


log(f"start ({mode} communicator)")
for i in range(3):
    free_before = torch.cuda.mem_get_info(dev)[0]
    try:
        with memlend.lend(cache) as loan:
            collectives()
    except memlend.LeakError as e:
        log(f"loan {i}: FAIL LeakError\n{e}")
        sys.exit(1)
    free_after = torch.cuda.mem_get_info(dev)[0]
    ok = bool((results == 0).all())
    intact = torch.equal(cache, before)
    log(
        f"loan {i} ({mode} communicator): results {'ok' if ok else 'WRONG'}, kept memory {'intact' if intact else 'CHANGED'}, "
        f"copied {loan.copied_bytes / MiB:.0f} MiB, free GPU memory outside the tensor changed by "
        f"{(free_before - free_after) / MiB:+.0f} MiB"
    )
    if not (ok and intact):
        sys.exit(1)
# Collectives captured in CUDA graphs: one captured before any loan and replayed inside and after loans, one captured
# and replayed inside a loan. NCCL may register graph-captured buffers; the tensor's physical pages never change.
n = 1024**2
static = torch.ones(n, device=dev)
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    dist.all_reduce(static)  # warm-up outside capture
torch.cuda.current_stream().wait_stream(side)
graph_before = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph_before):
    dist.all_reduce(static)


def replay_before():
    static.fill_(1)
    graph_before.replay()
    torch.cuda.synchronize()
    return bool((static == world).all())


def capture_inside():
    buf = torch.ones(n, device=dev)  # in the lent memory
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dist.all_reduce(buf)
    buf.fill_(2)
    g.replay()
    torch.cuda.synchronize()
    return bool((buf == 2 * world).all())


with memlend.lend(cache):
    ok_before_inside = replay_before()
    ok_capture_inside = capture_inside()
ok_before_after = replay_before()
log(f"graph-captured all_reduce: captured before, replayed in loan {'ok' if ok_before_inside else 'WRONG'}, after loan "
    f"{'ok' if ok_before_after else 'WRONG'}; captured and replayed in loan {'ok' if ok_capture_inside else 'WRONG'}")
if not (ok_before_inside and ok_before_after and ok_capture_inside):
    sys.exit(1)
del graph_before  # a live graph holding captured collectives can hang communicator teardown
torch.cuda.synchronize()
dist.barrier(device_ids=[local_rank])
log("PASS")
dist.destroy_process_group()
