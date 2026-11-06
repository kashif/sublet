"""NCCL collectives inside loans. Run with `torchrun --nproc-per-node 2 tests/distributed/nccl_loan.py [lazy|eager]`.

`lazy`: the process group is created without `device_id`, so the communicator is created by the first collective,
which happens inside a loan. `eager`: the communicator is created (and warmed up) before any loan.
Prints one line per check and exits non-zero on the first failure.
"""

import os
import sys

import torch
import torch.distributed as dist

from memlend import LeakError, LendableRegion

MiB = 1 << 20
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


region = LendableRegion(2048 * MiB, 64 * MiB, device=dev)
region.tensor.view(torch.int64).random_()
before = region.tensor.clone()
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
    assert all(region.tensor.data_ptr() <= t.data_ptr() < region.tensor.data_ptr() + region.nbytes for t in (x, gathered, b))


log(f"start ({mode} communicator)")
for i in range(3):
    free_before = torch.cuda.mem_get_info(dev)[0]
    try:
        with region.lend() as loan:
            collectives()
    except LeakError as e:
        log(f"loan {i}: FAIL LeakError\n{e}")
        sys.exit(1)
    free_after = torch.cuda.mem_get_info(dev)[0]
    ok = bool((results == 0).all())
    view, ref = region.tensor.view(region.num_blocks, -1), before.view(region.num_blocks, -1)
    untouched_ok = all(torch.equal(view[b], ref[b]) for b in range(region.num_blocks) if b not in loan.touched_blocks)
    log(
        f"loan {i} ({mode} communicator): results {'ok' if ok else 'WRONG'}, untouched blocks {'identical' if untouched_ok else 'CHANGED'}, "
        f"touched {len(loan.touched_blocks)} blocks, free GPU memory outside the region changed by "
        f"{(free_before - free_after) / MiB:+.0f} MiB"
    )
    if not (ok and untouched_ok):
        sys.exit(1)
    region.tensor.copy_(before)
# Collectives captured in CUDA graphs: one captured before any loan and replayed inside and after loans, one captured
# and replayed inside a loan. NCCL may register graph-captured buffers; the region's physical pages never change.
n = 1 << 20
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
    buf = torch.ones(n, device=dev)  # in the region
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dist.all_reduce(buf)
    buf.fill_(2)
    g.replay()
    torch.cuda.synchronize()
    return bool((buf == 2 * world).all())


with region.lend() as loan:
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
