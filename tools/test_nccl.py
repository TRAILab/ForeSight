import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")

t = torch.ones(4, device=f"cuda:{rank}") * rank
dist.broadcast(t, src=0)
print(f"[rank {rank}] broadcast ok: {t}")

dist.all_reduce(t)
print(f"[rank {rank}] all_reduce ok: {t}")

dist.destroy_process_group()
print(f"[rank {rank}] NCCL 2-rank test PASSED")
