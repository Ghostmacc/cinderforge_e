# cinderforge_e/dynamics/ddp_reduce.py
from __future__ import annotations
import torch, torch.distributed as dist

def is_dist():
    return dist.is_available() and dist.is_initialized()

def allreduce_mean(x: float) -> float:
    if not is_dist():
        return float(x)
    t = torch.tensor([x], dtype=torch.float32, device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())

def gather_bool_or(x: bool) -> bool:
    if not is_dist():
        return bool(x)
    t = torch.tensor([1 if x else 0], dtype=torch.int32, device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() > 0

def broadcast_weights(obj: dict) -> dict:
    if not is_dist():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
