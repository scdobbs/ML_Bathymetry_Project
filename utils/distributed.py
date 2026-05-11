"""
Single-node DDP helpers.

Design principle: the same code runs at WORLD_SIZE=1 and at WORLD_SIZE=N
with no branching at the call site. setup_distributed() inspects torchrun
environment variables and either initializes a process group or returns
solo defaults; all_reduce_mean() is a no-op outside DDP.
"""

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed if WORLD_SIZE > 1, else return solo defaults.

    Returns:
        (rank, local_rank, world_size) — (0, 0, 1) when not running under torchrun.
    """
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return 0, 0, 1

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def is_main_process() -> bool:
    """True on rank 0 (always True in solo mode)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def cleanup_distributed() -> None:
    """Tear down the process group if it's up."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean-reduce a tensor across ranks. No-op in solo mode."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def barrier() -> None:
    """Block until every rank reaches this point. No-op in solo mode."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
