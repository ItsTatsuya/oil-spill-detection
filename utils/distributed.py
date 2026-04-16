import datetime
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def setup_distributed(
    model: nn.Module,
    find_unused_parameters: bool = False,
) -> Tuple[nn.Module, bool, int]:
    num_gpus = torch.cuda.device_count()

    is_distributed = (
        "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", 1)) > 1
    )

    if not is_distributed:
        if num_gpus >= 1:
            device = torch.device("cuda:0")
            model = model.to(device)
            logger.info("Training on single GPU: cuda:0")
        else:
            logger.warning("No CUDA GPUs available. Training on CPU (very slow).")
        return model, False, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(minutes=30),
                device_id=local_rank,
            )
        except Exception as e:
            logger.error(f"DDP initialization failed: {e}")
            raise

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )

    if rank == 0:
        logger.info(f"Training on {world_size} GPUs with DDP (NCCL backend)")
        logger.info(f"  Rank: {rank}/{world_size}, Local rank: {local_rank}")

    return model, True, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / get_world_size()
    return rt
