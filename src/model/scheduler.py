from enum import Enum

import torch.optim as optim
from omegaconf import DictConfig


class Scheduler(Enum):
    COSINE_ANNEALING_WARM_RESTARTS = "CosineAnnealingWarmRestarts"


def get_scheduler(config: DictConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler:
    scheduler_name = config.scheduler.name
    scheduler = Scheduler(scheduler_name)

    if scheduler == Scheduler.COSINE_ANNEALING_WARM_RESTARTS:
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.CosineAnnealingWarmRestarts.T_0,
            eta_min=config.scheduler.CosineAnnealingWarmRestarts.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
