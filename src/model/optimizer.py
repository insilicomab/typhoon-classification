from enum import Enum

import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from model.optimizers.ranger21 import Ranger21
from model.optimizers.sam import SAM


class Optimizer(Enum):
    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"
    RANGER21 = "Ranger21"
    SAM = "SAM"


def get_optimizer(config: DictConfig, net: nn.Module) -> optim.Optimizer:
    optimizer_name = config.optimizer.name
    optimizer = Optimizer(optimizer_name)

    if optimizer == Optimizer.ADAM:
        return optim.Adam(
            net.parameters(),
            lr=config.optimizer.adam.lr,
            weight_decay=config.optimizer.adam.weight_decay,
        )
    elif optimizer == Optimizer.ADAMW:
        return optim.AdamW(
            net.parameters(),
            lr=config.optimizer.adamW.lr,
            weight_decay=config.optimizer.adamW.weight_decay,
        )
    elif optimizer == Optimizer.SGD:
        return optim.SGD(
            net.parameters(),
            lr=config.optimizer.sgd.lr,
            momentum=config.optimizer.sgd.momentum,
            weight_decay=config.optimizer.sgd.weight_decay,
        )
    elif optimizer == Optimizer.RANGER21:
        return Ranger21(
            net.parameters(),
            lr=config.optimizer.ranger21.lr,
            weight_decay=config.optimizer.ranger21.weight_decay,
            num_epochs=config.trainer.max_epochs,
            num_batches_per_epoch=config.optimizer.ranger21.num_batches_per_epoch,
        )
    elif optimizer == Optimizer.SAM:
        base_optimizer_name = config.optimizer.sam.base_optimizer
        base_optimizer = Optimizer(base_optimizer_name)

        if base_optimizer == Optimizer.SGD:
            base_optimizer = optim.SGD
            args = config.optimizer.sgd
        elif base_optimizer == Optimizer.ADAM:
            base_optimizer = optim.Adam
            args = config.optimizer.adam
        elif base_optimizer == Optimizer.ADAMW:
            base_optimizer = optim.AdamW
            args = config.optimizer.adamW
        else:
            raise ValueError(
                f"Unknown base optimizer of SAM: {base_optimizer_name}"
            )
        return SAM(
            net.parameters(),
            base_optimizer=base_optimizer,
            rho=config.optimizer.sam.rho,
            adaptive=config.optimizer.sam.adaptive,
            **args,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
