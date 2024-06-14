from enum import Enum

import torch
from omegaconf import DictConfig
from torch import nn

from model.losses.focal_loss import FocalLoss


class LossFunction(Enum):
    CROSS_ENTROPY = "CrossEntropyLoss"
    BINARY_CROSS_ENTROPY = "BinaryCrossEntropy"
    FOCAL_LOSS = "FocalLoss"


def get_loss_fn(config: DictConfig) -> torch.nn.Module:
    loss_fn_name = config.loss_fn.name
    loss_fn = LossFunction(loss_fn_name)

    if loss_fn == LossFunction.CROSS_ENTROPY:
        return nn.CrossEntropyLoss()

    elif loss_fn == LossFunction.BINARY_CROSS_ENTROPY:
        return nn.BCELoss()

    elif loss_fn == LossFunction.FOCAL_LOSS:
        return FocalLoss(
            gamma=config.loss_fn.focal.gamma, reduction=config.loss_fn.focal.reduction
        )

    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
