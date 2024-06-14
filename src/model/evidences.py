# source from https://muratsensoy.github.io/uncertainty.html and converted for pytorch
import torch
import torch.nn.functional as F


def relu_evidence(logits: torch.Tensor) -> torch.Tensor:
    return F.relu(logits)


def calculate_uncertainty(logits: torch.Tensor, num_classes: int) -> float:
    evidence = relu_evidence(logits)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1)
    uncertainty = uncertainty.cpu().numpy()
    return uncertainty
