from typing import Any
import torch


class BoundaryLoss:
    def __init__(self) -> None:
        pass
    
    def __call__(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(probs * targets)