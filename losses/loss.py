from typing import List, cast, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

from monai.losses.dice import DiceLoss, DiceFocalLoss, DiceCELoss, GeneralizedDiceLoss

from .utils import dist_map_transform

class Loss:
    def __init__(self, 
                 losses: Sequence[str], 
                 num_classes: int, 
                 loss_combine: str, 
                 one_hot: bool,
                 include_background: bool) -> None:
        self.losses = []
        self.num_classes = num_classes
        self.loss_combine = loss_combine
        self.one_hot = one_hot
        self.include_background = include_background
        self.dist_transform = dist_map_transform()
        
        for name in losses.split(','):
            if name == "mse":
                self.losses.append(MSELoss())
            elif name == "ce":
                self.losses.append(CrossEntropyLoss())
            elif name == "bce":
                self.losses.append(BCEWithLogitsLoss())
            elif name == "dice":
                self.losses.append(DiceLoss(sigmoid=True))
            elif name == "boundary":
               self.losses.append(BoundaryLoss(num_classes, one_hot))
            elif name == "dice_ce":
                self.losses.append(DiceCELoss(sigmoid=True))
            elif name == "dice_focal":
                self.losses.append(DiceFocalLoss(sigmoid=True))
            elif name == "generalized_dice":
               self.losses.append(GeneralizedDiceLoss(sigmoid=True))
            elif name == "multi_neighbor":
               self.losses.append(MultiNeighborLoss(num_classes=num_classes))

        print(f"loss : {self.losses}")
        
    def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
        losses = []
        # if not self.include_background:
        #     preds = preds[:, 1:, ...]
        #     labels = labels[:, 1:, ...]
            
        for loss in self.losses:
            if isinstance(loss, MSELoss):
                losses.append(loss(torch.sigmoid(preds), labels))
            elif isinstance(loss, BoundaryLoss):
                losses.append(loss(preds, self.dist_transform(labels)))
            else:
                losses.append(loss(preds, labels))
            
        if len(losses) == 1: return losses[0]
        
        if self.loss_combine == 'sum':
            return torch.stack(losses).sum()
        elif self.loss_combine == 'mean':
            return torch.stack(losses).mean()
        elif self.loss_combine == 'log':
            return torch.log(torch.stack(losses).sum())
        else:
            raise NotImplementedError("Unsupported value for loss_combine. Please choose from 'sum', 'mean', or 'log'.")

# Reference : https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py
class BoundaryLoss(_Loss):
    def __init__(self, num_classes: int, one_hot: bool):
        super().__init__()
        self.num_classes = num_classes
        self.one_hot = one_hot

    def forward(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        loss = 0 
        dist_maps = dist_maps.to(probs.device)
        
        if self.one_hot:
            for c in range(self.num_classes):
                pc = probs[:, c, ...].type(probs.dtype).to(probs.device)
                dc = dist_maps[:, c, ...].type(probs.dtype).to(probs.device)
                # loss += torch.mean(torch.einsum("bkwh,bkwh->bkwh", pc, dc))
                loss += torch.einsum("bkwh,bkwh->bkwh", pc, dc).mean()
            
            # return torch.log(loss) / (self.num_classes*probs.size(0))
            return loss / (self.num_classes*probs.size(0))
        else:
            pc = probs.to(probs.device)
            dc = dist_maps.to(probs.device)
            # loss += torch.mean(torch.einsum("bkwh,bkwh->bkwh", pc, dc))

            return torch.einsum("bkdwh,bkdwh->bkdwh", pc, dc).mean() / probs.size(0)

class MultiNeighborLoss(_Loss):
    def __init__(self, num_classes: int, include_background: bool = True, reduction: str = "mean"):
        super(MultiNeighborLoss, self).__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.reduction = reduction
        self.max_angles = self.num_classes * (self.num_classes - 1) // 2
    
    def forward(self, probs: torch.Tensor, labels: torch.Tensor):
        assert probs.ndim == labels.ndim and probs.ndim == 5, "The dimensions of tensors 'probs' and 'labels' should be 5."
        
        delta = torch.zeros(probs.size(0), self.max_angles).to(probs.device)  # delta를 텐서로 초기화
        
        for i in range(probs.size(0)):
            p_angles, l_angles = self.compute_angles(probs[i, ...]), self.compute_angles(labels[i, ...])
            delta[i, :] = torch.square(p_angles - l_angles)
                    
        if self.reduction == "mean":
            return torch.mean(delta)
        
    def compute_angles(self, t: torch.Tensor):
        idx = 0 
        angles = torch.zeros(self.max_angles).to(t.device)
        centroids = [None for _ in range(self.num_classes)]
        
        for i in range(self.num_classes):
            if i == 0: continue # do not consider backgrounds
            _, z, y, x = torch.where(t == i)
            centroids[i] = torch.stack([torch.mean(x.float()), torch.mean(y.float()), torch.mean(z.float())])
        
        for i in range(1, self.num_classes):
            for j in range(i+1, self.num_classes):
                if centroids[i] is None or centroids[j] is None:
                    pass
                else:
                    m, n = centroids[i], centroids[j] # 2 vectors to calculate angles
                    angle = torch.acos(torch.dot(m, n) / (torch.norm(m) * torch.norm(n)))
                    
                    if torch.isnan(angle):
                        angle = torch.randn((1, )).to(x.device)
                    
                    angles[idx] = angle
                    idx += 1
                    
        return angles
    
    
                    
                    
if __name__ == "__main__":
    num_classes = 16
    device = torch.device("cuda:1")
    for _ in range(100):
        probs = torch.randint(0, num_classes, (10, num_classes, 96, 96, 96)).to(device)
        labels = torch.randint(0, num_classes, (10, num_classes, 96, 96, 96)).to(device)
        
        loss = MultiNeighborLoss(num_classes)
        
        l = loss(probs, labels)
        
        print(f"loss : {l:.4f}")
    
    