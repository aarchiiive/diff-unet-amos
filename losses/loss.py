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
               self.losses.append(MultiNeighborLoss(num_classes=num_classes, include_background=include_background))
            elif name == "hausdorff_er":
                self.losses.append(HausdorffERLoss(num_classes=num_classes))

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
            return torch.log(1 + torch.stack(losses).sum())
        else:
            raise NotImplementedError("Unsupported value for loss_combine. Please choose from 'sum', 'mean', or 'log'.")


# simple torch implementation of ofdistance_transform_edt from scipy
def distance_transform_edt(t: torch.Tensor):
    dist = torch.zeros_like(t, dtype=t.dtype).to(t.device)
    for x in range(t.shape[0]):
        for y in range(t.shape[1]):
            if t[x, y] == 0:
                continue
            min_dist = float('inf')
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    if t[i, j] == 0:
                        d = ((x - i)**2 + (y - j)**2)**0.5
                        if d < min_dist:
                            min_dist = d
            dist[x, y] = min_dist
    return dist

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

# References : https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""
    
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    # @torch.no_grad()
    def distance_field(self, t: torch.Tensor) -> torch.Tensor:
        field = torch.zeros_like(t).cuda()
        
        for batch in range(len(t)):
            fg_mask = t[batch] > 0.5
            
            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = distance_transform_edt(fg_mask) # Replace this
                bg_dist = distance_transform_edt(bg_mask) # Replace this

                field[batch] = fg_dist + bg_dist
                
        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.ndim == target.ndim, "Prediction and target need to be of same dimension"

        pred_dt = self.distance_field(pred)
        target_dt = self.distance_field(target)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        return loss


class HausdorffERLoss(_Loss):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, 
                 num_classes: int, 
                 alpha: float = 2.0, 
                 erosions: int = 5,
                 scaler: str = 'log'):
        super(HausdorffERLoss, self).__init__()
        self.num_classes  = num_classes
        self.alpha = alpha
        self.erosions = erosions
        self.scaler = scaler
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        bound = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        self.kernel = torch.stack([bound, cross, bound]).unsqueeze(0).unsqueeze(0) / 7.0
        self.kernel = self.kernel.repeat(1, self.num_classes, *self.kernel.shape[2:])

    # @torch.no_grad()
    def perform_erosion(self, probs: torch.Tensor, labels: torch.Tensor):
        bound = ((probs - labels) ** 2).float()
        eroted = torch.zeros_like(bound)
        self.kernel = self.kernel.to(dtype=bound.dtype, device=bound.device)
        
        for i in range(len(bound)):
            for k in range(self.erosions):
                # Note : conv2d does not work entirely same with scipy.ndimage.convolve
                dilation = F.conv3d(bound[i].unsqueeze(0), self.kernel, padding=4)
                
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0
                ptp = torch.max(erosion) - torch.min(erosion)
                
                if ptp != 0:
                    erosion = (erosion - erosion.min()) / ptp

                eroted[i] += erosion.squeeze(0) * (k + 1) ** self.alpha
    
        nan_indices = torch.isnan(eroted)
        if nan_indices.any():
            eroted[nan_indices] = 0

        return eroted

    def forward(self, probs: torch.Tensor, labels: torch.Tensor):
        assert probs.ndim == labels.ndim == 5, "The dimensions of probs and labels should be same and 5."

        eroted = self.perform_erosion(probs, labels)
        loss = eroted.mean()
        
        if self.scaler == "log":
            return torch.log(1 + loss)
        elif self.scaler == "sqrt":
            return torch.sqrt(loss)
        elif self.scaler == "sqrt_log":
            return torch.sqrt(torch.log(1 + loss))
        
        
class MultiNeighborLoss(_Loss):
    def __init__(self, 
                 num_classes: int, 
                 include_background: bool = True, 
                 reduction: str = "mean", 
                 centroid_method: str = "mean"):
        super(MultiNeighborLoss, self).__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.reduction = reduction
        self.centroid_method = centroid_method
        self.max_angles = self.num_classes * (self.num_classes - 1) // 2
        
        if not include_background: self.num_classes += 1
    
    def forward(self, probs: torch.Tensor, labels: torch.Tensor):
        assert probs.ndim == labels.ndim == 5, "The dimensions of probs and labels should be same and 5."
        
        delta = []
        for i in range(probs.size(0)):
            p_angles, l_angles = self.compute_angles(torch.sigmoid(probs[i, ...])), self.compute_angles(labels[i, ...])
            delta.append(torch.square(p_angles - l_angles))
                    
        if self.reduction == "mean":
            return torch.mean(torch.cat(delta))
        
    def compute_angles(self, t: torch.Tensor) -> torch.Tensor:
        idx = 0 
        angles = torch.zeros(self.max_angles).to(t.device)
        centroids = [None for _ in range(self.num_classes)]
        
        t = torch.argmax(t, dim=0)
        
        for i in range(self.num_classes):
            if i == 0: continue # do not consider backgrounds
            z, y, x = torch.where(t == i)
            centroids[i] = torch.stack(self.compute_centroids(x, y, z))
        
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
    
    def compute_centroids(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        if self.centroid_method == "mean":
            return [torch.mean(x.float()), torch.mean(y.float()), torch.mean(z.float())]
        else:
            raise NotImplementedError(f"The centroid method is not supported. : {self.centroid_method}")
                    
if __name__ == "__main__":
    num_classes = 16
    device = torch.device("cuda:1")
    loss = HausdorffERLoss(num_classes)
    for _ in range(100):
        probs = torch.randint(0, num_classes, (10, num_classes, 96, 96, 96)).to(device)
        labels = torch.randint(0, num_classes, (10, num_classes, 96, 96, 96)).to(device)
        
        l = loss(probs, labels)
        
        print(f"loss : {l:.4f}")
    
    