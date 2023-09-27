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

import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss

# References : https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32) / 5.0
        bound = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        self.kernel2D = cross
        self.kernel3D = torch.stack([bound, cross, bound]) / 7.0

    @torch.no_grad()
    def perform_erosion(self, pred, target, debug):
        bound = (pred - target) ** 2

        if bound.dim() == 5:
            kernel = self.kernel3D
        elif bound.dim() == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.dim()} is not supported.")

        eroted = torch.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):
            # debug
            erosions.append(bound[batch][0].clone())

            for k in range(self.erosions):
                dilation = F.conv2d(bound[batch].unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)
                
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                bound[batch] = erosion.squeeze(0)
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(erosion[0].clone())

        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(self, pred, target, debug=False):
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert pred.dim() == target.dim(), "Prediction and target need to be of same dimension"

        if debug:
            eroted, erosions = self.perform_erosion(pred, target, debug)
            return eroted.mean(), erosions
        else:
            eroted = self.perform_erosion(pred, target, debug)
            loss = eroted.mean()
            return loss
        
        
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
        assert probs.ndim == labels.ndim == 5, "The dimensions of tensors 'probs' and 'labels' should be 5."
        
        delta = []
        for i in range(probs.size(0)):
            p_angles, l_angles = self.compute_angles(probs[i, ...]), self.compute_angles(labels[i, ...])
            delta.append(torch.square(p_angles - l_angles))
                    
        if self.reduction == "mean":
            return torch.mean(torch.cat(delta))
        
    def compute_angles(self, t: torch.Tensor) -> torch.Tensor:
        idx = 0 
        angles = torch.zeros(self.max_angles).to(t.device)
        centroids = [None for _ in range(self.num_classes)]
        
        for i in range(self.num_classes):
            if i == 0: continue # do not consider backgrounds
            _, z, y, x = torch.where(t == i)
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
    for _ in range(100):
        probs = torch.randint(0, num_classes, (10, num_classes, 96, 96, 96)).to(device)
        labels = torch.randint(0, num_classes, (10, num_classes, 96, 96, 96)).to(device)
        
        loss = MultiNeighborLoss(num_classes)
        
        l = loss(probs, labels)
        
        print(f"loss : {l:.4f}")
    
    