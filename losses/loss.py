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
        
        for l in losses.split(','):
            if l == "mse":
                self.losses.append(MSELoss())
            elif l == "ce":
                self.losses.append(CrossEntropyLoss())
            elif l == "bce":
                self.losses.append(BCEWithLogitsLoss())
            elif l == "dice":
                self.losses.append(DiceLoss(sigmoid=True))
            elif l == "boundary":
               self.losses.append(BoundaryLoss(num_classes, one_hot))
            elif l == "dice_ce":
                self.losses.append(DiceCELoss(sigmoid=True))
            elif l == "dice_focal":
                self.losses.append(DiceFocalLoss(sigmoid=True))
            elif l == "generalized_dice":
               self.losses.append(GeneralizedDiceLoss(sigmoid=True))

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


# class HausdorffLoss:
#     """
#     Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
#     """
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert simplex(target)
#         assert probs.shape == target.shape

#         B, K, *xyz = probs.shape  # type: ignore

#         pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
#         tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
#         assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

#         target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
#                                               for b in range(B)], axis=0)
#         assert target_dm_npy.shape == tc.shape == pc.shape
#         tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

#         pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
#         pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
#                                             for b in range(B)], axis=0)
#         assert pred_dm_npy.shape == tc.shape == pc.shape
#         pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

#         delta = (pc - tc)**2
#         dtm = tdm**2 + pdm**2

#         multipled = torch.einsum("bkwh,bkwh->bkwh", delta, dtm)

#         loss = multipled.mean()

#         return loss


# # Reference : https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L6
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,D,H,W => N,C,D*H*W
#             input = input.transpose(1, 2)    # N,C,D*H*W => N,D*H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,D*H*W,C => N*D*H*W,C
#         target = target.view(-1, 1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1 - pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()