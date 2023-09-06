# Reference : https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py

from typing import Callable, Union, List, Tuple, cast

import numpy as np
from functools import partial
from operator import itemgetter

import torch
from torch import Tensor
from torchvision import transforms

from .utils import simplex, one_hot, probs2one_hot, class2one_hot, one_hot2dist, one_hot2hd_dist

class BoundaryLoss:
    def __init__(self, num_classes: int):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.num_classes = num_classes

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        # assert simplex(probs)
        # assert not one_hot(dist_maps)

        loss = 0 
        for c in self.num_classes:
            pc = probs[:, c, ...].type(torch.float32)
            dc = dist_maps[:, c, ...].type(torch.float32)
            loss += torch.einsum("bcdwh,bcdwh->bcdwh", pc, dc).mean()

        return loss / self.num_classes

class HausdorffLoss:
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = torch.einsum("bkwh,bkwh->bkwh", delta, dtm)

        loss = multipled.mean()

        return loss