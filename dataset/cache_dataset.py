from __future__ import annotations

import sys
import time
import collections.abc

from copy import deepcopy
from collections.abc import Callable, Sequence
from multiprocessing.managers import ListProxy

import torch

from torch.nn import functional as F
from torch.utils.data import Subset
from torch.multiprocessing import Manager

from monai.data import CacheDataset
from monai.data.utils import pickle_hashing
from monai import transforms
from monai.transforms import (
    LoadImaged,
    RandomizableTrait,
    Transform,
    convert_to_contiguous,
)

class LabelSmoothingCacheDataset(CacheDataset):
    def __init__(
        self,
        data: Sequence,
        transform: Sequence[Callable] | Callable | None = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int | None = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_as_key: bool = False,
        hash_func: Callable[..., bytes] = pickle_hashing,
        runtime_cache: bool | str | list | ListProxy = False,
        num_classes: int = 14,
        smoothing_alpha: float = 0.3,
        smoothing_type: str = "distance",
        epsilon: float = 1e-6,
    ) -> None:
        self.num_classes = num_classes
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_type = smoothing_type
        self.epsilon = epsilon
        
        self.image_loader = transforms.Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ])
        
        super().__init__(
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            hash_as_key=hash_as_key,
            hash_func=hash_func,
            runtime_cache=runtime_cache,
        )
    
    def label_smoothing(self, labels: torch.Tensor) -> torch.Tensor:
        labels = F.one_hot(labels.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        org = labels.squeeze(0)
        B, C, W, H, D = labels.shape
        
        # Pre-compute all indices
        indices = torch.stack(torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            torch.arange(D),
            indexing='ij'
        ), dim=-1).float().to(labels.device) # Shape: [W, H, D, 3]

        # Initialize tensor to hold centroids
        centroids = torch.zeros(self.num_classes, 3, device=labels.device)
        
        # Calculate class-wise centroids
        for i in range(self.num_classes):
            # Extract the i-th channel for class i
            class_mask = labels[:, i, :, :, :]

            # Ensure the mask is 4D and of boolean type
            class_mask = class_mask.squeeze(0).bool()  # Reduce first dimension and convert to bool

            # Ensure indices and class_mask have compatible shapes
            if class_mask.shape != indices.shape[:-1]:
                raise ValueError(f"Shape mismatch: class_mask has shape {class_mask.shape}, but indices has shape {indices.shape}")

            masked_indices = indices[class_mask]
            if masked_indices.numel() > 0:  # Check if there are any elements
                centroid = masked_indices.mean(dim=0)
                centroids[i] = centroid
                
        # Expand centroids to match broadcasting dimensions
        centroids = centroids[:, None, None, None, None, :]

        # Calculate distances with correct broadcasting
        distances = torch.norm(indices[None, None, :, :, :, :] - centroids, dim=-1)
        
        labels = 1 / (distances.squeeze(1) + self.epsilon) * self.smoothing_alpha
        labels = torch.abs(org - labels)
        
        # i = 48
        
        # for x in range(200, W):
        #     for y in range(200, H):
        #         if org[:, x, y, i][0] < 1:
        #             print("="*200)
        #             print("[org]")
        #             print(org[:, x, y, i])
        #             # print("[distances]")
        #             # print(distances[:, 0, x, y, i])
        #             print("[labels]")
        #             print(labels[:, x, y, i])
        
        return labels
    
    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]
        item = self.image_loader(item)
        
        if self.smoothing_type == "distance":
            item['label'] = self.label_smoothing(item['label'])
            
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        item = self.transform(item, end=first_random, threading=True)

        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item