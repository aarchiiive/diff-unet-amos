from __future__ import annotations

import sys
import time
import collections.abc

from copy import deepcopy
from collections.abc import Callable, Sequence
from typing import Dict
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
        
    def compute_distance(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        org = labels
        labels = F.one_hot(labels.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        B, C, W, H, D = labels.shape
        
        # Pre-compute all indices
        indices = torch.stack(torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            torch.arange(D),
            indexing='ij'
        ), dim=-1).float().to(labels.device)
        
        centroids = torch.zeros(self.num_classes, 3, device=labels.device)
        
        for i in range(self.num_classes):
            class_mask = labels[:, i, :, :, :]
            class_mask = class_mask.squeeze(0).bool()
            
            masked_indices = indices[class_mask]
            if masked_indices.numel() > 0:
                centroid = masked_indices.mean(dim=0)
                centroids[i] = centroid
                
        centroids = centroids[:, None, None, None, None, :]
        
        distances = torch.norm(indices[None, None, :, :, :, :] - centroids, dim=-1).squeeze(1)
        distances_dict = {f"distance_{i}": distances[i, :, :, :].unsqueeze(0) for i in range(self.num_classes)}
        
        return distances_dict
    
    def label_smoothing(self, labels: torch.Tensor) -> torch.Tensor:
        org = labels
        labels = F.one_hot(labels.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        # org = labels.squeeze(0)
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
        distances = torch.norm(indices[None, None, :, :, :, :] - centroids, dim=-1).squeeze(1).unsqueeze(0)
        
        # labels = 1 / (distances.squeeze(1) + self.epsilon) * self.smoothing_alpha
        # labels = self.damped_sine_wave(distances.squeeze(1)) * self.smoothing_alpha # wandb : diff-swin-unetr-btcv-10
        
        # labels = torch.abs(org - labels)
        
        return distances
    
    def damped_sine_wave(self, x: torch.Tensor, lambda_decay=0.05, omega=0.1, phi=0) -> torch.Tensor:
        signal = torch.exp(-lambda_decay * x) * torch.sin(omega * x + phi)
        return signal
    
    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]
        item = self.image_loader(item)
        
        if self.smoothing_type == "distance":
            for k, v in self.compute_distance(item['label']).items():
                item[k] = v
            
        # print(self.data[idx], item['image'].shape, item['label'].shape, item['distance_0'].shape)
            
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        item = self.transform(item, end=first_random, threading=True)
        
        print("After transform:", item['image'].shape, item['label'].shape, item['distance_0'].shape, item['distance_1'].shape)
        
        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item
    
    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
    