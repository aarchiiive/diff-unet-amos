from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.utils import ensure_tuple_rep


class CompositionalMixer(nn.Module):  
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        image_size: Sequence[int] | int = 96,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.image_size = ensure_tuple_rep(image_size, 3)    
        
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_channels, in_channels),    
            nn.Dropout(drop_rate)
        )
        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_channels, out_channels),    
            nn.Dropout(drop_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        
        x = x.view(B, C, -1).transpose(2, 1) # [B, C, D*H*W] -> [B, D*H*W, C]
        x0 = x
        
        x1 = self.layer_norm1(x)
        x1 = self.layer1(x1)
        
        x2 = self.layer_norm2(x0 + x1)
        x2 = torch.cat((x0, x2), dim=2)
        x2 = self.out(x2)
        
        out = x2.view(B, -1, D, H, W)
        
        return out
    
    
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        self.grid = torch.stack(grids)
        self.grid = torch.unsqueeze(self.grid, 0)
        self.grid = self.grid.type(torch.FloatTensor)

    def forward(self, src: torch.Tensor, flow: torch.Tensor):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
    

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
    
    
class VXM(nn.Module):
    def __init__(
        self,
        image_size: Sequence[int] | int = (96, 96, 96),
        in_channels: int = 13,
        out_channels: int = 13,
        ndims: int = 3,
        int_steps: int = 7,
        int_downsize: int = 1,
        interpolation: str = 'bilinear',
    ) -> None:
        super().__init__()
        assert ndims == 3, 'Only 3D is supported'
        
        self.flow = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None
            
        self.integrate = VecInt([int(dim / int_downsize) for dim in image_size], int_steps)
        self.transformer = SpatialTransformer([int(dim / int_downsize) for dim in image_size], interpolation)
        
        self._init_weights()
        
    def _init_weights(self):
        self.flow.weight = nn.Parameter(torch.normal(0, 1e-5, size=self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
    
    def forward(self, x: torch.Tensor, image: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        pos_flow = self.flow(x)
        if self.resize: pos_flow = self.resize(pos_flow)
        neg_flow = -pos_flow
        
        pos_flow = self.integrate(pos_flow)
        neg_flow = self.integrate(neg_flow)
        
        pos_flow = self.transformer(image, pos_flow)
        neg_flow = self.transformer(noise, neg_flow)
        
        # out = torch.cat((pos_flow, neg_flow), dim=1)    
        
        return pos_flow