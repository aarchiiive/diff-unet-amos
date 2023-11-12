from typing import List, Sequence, Union, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from monai.networks.blocks import Convolution

from ..diffusion import get_timestep_embedding, nonlinearity, TimeStepEmbedder

"""

[References] 
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/attentionunet.py

"""

def conv_bn(spatial_dims: int) -> Sequence[nn.Module]:
    if spatial_dims == 2:
        return nn.Conv2d, nn.BatchNorm2d
    elif spatial_dims == 3:
        return nn.Conv3d, nn.BatchNorm3d
    else:
        raise NotImplementedError(f"spatial_dims value of {spatial_dims} is not supported. Please use ndim 2 or 3.")


class MaxPool(nn.Module):
    def __init__(self, spatial_dims: int, pool_size: int) -> None:
        super().__init__()
        if spatial_dims == 2:
            self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        elif spatial_dims == 3:
            self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        else:
            raise NotImplementedError(f"spatial_dims value of {spatial_dims} is not supported. Please use ndim 2 or 3.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

class Conv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
        super(Conv,self).__init__()
        conv, batch_norm = conv_bn(spatial_dims)
        
        self.conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            batch_norm(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            batch_norm(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
        super(UpConv, self).__init__()
        conv, batch_norm = conv_bn(spatial_dims)
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            batch_norm(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class AttentionLayer(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        f_g = f_l = out_channels
        f_int = out_channels // 2
        
        conv, batch_norm = conv_bn(spatial_dims)
            
        self.up = UpConv(spatial_dims, in_channels, out_channels, dropout)
        self.w_g = nn.Sequential(
            conv(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            batch_norm(f_int)
        )
        
        self.w_x = nn.Sequential(
            conv(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            batch_norm(f_int)
        )

        self.psi = nn.Sequential(
            conv(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            batch_norm(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.out = Conv(spatial_dims, in_channels, out_channels, dropout)
        
    def forward(self, x1, x2):
        g = self.up(x1)
        
        # attention
        psi = self.relu(self.w_g(g) + self.w_x(x2))
        psi = self.psi(psi)
        x = x2 * psi
        # concatenate
        x = torch.cat((x, g),dim=1)     
        # print(f"out : {x.shape}")
        
        return self.out(x)
    
    
class AttentionUNet(nn.Module):
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 3, 
                 out_channels: int = 1,
                 features: Sequence[int] = [64, 128, 256, 512, 1024],
                 kernel_size: int = 1,
                 stride: int = 1,
                 pool_size: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        features = reversed(features)
        self.head = Conv(spatial_dims, in_channels, features[0], dropout)
        self.down = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.down.append(nn.Sequential(
                MaxPool(spatial_dims, pool_size),
                Conv(spatial_dims, features[i], features[i+1], dropout)
            ))
            
        features = reversed(features)
        self.up = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.up.append(AttentionLayer(spatial_dims, features[i], features[i+1], dropout))
        
        self.out = nn.Conv3d(features[-1], out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsampling
        embeddings = self.downsample(x)
        embeddings.reverse()
        
        # upsampling
        x = self.upsample(embeddings)
        
        return self.out(x)
    
    def downsample(self, x: torch.Tensor) -> List[torch.Tensor]:
        _x = [self.head(x)]
        for i in range(len(self.down)):
            _x.append(self.down[i](_x[-1]))
        
        return _x
    
    def upsample(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        x = None
        for i in range(len(self.up)):
            x = self.up[i](embeddings[i], embeddings[i+1]) if x is None else self.up[i](x, embeddings[i+1])
        
        return x
    
    
""" Attention-UNet using timesteps """

class TwoConv(nn.Sequential):
    """two convolutions."""

    # @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.conv_1(x)
        return x 
     
class AttentionCatLayer(AttentionLayer):
    def __init__(self, 
                 spatial_dims: int, 
                 in_channels: int, 
                 cat_channels: int,
                 out_channels: int, 
                 act: Union[str, tuple],
                 norm: Union[str, tuple],
                 bias: bool,
                 dropout: float,
                 halves: bool = True):
        super().__init__(spatial_dims, in_channels, out_channels, dropout)

        up_channels = in_channels // 2 if halves else in_channels
        self.convs = TwoConv(spatial_dims, cat_channels + up_channels, out_channels, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, temb: torch.Tensor):
        # print(f"x : {x.shape}")
        # print(f"x_e : {x_e.shape}")
        x0: torch.Tensor = super().forward(x, x_e)
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x0 = torch.nn.functional.pad(x0, sp, "replicate")
        x = self.convs(torch.cat([x_e, x0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        
        return x
    
class AttentionUNetEncoder(nn.Module):
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 3, 
                 features: Sequence[int] = [64, 128, 256, 512, 1024],
                 pool_size: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.head = Conv(spatial_dims, in_channels, features[0], dropout)
        self.down = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.down.append(nn.Sequential(
                MaxPool(spatial_dims, pool_size),
                Conv(spatial_dims, features[i], features[i+1], dropout)
            ))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _x = [self.head(x)]
        for i in range(len(self.down)):
            _x.append(self.down[i](_x[-1]))
        
        return _x
    
class AttentionUNetDecoder(nn.Module):
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 3, 
                 out_channels: int = 1,
                 act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: Union[str, tuple] = ("instance", {"affine": True}),
                 bias: bool = True,
                 features: Sequence[int] = [64, 128, 256, 512, 1024],
                 kernel_size: int = 1,
                 stride: int = 1,
                 pool_size: int = 2,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        if isinstance(features , tuple): features = list(features)
        self.temb = TimeStepEmbedder()
        self.head = Conv(spatial_dims, in_channels, features[0], dropout)
        self.down = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            # print(f"{features[i]}, {features[i+1]}")
            self.down.append(nn.Sequential(
                MaxPool(spatial_dims, pool_size),
                Conv(spatial_dims, features[i], features[i+1], dropout)
            ))
            
        features.reverse()
        self.up = nn.ModuleList()
        # print()
        
        for i in range(len(features[:-1])):
            # print(f"{features[i]}, {features[i+1]}")
            self.up.append(AttentionCatLayer(
                spatial_dims=spatial_dims, 
                in_channels=features[i], 
                cat_channels=features[i+1], 
                out_channels=features[i+1] if features[i] != features[i+1] else features[i]*2, 
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
            ))
        
        self.out = nn.Conv3d(features[-1], out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, embeddings: List[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t)
        x = torch.cat([image, x], dim=1)
        
        # downsampling
        _x = self.downsample(x, embeddings)
        
        # reverse embeddings
        _x.reverse()
        
        # upsampling
        x = self.upsample(_x, temb)
        
        return self.out(x)
    
    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        return self.temb.dense[1](temb)
    
    def downsample(self, x: torch.Tensor, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        _x = [self.head(x) + embeddings[0]]
        for i in range(len(self.down)):
            d = self.down[i](_x[-1])  + embeddings[i + 1]
            _x.append(d)
        
        return _x
    
    def upsample(self, _x: List[torch.Tensor], temb: torch.Tensor) -> torch.Tensor:
        x = None
        for i in range(len(self.up)):
            # print(i, _x[i].shape, _x[i+1].shape)
            x = self.up[i](_x[i], _x[i+1], temb) if x is None else self.up[i](x, _x[i+1], temb)
        
        return x