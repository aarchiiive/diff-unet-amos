from typing import Optional, Sequence, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from models.diff_unet import DiffUNet
from models.basic_unet import BasicUNetEncoder

from .layers import SmoothLayer

class SmoothUNetEncoder(BasicUNetEncoder):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        image_size: int = 256,
        spatial_size: int = 96,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
        ndims: int = 5, 
        p: int = 1,
        smoothing: bool = True,
    ):
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            features,
            act,
            norm,
            bias,
            dropout,
            upsample,
            dimensions,
        )
        # for short code
        w = h = image_size
        d = spatial_size
        
        self.smoothing = smoothing
        self.smooth = [SmoothLayer(features[0], d, w, h)]
        # self.ffparser = [FFParser(features[0], d, w, h)]
        
        for i, f in enumerate(features[1:4]):
            self.smooth.append(SmoothLayer(f, d // (2**(i+1)), w // (2**(i+1)), h // (2**(i+1))))
            # self.ffparser.append(FFParser(f, d // (2**(i+1)), w // (2**(i+1)), h // (2**(i+1))))
            
        self.smooth = nn.ModuleList(self.smooth)
        # self.ffparser = nn.ModuleList(self.ffparser)
        self.down = nn.ModuleList([self.down_1,
                                   self.down_2,
                                   self.down_3,
                                   self.down_4])
        
    def forward(self, x: torch.Tensor):
        _x = [self.conv_0(x)]
        
        for i in range(4):
            s = self.smooth[i](_x[i])
            # w = self.ffparser[i](s)
            _x.append(self.down[i](s))
        
        return _x