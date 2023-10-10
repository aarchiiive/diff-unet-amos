from typing import Optional, Sequence, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from models.diff_unet import DiffUNet
from layers.ffparser import FFParser
from layers.basic_unet import BasicUNetEncoder
from layers.basic_unet_denoise import get_timestep_embedding, nonlinearity, BasicUNetDecoder


class SmoothLayer(nn.Module):
    def __init__(self, 
                 in_features,
                 spatial_size=96,
                 width=256,
                 height=256,
                 ndims=5, 
                 p=1, 
                 k=1.0) -> None:
        super().__init__()
        assert ndims == 4 or ndims == 5
        if ndims == 4: # (B, D, W, H) or (B, D, H, W)
            self.dims = (1, 2, 3)
        if ndims == 5: # (B, C, D, W, H) or (B, C, D, H, W)
            self.dims = (2, 3, 4)
            
        # for short code
        w, h = width, height
        d = spatial_size
        
        self.p = p
        self.k = k
        self.padding = (p,) * len(self.dims) * 2
        self.shifts = [(p, 0, 0),
                       (-p, 0, 0),
                       (0, p, 0),
                       (0, -p, 0),
                       (0, 0, p),
                       (0, 0, -p)]
        self.weights = nn.Parameter(torch.randn(in_features, d, w, h) * 0.5)
        
    def forward(self, x: torch.Tensor):
        p = self.p
        _x = F.pad(x.clone(), self.padding, "constant", 0)
        laplacian = -6 * _x
        for shift in self.shifts:
            laplacian = laplacian + torch.roll(_x, shifts=shift, dims=self.dims)
        
        laplacian = laplacian[..., p:-p, p:-p, p:-p] * self.weights
        x = x + laplacian
        
        return x


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


class SmoothUNetDecoder(BasicUNetDecoder):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("layer", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
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
        self.smoothing = smoothing
        self.upcat = nn.ModuleList([self.upcat_4,
                                    self.upcat_3,
                                    self.upcat_2,
                                    self.upcat_1])
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, embeddings: torch.Tensor = None, image: torch.Tensor = None):
        t_emb = get_timestep_embedding(t, 128)
        t_emb = self.temb.dense[0](t_emb)
        t_emb = nonlinearity(t_emb)
        t_emb = self.temb.dense[1](t_emb)

        x = torch.cat([image, x], dim=1)
            
        x0 = self.conv_0(x, t_emb) + embeddings[0]
        x1 = self.down_1(x0, t_emb) + embeddings[1]
        x2 = self.down_2(x1, t_emb) + embeddings[2]
        x3 = self.down_3(x2, t_emb) + embeddings[3]
        x4 = self.down_4(x3, t_emb) + embeddings[4]
        
        u4 = self.upcat[0](x4, x3, t_emb)
        u3 = self.upcat[1](u4, x2, t_emb)
        u2 = self.upcat[2](u3, x1, t_emb)
        u1 = self.upcat[3](u2, x0, t_emb)

        logits = self.final_conv(u1)
        return logits

    
class SmoothDiffUNet(DiffUNet):
    def __init__(
        self, 
        image_size,
        spatial_size,
        num_classes,
        device,
        mode,
    ):
        super().__init__(
            image_size,
            spatial_size,
            num_classes,
            device,
            mode,
        )
        self.embed_model = SmoothUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64], image_size, spatial_size)
        self.model = SmoothUNetDecoder(3, num_classes+1, num_classes, [64, 64, 128, 256, 512, 64], 
                                       act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}), smoothing=False)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        return super(SmoothDiffUNet, self).forward(image, x, pred_type, step, embedding)