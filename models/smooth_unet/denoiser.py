from typing import Optional, Sequence, Union

import torch 
import torch.nn as nn 

from models.basic_unet import BasicUNetRDenoiser, get_timestep_embedding, nonlinearity


class SmoothUNetDenoiser(BasicUNetRDenoiser):
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
