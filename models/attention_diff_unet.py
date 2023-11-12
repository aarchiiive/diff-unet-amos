from typing import Sequence, Tuple, Union, List

from .diffusion.diffusion import Diffusion
from .attention_unet import AttentionUNetEncoder, AttentionUNetDecoder

class AttentionDiffUNet(Diffusion):
    def __init__(
        self, 
        spatial_dims: int = 3,
        in_channels: int = 3, 
        out_channels: int = 1,
        image_size: int = 96,
        spatial_size: int = 96,
        features: Sequence[int] = [32, 64, 128, 256, 512], # [64, 128, 256, 512, 1024],
        dropout: float = 0.2,
        timesteps: int = 1000,
        mode: str = "train"
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            spatial_size=spatial_size,
            features=features,
            dropout=dropout,
            timesteps=timesteps,
            mode=mode,
        )      
        self.embed_model = AttentionUNetEncoder(spatial_dims, in_channels, features=features, dropout=dropout)    
        self.model = AttentionUNetDecoder(spatial_dims, out_channels+1, out_channels, features=features, dropout=dropout)    
        