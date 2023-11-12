from typing import Sequence

# from layers.basic_unet import BasicUNetEncoder
from .basic_unet.pretrained import BasicUNetEncoder
from .basic_unet import BasicUNetRDenoiser
from .diffusion.diffusion import Diffusion


class DiffUNet(Diffusion):
    def __init__(
        self, 
        spatial_dims: int = 3,
        in_channels: int = 3, 
        out_channels: int = 1,
        image_size: int = 96,
        spatial_size: int = 96,
        features: Sequence[int] = [64, 64, 128, 256, 512, 64],
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
        self.embed_model = BasicUNetEncoder(3, in_channels, 2, features)
        self.model = BasicUNetRDenoiser(3, out_channels+1, out_channels, features, 
                                        act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))