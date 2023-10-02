from typing import Any, Tuple

from models.diff_unet import DiffUNet
from models.smooth_diff_unet import SmoothDiffUNet
from models.swin_unetr import SwinUNETR
from models.attention_unet import AttentionUNet
# from monai.networks.nets import AttentionUnet

class ModelHub:
    def __init__(self) -> None:
        pass
    
    def __call__(self, model_name: str, **kwargs: Any) -> Any:
        if model_name == "diff_unet":
            model = DiffUNet(**kwargs)
        elif model_name == "smooth_diff_unet":
            model = SmoothDiffUNet(**kwargs)
        elif model_name == "swin_unetr":
            model = SwinUNETR(img_size=self.parse_image_size(**kwargs),
                              in_channels=1,
                              out_channels=kwargs['num_classes'],
                              feature_size=48)
        elif model_name == "attention_unet":
            model = AttentionUNet(in_channels=1,
                                  out_channels=kwargs['num_classes'])
            # model = AttentionUnet(spatial_dims=3,
            #                       in_channels=1,
            #                       out_channels=kwargs['num_classes'],
            #                       channels=[64, 128, 256, 512, 1024],
            #                       strides=[2, 2, 2, 2, 2])
        else:
            raise ValueError(f"Invalid model type: {model_name}")
        
        return model

    def parse_image_size(self, **kwargs) -> Tuple[int, int, int]:
        return (kwargs['spatial_size'], kwargs['image_size'], kwargs['image_size'])