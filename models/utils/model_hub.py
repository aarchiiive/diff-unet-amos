from typing import Any, Tuple

from monai.networks.nets.swin_unetr import SwinUNETR

from models.diff_unet import DiffUNet
from models.smooth_diff_unet import SmoothDiffUNet
from models.diff_swin_unetr import DiffSwinUNETR
# from models.attention_unet import AttentionUNet
from models.attention_diff_unet import AttentionDiffUNet

class ModelHub:
    def __init__(self) -> None:
        pass
    
    def __call__(self, model_name: str, **kwargs: Any) -> Any:
        if model_name == "diff_unet":
            model = DiffUNet(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
            )
        elif model_name == "smooth_diff_unet":
            model = SmoothDiffUNet(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
            )
        elif model_name == "attention_diff_unet":
            model = AttentionDiffUNet(**kwargs)
        elif model_name == "diff_swin_unetr":
            model = DiffSwinUNETR(
                image_size=self.parse_image_size(**kwargs),
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                noise_ratio=kwargs['noise_ratio'],
                feature_size=48
            )
        elif model_name == "swin_unetr":
            model = SwinUNETR(
                img_size=self.parse_image_size(**kwargs),
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                feature_size=48
            )
        # elif model_name == "attention_unet":
        #     model = AttentionUNet(in_channels=1,
        #                           out_channels=kwargs['num_classes'])
        
        else:
            raise ValueError(f"Invalid model type: {model_name}")
        
        return model

    def parse_image_size(self, **kwargs) -> Tuple[int, int, int]:
        return (kwargs['spatial_size'], kwargs['image_size'], kwargs['image_size'])