from typing import Sequence, Tuple, Union, List

import torch 
import torch.nn as nn 

# from layers.basic_unet import BasicUNetEncoder
from layers.pretrained.basic_unet import BasicUNetEncoder
from layers.basic_unet_denoise import BasicUNetDecoder

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler


class DiffUNet(nn.Module):
    def __init__(self, 
                 image_size: Union[int, Tuple[int, int]],
                 spatial_size: int,
                 num_classes: int,
                 device: torch.device,
                 mode: str,
                 ):
        super().__init__()
        
        if isinstance(image_size, tuple):
            width, height = image_size
        elif isinstance(image_size, int):
            width = height = image_size
            
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.mode = mode
        
        # deprecated soon
        # if pretrained:
        #     from layers.pretrained.basic_unet import BasicUNetEncoder
        # else:
        #     from layers.basic_unet import BasicUNetEncoder
            
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDecoder(3, num_classes+1, num_classes, [64, 64, 128, 256, 512, 64], 
                                      act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)
        

    def forward(self, image: torch.Tensor = None, x: torch.Tensor = None, pred_type: str = None, step=None, embedding=None):
        if image is not None and x is not None: assert image.device == x.device
        
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, 
                                                                (image.shape[0], self.num_classes, *image.shape[2:]), 
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            if self.mode == "train":
                sample_out = sample_out["pred_xstart"].to(image.device)
                return sample_out
            elif self.mode == "test":
                sample_return = torch.zeros_like((image.shape[0], self.num_classes, *image.shape[2:])).to(image.device)
                all_samples = sample_out["all_samples"]
                
                for sample in all_samples:
                    sample_return += sample.to(image.device)

                return sample_return