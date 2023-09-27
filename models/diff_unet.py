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
                 timesteps: int,
                 mode: str,
                 ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.mode = mode
        
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDecoder(3, num_classes+1, num_classes, [64, 64, 128, 256, 512, 64], 
                                      act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", timesteps)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(timesteps, [timesteps]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(timesteps, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(timesteps)
        

    def forward(self, 
                image: torch.Tensor = None, 
                x: torch.Tensor = None, 
                pred_type: str = None, 
                step=None, 
                embedding=None):
        if image is not None and x is not None: assert image.device == x.device
        
        if pred_type == "q_sample":
            _sample, _t, _noise = [], [], []
            for i in range(len(x)):
                batch = x[i, ...].unsqueeze(0)
                noise = torch.randn_like(batch).to(x.device)
                t, _ = self.sampler.sample(batch.shape[0], batch.device)
                sample = self.diffusion.q_sample(batch, t, noise)
                _sample.append(sample)
                _noise.append(noise)
                _t.append(t.unsqueeze(0))
                
            return torch.cat(_sample, dim=0), torch.cat(_t, dim=0), torch.cat(_noise, dim=0)

        elif pred_type == "denoise":
            assert image.size(0) == x.size(0) == step.size(0)
            res = []
            for i in range(len(image)):
                batch = image[i, ...].unsqueeze(0)
                x_batch = x[i, ...].unsqueeze(0)
                embeddings = self.embed_model(batch)
                res.append(self.model(x_batch, t=step[i, ...], embeddings=embeddings, image=batch))
                
            return torch.cat(res, dim=0)

        elif pred_type == "ddim_sample":
            res = []
            for i in range(len(image)):
                batch = image[i, ...].unsqueeze(0)
                embeddings = self.embed_model(batch)
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, 
                                                                    (1, self.num_classes, *image.shape[2:]), 
                                                                    model_kwargs={"image": batch, "embeddings": embeddings})
                if self.mode == "train":
                    res.append(sample_out["pred_xstart"].to(image.device))
                elif self.mode == "test":
                    sample_return = torch.zeros_like((1, self.num_classes, *image.shape[2:])).to(image.device)
                    all_samples = sample_out["all_samples"]
                    
                    for sample in all_samples:
                        sample_return += sample.to(image.device)

                    res.append(sample_return) 
                
            return torch.cat(res, dim=0)