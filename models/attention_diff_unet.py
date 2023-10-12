from typing import Sequence, Tuple, Union, List

import torch
import torch.nn as nn

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

from .attention_unet import AttentionUNetEncoder, AttentionUNetDecoder

class AttentionDiffUNet(nn.Module):
    def __init__(self, 
                  spatial_dims: int = 3,
                  in_channels: int = 3, 
                  out_channels: int = 1,
                  features: Sequence[int] = [32, 64, 128, 256, 512], # [64, 128, 256, 512, 1024],
                  dropout: float = 0.2,
                  timesteps: int = 1000,
                  mode: str = "train"):
        super().__init__()     
        self.num_classes = out_channels
        self.mode = mode

        self.embed_model = AttentionUNetEncoder(spatial_dims, in_channels, features=features, dropout=dropout)    
        self.model = AttentionUNetDecoder(spatial_dims, out_channels+1, out_channels, features=features, dropout=dropout)    
        
        betas = get_named_beta_schedule("linear", timesteps)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(timesteps, [timesteps]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.RESCALED_KL,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(timesteps, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.RESCALED_KL,
                                                )
        self.sampler = UniformSampler(timesteps)
    
    def forward(self, 
                image: torch.Tensor = None, 
                x: torch.Tensor = None,
                step: torch.Tensor = None,
                pred_type: str = None):
        if image is not None and x is not None: assert image.device == x.device
        
        if pred_type == "q_sample":
            return self.q_sample(x)
        elif pred_type == "denoise":
            return self.denoise(image, x, step)
        elif pred_type == "ddim_sample":
            return self.ddim_sample(image)
        else:
            raise NotImplementedError(f"No such prediction type : {pred_type}")
        
    def q_sample(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
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
    
    def denoise(self, image: torch.Tensor, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        assert image.size(0) == x.size(0) == step.size(0)
        res = []
        for i in range(len(image)):
            batch = image[i, ...].unsqueeze(0)
            x_batch = x[i, ...].unsqueeze(0)
            embeddings = self.embed_model(batch)
            res.append(self.model(x_batch, t=step[i, ...], embeddings=embeddings, image=batch))
            
        return torch.cat(res, dim=0)
    
    def ddim_sample(self, image: torch.Tensor) -> torch.Tensor:
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
                sample_return = torch.zeros((1, self.num_classes, *image.shape[2:])).to(image.device)
                all_samples = sample_out["all_samples"]
                
                for sample in all_samples:
                    sample_return += sample.to(image.device)

                res.append(sample_return) 
            
        return torch.cat(res, dim=0)