import torch 
import torch.nn as nn 

from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denoise import BasicUNetDecoder

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler


class DiffUNet(nn.Module):
    def __init__(self, 
                 image_size,
                 depth,
                 num_classes,
                 device,
                 pretrained,
                 mode,
                 ):
        super().__init__()
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
            
        self.depth = depth
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.pretrained = pretrained
        self.mode = mode
        
        self.embed_model = BasicUNetEncoder(3, 1, num_classes, [64, 64, 128, 256, 512, 64])
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
        

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
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
                                                                (1, self.num_classes, self.depth, self.width,  self.height), 
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            if self.mode == "train":
                sample_out = sample_out["pred_xstart"].to(self.device)
                return sample_out
            elif self.mode == "test":
                sample_return = torch.zeros((1, self.num_classes, self.depth, self.width, self.height)).to(self.device)
                all_samples = sample_out["all_samples"]
                index = 0
                
                for sample in all_samples:
                    sample_return += sample.to(self.device)
                    index += 1

                return sample_return