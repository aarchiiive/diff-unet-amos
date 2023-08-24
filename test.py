import glob
import natsort
from tqdm import tqdm
import numpy as np

# from medpy import metric

import torch 
import torch.nn as nn 

from monai.data import DataLoader
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer

from light_training.trainer import Trainer

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder

from utils import get_amosloader

from metric import *

from dataset_path import data_dir

set_determinism(123)


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.01] = 0.01
    uncer_out = - pred_out * torch.log(pred_out)

    return uncer_out


class DiffUNet(nn.Module):
    def __init__(self, 
                 image_size,
                 depth,
                 num_classes
                 ):
        super().__init__()
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
            
        self.depth = depth
        self.num_classes = num_classes
        
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, num_classes+1, num_classes, [64, 64, 128, 256, 512, 64], 
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

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.num_classes, self.depth, self.width, self.height), model_kwargs={"image": image, "embeddings": embeddings})

            sample_return = torch.zeros((1, self.num_classes, self.depth, self.width, self.height))
            all_samples = sample_out["all_samples"]
            index = 0
            for sample in all_samples:
                sample_return += sample.cpu()
                index += 1

            return sample_return
        
class AMOSTester:
    def __init__(self, 
                 model_path,
                 image_size=256,
                 depth=96,
                 num_classes=16,
                 device="cpu", 
                 use_wandb=True
                ):
        
        self.model_path = model_path
        self.device = torch.device(device)
        self.num_classes = num_classes
         
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
            
        self.window_infer = SlidingWindowInferer(roi_size=[depth, image_size, image_size],
                                                 sw_batch_size=1,
                                                 overlap=0.6)
        
        self.window_infer.device = self.device
        self.window_infer.sw_device = self.device
        
        self.model = DiffUNet(image_size=image_size,
                              depth=depth,
                              num_classes=num_classes,
                              ).to(self.device)
        
        self.best_mean_dice = 0.0
        
        self.load_checkpoint(model_path)

    def load_checkpoint(self, model_path):
        print(f"Checkpoint loaded from {model_path}.....")
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        
    def get_input(self, batch):
        image = batch["image"]
        label = batch["raw_label"]
        
        label = self.convert_labels(label)

        label = label.float()
        return image, label

    def convert_labels(self, labels):
        labels_new = []
        for i in range(self.num_classes):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu()

        d, w, h = label.shape[2], label.shape[3], label.shape[4]

        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h))
        output = output.numpy()
        target = label.cpu().numpy()

        dices = []
        # hd = []
        for i in range(0, self.num_classes):
            pred = output[:, i]
            gt = target[:, i]

            if pred.sum() > 0 and gt.sum()>0:
                dice = dice_coef(pred, gt)
                # hd95 = hd95(pred, gt)
                # dice = metric.binary.dc(pred, gt)
                # hd95 = metric.binary.hd95(pred, gt)
            elif pred.sum() > 0 and gt.sum()==0:
                dice = 1
                # hd95 = 0
            else:
                dice = 0
                # hd95 = 0

            dices.append(dice)
            # hd.append(hd95)
        
        all_m = []
        for d in dices:
            all_m.append(d)
        # for h in hd:
        #     all_m.append(h)
        
        print(f"mean dice : {sum(dices) / self.num_classes:.4f}", )
        
        return all_m 
    
    def test(self, val_dataset,):
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        val_outputs = []
        
        self.model.eval()
        
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch = {
                x: batch[x].to(self.device)
                for x in batch if isinstance(batch[x], torch.Tensor)
            }

            with torch.no_grad():
                val_out = self.validation_step(batch)
                assert val_out is not None 

            return_list = False
            val_outputs.append(val_out)
            
        if isinstance(val_out, list) or isinstance(val_out, tuple):
            return_list = True

        val_outputs = torch.tensor(val_outputs)
        if not return_list:
            length = 0
            v_sum = 0.0
            for v in val_outputs:
                if not torch.isnan(v):
                    v_sum += v
                    length += 1

            if length == 0:
                v_sum = 0
            else :
                v_sum = v_sum / length             
        else :
            num_val = len(val_outputs[0])
            length = [0.0 for i in range(num_val)]
            v_sum = [0.0 for i in range(num_val)]

            for v in val_outputs:
                for i in range(num_val):
                    if not torch.isnan(v[i]):
                        v_sum[i] += v[i]
                        length[i] += 1

            for i in range(num_val):
                if length[i] == 0:
                    v_sum[i] = 0
                else :
                    v_sum[i] = v_sum[i] / length[i]
                    
        return v_sum, val_outputs

if __name__ == "__main__":
    model_path = natsort.natsorted(glob.glob("logs/amos/model/*.pt"))[-1]
    test_ds = get_amosloader(data_dir=data_dir, mode="test")

    tester = AMOSTester(model_path=model_path,
                         device="cpu",
                         use_wandb=False)

    v_mean, v_out = tester.test(test_ds)

    print(f"v_mean is {v_mean}")