from typing import Sequence, Tuple

import os
import wandb

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.metrics import DiceMetric, DiceHelper
from monai.transforms import AsDiscrete

from dataset.base_dataset import BaseDataset
from models.model_type import ModelType
from losses.loss import Loss
from utils import model_hub, get_model_type, get_class_names

class Engine:
    def __init__(
        self,
        model_name="diff_unet", 
        data_name="amos",
        data_path=None,
        batch_size=1,
        sw_batch_size=4,
        image_size=256,
        spatial_size=96,
        timesteps=1000,
        classes=None,
        device="cpu",
        num_workers=2,
        losses="mse,cse,dice",
        loss_combine='sum',
        model_path=None,
        project_name=None,
        wandb_name=None,
        include_background=False,
        use_amp=True,
        use_cache=True,
        use_wandb=True,
        mode="train",
    ):
        self.model_name = model_name
        self.model_type = get_model_type(model_name)
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.sw_batch_size = sw_batch_size
        self.image_size = image_size
        self.spatial_size = spatial_size
        self.timesteps = timesteps
        self.class_names = get_class_names(classes, include_background)
        self.num_classes = len(self.class_names)
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.losses = losses
        self.loss_combine = loss_combine
        self.model_path = model_path
        self.project_name = project_name
        self.wandb_name = wandb_name
        self.include_background = include_background
        self.use_amp = use_amp
        self.use_cache = use_cache
        self.use_wandb = use_wandb
        self.one_hot = True # self.model_type == ModelType.Diffusion
        self.mode = mode
        self.global_step = 0
        self.best_mean_dice = 0
        self.loss = 0
        
        print(f"number of classes : {self.num_classes}")
        
        if isinstance(image_size, tuple):
            width, height = image_size
        elif isinstance(image_size, int):
            width = height = image_size
        
        if use_wandb and mode == "test":
            self.table = None # reserved
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.tensor2pil = transforms.ToPILImage()
        self.criterion = Loss(self.losses, 
                              self.num_classes, 
                              self.loss_combine, 
                              self.one_hot,
                              self.include_background)
        self.dice_metric = DiceHelper(include_background=True, # self.include_background, 
                                      get_not_nans=False,
                                      num_classes=self.num_classes,
                                      reduction="mean",
                                      ignore_empty=False) # False
        self.inferer = SlidingWindowInferer(roi_size=[spatial_size, width, height],
                                            sw_batch_size=4,
                                            overlap=0.6)
        
    def load_checkpoint(self, model_path: str):
        pass # to be implemented
    
    def load_model(self):
        return model_hub(
            model_name=self.model_name, 
            spatial_size=self.spatial_size,
            timesteps=self.timesteps,
            num_classes=self.num_classes,
            mode=self.mode).to(self.device)
    
    def save_model(
        self,
        model, 
        optimizer=None, 
        scheduler=None, 
        epoch=None, 
        save_path=None, 
    ):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch+1,
            'loss': self.loss,
            'global_step': self.global_step,
            'best_mean_dice': self.best_mean_dice,
            'project_name': self.project_name,
            'id': wandb.run.id if self.use_wandb else 0,
        }
        
        torch.save(state, save_path)

        print(f"model is saved in {save_path}")
    
    def get_dataloader(self, dataset: BaseDataset, batch_size: int = 1, shuffle: bool = False):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers)
        
    def set_dataloader(self):
        pass
    
    def set_losses(self):
        pass
        
    def get_input(self, batch: dict):
        image = batch["image"].to(self.device)
        if self.mode == "train":
            label = batch["label"].to(self.device)
        elif self.mode == "test":
            label = batch["raw_label"].to(self.device)
            
        label = self.convert_labels(label).float()
        
        return image, label

    def convert_labels(self, labels: torch.Tensor):
        if self.one_hot:
            new_labels = [labels == i for i in sorted(self.class_names.keys())]
            return torch.cat(new_labels, dim=1) 
        else:
            return labels
    
    def infer(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.get_input(batch)    
        imgsz = (self.spatial_size, self.image_size, self.image_size)
        
        if self.model_type == ModelType.Diffusion:
            if isinstance(self.model, nn.DataParallel):
                output = sliding_window_inference(image, imgsz, self.sw_batch_size, self.model.module, pred_type="ddim_sample")
            else:
                output = sliding_window_inference(image, imgsz, self.sw_batch_size, self.model, pred_type="ddim_sample")
        else:
            output = sliding_window_inference(image, imgsz, self.sw_batch_size, self.model)
            
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        
        return image, output, label
    
    def get_numpy_image(self, t: torch.Tensor, shape: tuple, is_label: bool = False):
        _, _, d, w, h = shape
        index = int(d * 0.75)
        if is_label: t = torch.argmax(t, dim=1)
        else: t = t.squeeze(0) * 255
        t = t[:, index, ...].to(torch.uint8)
        t = t.cpu().numpy()
        t = np.transpose(t, (1, 2, 0))
        if is_label: 
            t = t[:, :, 0]
            # t = cv2.resize(t, (w, h))
  
        return t
    
    def tensor2images(self, 
                      image: torch.Tensor, 
                      label: torch.Tensor, 
                      output: torch.Tensor, 
                      shape: tuple):
        return {
            "image" : self.get_numpy_image(image, shape),
            "label" : self.get_numpy_image(label, shape, is_label=True),
            "output" : self.get_numpy_image(output, shape, is_label=True),
        }
    
    def log(self, k, v, step=None):
        if self.use_wandb:
            wandb.log({k: v}, step=step if step is not None else self.global_step)
            
    def log_per_class(self, dice, hd95, step):
        if self.use_wandb:
            pass
            # wandb.log()
            
    def log_plot(self, 
                 vis_data: dict, 
                 mean_dice: float, 
                 mean_hd95: float, 
                 mean_iou: float, 
                 dices: Sequence[float], 
                 filename: str):
        patient = os.path.basename(filename).split(".")[0]
        
        plot = wandb.Image(
            vis_data["image"],
            masks={
                "prediction" : {
                    "mask_data" : vis_data["output"],
                    "class_labels" : self.class_names 
                },
                "label" : {
                    "mask_data" : vis_data["label"],
                    "class_labels" : self.class_names 
                }
            },
        )
        
        self.table.add_data(*([patient, plot, mean_dice, mean_hd95, mean_iou]+[d for d in dices.values()]))
        
        # wandb.log({"table": self.table})
        # self.table = wandb.Table(columns=["patient", "image", "dice"]+[n for n in self.class_names.values()])