import os
import wandb

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
from torchvision import transforms
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from utils import load_model


class Engine:
    def __init__(
        self,
        model_name="diff_unet", 
        image_size=256,
        spatial_size=96,
        class_names=None,
        num_classes=16, 
        device="cpu",
        num_workers=2,
        model_path=None,
        project_name=None,
        wandb_name=None,
        pretrained=True,
        use_amp=True,
        use_wandb=True,
        mode="train",
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.spatial_size = spatial_size
        self.class_names = class_names
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.model_path = model_path
        self.project_name = project_name
        self.wandb_name = wandb_name
        self.pretrained = pretrained
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        self.mode = mode
        self.global_step = 0
        self.best_mean_dice = 0
        
        if isinstance(image_size, tuple):
            width = image_size[0]
            height = image_size[1]
        elif isinstance(image_size, int):
            width = height = image_size
        
        if class_names is not None:
            self.class_names = class_names
            self.num_classes = len(class_names)
        else:
            self.num_classes = num_classes
            
        if use_wandb and mode == "test":
            self.table = None # reserved
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.window_infer = SlidingWindowInferer(roi_size=[spatial_size, spatial_size, spatial_size],
                                                 sw_batch_size=1,
                                                 overlap=0.6)
        self.tensor2pil = transforms.ToPILImage()
        
    def load_checkpoint(self, model_path):
        pass
    
    def load_model(self):
        return load_model(model_name=self.model_name, 
                          image_size=self.image_size,
                          spatial_size=self.spatial_size,
                          num_classes=self.num_classes,
                          device=self.device,
                          pretrained=self.pretrained,
                          mode=self.mode).to(self.device)
    
    def get_dataloader(self, dataset, batch_size=1, shuffle=False):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers)
        
    def get_input(self, batch):
        image = batch["image"]
        if self.mode == "train":
            label = batch["label"]
        elif self.mode == "test":
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

    def get_numpy_image(self, t, shape, is_label=False):
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
    
    def tensor2images(self, image, label, output, shape):
        return {
            "image" : self.get_numpy_image(image, shape),
            "label" : self.get_numpy_image(label, shape, is_label=True),
            "output" : self.get_numpy_image(output, shape, is_label=True),
        }
    
    def log(self, k, v, step=None):
        if self.use_wandb:
            wandb.log({k: v}, step=step if step is not None else self.global_step)
    
    def log_plot(self, vis_data, mean_dice, dices, filename):
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
        
        self.table.add_data(*([patient, plot, mean_dice]+[d for d in dices.values()]))
        
        # wandb.log({"table": self.table})
        # self.table = wandb.Table(columns=["patient", "image", "dice"]+[n for n in self.class_names.values()])