import os
import wandb
import torch
import numpy as np

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

from torchvision import transforms
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from utils import load_model

from scipy.ndimage import distance_transform_edt as eucl_distance

class Engine:
    def __init__(
        self,
        model_name="diff_unet", 
        image_size=256,
        depth=96,
        class_names=None,
        num_classes=16, 
        device="cpu",
        num_workers=2,
        model_path=None,
        wandb_name=None,
        pretrained=True,
        use_amp=True,
        use_wandb=True,
        mode="train",
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.depth = depth
        self.class_names = class_names
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.model_path = model_path
        self.wandb_name = wandb_name
        self.pretrained = pretrained
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        self.mode = mode
        self.global_step = 0
        self.best_mean_dice = 0
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
        
        if class_names is not None:
            self.class_names = class_names
            self.num_classes = len(class_names)
        else:
            self.num_classes = num_classes
            
        if use_wandb and mode == "test":
            self.table = wandb.Table(columns=["patient", "image", "dice"]+[n for n in self.class_names.values()])
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.window_infer = SlidingWindowInferer(roi_size=[depth, self.width, self.height],
                                                 sw_batch_size=1,
                                                 overlap=0.6)
        self.tensor2pil = transforms.ToPILImage()
        
    def load_checkpoint(self, model_path):
        pass
    
    def load_model(self):
        return load_model(model_name=self.model_name, 
                          image_size=self.image_size,
                          depth=self.depth,
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
        label = batch["label"]
        
        label = self.convert_labels(label)
        label = label.float()
        
        return image, label

    def convert_labels(self, labels):
        labels_new = []
        for i in range(self.num_classes):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def get_numpy_image(self, t, index, is_label=False):
        if is_label: t = torch.argmax(t, dim=1)
        else: t = t.squeeze(0) * 255
        t = t[:, index, ...].to(torch.uint8)
        t = t.cpu().numpy()
        t = np.transpose(t, (1, 2, 0))
        if is_label: t = t[:, :, 0]
  
        return t
    
    def tensor2images(self, image, label, output, index=0):
        return {
            "image" : self.get_numpy_image(image, index),
            "label" : self.get_numpy_image(label, index, is_label=True),
            "output" : self.get_numpy_image(output, index, is_label=True),
        }
    
    # def one_hot2dist(self, seg: np.ndarray, resolution: Tuple[float, float, float] = None, dtype=None) -> np.ndarray:
    #     assert one_hot(torch.tensor(seg), axis=0)
    #     K: int = len(seg)

    #     res = np.zeros_like(seg, dtype=dtype)
    #     for k in range(K):
    #         posmask = seg[k].astype(np.bool)

    #         if posmask.any():
    #             negmask = ~posmask
    #             res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
    #                 - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
    #         # The idea is to leave blank the negative classes
    #         # since this is one-hot encoded, another class will supervise that pixel

    #     return res
        
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