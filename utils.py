from typing import Dict, Sequence, Union, Tuple

import os
import glob 
import yaml
import argparse
from prettytable import PrettyTable
from collections import OrderedDict

from monai import transforms
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

from dataset.amos_dataset import AMOSDataset
from dataset.msd_dataset import MSDDataset
from dataset.btcv_dataset import BTCVDataset
from models.model_hub import ModelHub
from models.model_type import ModelType


def model_hub(model_name: str, **kwargs):
    return ModelHub().__call__(model_name, **kwargs)

def get_model_type(model_name: str):
    assert model_name in ["diff_unet", "smooth_diff_unet", "swin_unetr", "attention_unet"]
    if model_name in ["diff_unet", "smooth_diff_unet"]:
        return ModelType.Diffusion
    elif model_name in ["swin_unetr"]:
        return ModelType.SwinUNETR
    elif model_name in ["attention_unet"]:
        return ModelType.AttentionUNet

# def get_data_path(name: str = "amos"):
#     if name == "amos":
#         return "/home/song99/ws/datasets/AMOS"
#     elif name == "msd":
#         return "/home/song99/ws/datasets/MSD"
#     elif name == "btcv":
#         return "/home/song99/ws/datasets/BTCV"

def get_class_names(classes: Dict[int, str], include_background: bool = False, bg_index: int = 0):
     with open(classes, "r") as f:
        classes = OrderedDict(yaml.safe_load(f))
        if not include_background: del classes[0]
        return classes

def get_dataloader(
    data_path: str, 
    image_size: int = 256, 
    spatial_size: int = 96, 
    num_samples: int = 2, 
    num_workers: int = 8,
    batch_size: int = 1,
    cache_rate: float = 1.0,
    mode: str = "train", 
):
    transform = {}
    transform["train"] = transforms.Compose(
        [   
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.RandScaleCropd(
                keys=["image", "label"], 
                roi_scale=[0.75, 0.85, 1.0],
                random_size=False
            ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(spatial_size, image_size, image_size),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),

            # transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    transform["val"] = transforms.Compose(
        [   
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            # transforms.Resized(keys=["image", "label"], spatial_size=(spatial_size, image_size, image_size)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    transform["test"] = transforms.Compose(
        [   
            transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),            
            transforms.ToTensord(keys=["image"]),
        ]
    )
    
    def parse_type(p):
        if p == "train":
            return "training"
        elif p == "val":
            return "validation"
        else:
            return p
    
    dataloader = {}
    for p in ["train", "val", "test"]:
        if mode == "train" and p == "test": continue
        data = load_decathlon_datalist(os.path.join(data_path, "dataset.json"), True, parse_type(p))
        dataset = CacheDataset(
            data=data,
            transform=transform[p],
            cache_num=len(data),
            cache_rate=cache_rate,
            num_workers=max(num_workers, 20),
        )
        dataloader[p] = ThreadDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size if p == "train" else 1, 
            shuffle=True if p == "train" else False
        )
        

    return dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)
        
    table = PrettyTable(["Argument", "Value"])
    for arg, value in args.__dict__.items():
        table.add_row([arg, value])
    
    print(table)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    