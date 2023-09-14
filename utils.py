from typing import Dict, Sequence, Union, Tuple

import os
import glob 
import yaml
import argparse
from prettytable import PrettyTable
from collections import OrderedDict

from monai import transforms

from dataloader.amos_dataset import AMOSDataset
from dataloader.msd_dataset import MSDDataset
from models.model_hub import ModelHub
from models.model_type import ModelType


def model_hub(model_name: str, **kwargs):
    return ModelHub().__call__(model_name, **kwargs)

def get_model_type(model_name: str):
    assert model_name in ["diff_unet", "smooth_diff_unet", "swin_unetr"]
    if model_name in ["diff_unet", "smooth_diff_unet"]:
        return ModelType.Diffusion
    elif model_name in ["swin_unetr"]:
        return ModelType.SwinUNETR

def get_data_path(name: str = "amos"):
    if name == "amos":
        return "/home/song99/ws/datasets/AMOS"
    elif name == "msd":
        return "/home/song99/ws/datasets/MSD"

def get_class_names(classes: Dict[int, str], remove_bg: bool =False, bg_index: int = 0):
     with open(classes, "r") as f:
        classes = OrderedDict(yaml.safe_load(f))
        if remove_bg: del classes[0]
        return classes

def get_dataloader(
    data_path: str, 
    data_name: str, 
    image_size: int = 256, 
    spatial_size: int = 96, 
    num_samples: int = 1, 
    mode: str = "train", 
    one_hot: bool = True,
    remove_bg: bool = False,
    use_cache: bool = True,
):
    transform = {}
    transform["train"] = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            transforms.RandScaleCropd(
                keys=["image", "label"], 
                roi_scale=[0.75, 0.85, 0.95],
                random_size=False
            ),
            transforms.Resized(
                keys=["image", "label"],
                spatial_size=(spatial_size, image_size, image_size),
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
            
            # transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),

            # transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    transform["val"] = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.Resized(
                keys=["image", "label"],
                spatial_size=(spatial_size, image_size, image_size),
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    transform["test"] = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),            
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    
    data = {
        "train" : 
            {"images" : sorted(glob.glob(f"{data_path}/imagesTr/*.nii.gz")),
             "labels" : sorted(glob.glob(f"{data_path}/labelsTr/*.nii.gz"))},
        "val" : 
            {"images" : sorted(glob.glob(f"{data_path}/imagesVa/*.nii.gz")),
             "labels" : sorted(glob.glob(f"{data_path}/labelsVa/*.nii.gz"))},
        "test" : 
            {"images" : sorted(glob.glob(f"{data_path}/imagesVa/*.nii.gz")),
             "labels" : sorted(glob.glob(f"{data_path}/labelsVa/*.nii.gz"))},
    }
    
    for p in ["train", "val", "test"]:
        paired = []
        for image_path, label_path in zip(data[p]["images"], data[p]["labels"]):
            pair = [image_path, label_path]
            paired.append(pair)
            
        data[p]["files"] = paired
    
    if data_name == "amos":
        Dataset = AMOSDataset
    elif data_name == "msd":
        Dataset = MSDDataset
    
    dataset = {}
    
    for p in ["train", "val", "test"]:
        dataset[p] = Dataset(
            data_list=data[p]["files"], 
            image_size=image_size,
            spatial_size=spatial_size,
            transform=transform[p], 
            data_path=data_path, 
            mode=p,
            one_hot=one_hot,
            remove_bg=remove_bg,
            use_cache=use_cache and (p not in ["val", "test"]),
        )

    if mode == "train":
        return [dataset["train"], dataset["val"]]
    elif mode == "test":
        return dataset["test"]
    else:
        return [dataset["train"], dataset["val"], dataset["test"]]

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
    