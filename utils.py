from __future__ import annotations
from typing import Dict, Sequence, Union, Tuple

import os
import glob 
import json
import yaml
import argparse
from pathlib import Path
from prettytable import PrettyTable
from collections import OrderedDict

from monai.config import KeysCollection, PathLike
from monai import transforms
from monai.data.decathlon_datalist import _append_paths
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    # load_decathlon_datalist,
)

from dataset.cache_dataset import LabelSmoothingCacheDataset
from models.utils.model_hub import ModelHub
from models.utils.model_type import ModelType


def model_hub(model_name: str, **kwargs):
    return ModelHub().__call__(model_name, **kwargs)

def get_model_type(model_name: str):
    assert model_name in ["diff_unet", "smooth_diff_unet", "swin_unetr", "diff_swin_unetr", "attention_unet", "attention_diff_unet"]
    if "diff" in model_name:
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


def load_decathlon_datalist(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: PathLike | None = None,
) -> list[dict]:
    """Load image/label paths of decathlon challenge from JSON file

    Json file is similar to what you get from http://medicaldecathlon.com/
    Those dataset.json files

    Args:
        data_list_file_path: the path to the json file of datalist.
        is_segmentation: whether the datalist is for segmentation task, default is True.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, is_segmentation, expected_data)

def get_dataloader(
    data_path: str, 
    image_size: int = 256, 
    spatial_size: int = 96, 
    num_classes: int = 14,
    num_samples: int = 1, 
    num_workers: int = 8,
    batch_size: int = 1,
    cache_rate: float = 1.0,
    label_smoothing: bool = False,
    smoothing_alpha: float = 0.3,
    smoothing_order: float = 1.0,
    mode: str = "train", 
):
    transform = {}
    keys = ["image", "label"]+[f"distance_{i}" for i in range(num_classes)]
    # interpolations = ["nearest"]+["bilinear"]*(num_classes+1)
    interpolations = ["nearest"]*len(keys)
    
    transform["train"] = transforms.Compose(
        [   
            # transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True), # *** turn off if using label smoothing *** 
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
            # transforms.RandScaleCropd(
            #     keys=["image", "label"], 
            #     roi_scale=[0.75, 0.85, 1.0],
            #     random_size=False
            # ),
            # transforms.Resized(keys=["image", "label"], spatial_size=(spatial_size, image_size, image_size)),
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

            transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
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
    
    if mode == "train":
        phase = ["train", "val", "test"]
    elif mode == "test":
        phase = ["val"]

    dataloader = {}
    for p in phase:
        if mode == "train" and p == "test": continue
        elif mode == "test" and p == "train": continue
        data = load_decathlon_datalist(os.path.join(data_path, "dataset.json"), True, parse_type(p))
        
        if p == "train":
            if label_smoothing:
                dataset = LabelSmoothingCacheDataset(
                    data=data,
                    transform=transform[p],
                    cache_num=len(data),
                    cache_rate=cache_rate,
                    num_workers=max(num_workers, 20),
                    num_classes=num_classes,
                    smoothing_alpha=smoothing_alpha,
                    smoothing_order=smoothing_order,
                )
            else:
                dataset = CacheDataset(
                    data=data,
                    transform=transform[p],
                    cache_num=len(data),
                    cache_rate=cache_rate,
                    num_workers=max(num_workers, 20),
                )
        else:
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
    