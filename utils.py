import os
import glob 
import yaml
import argparse
from prettytable import PrettyTable
from collections import OrderedDict

import torch
import torch.nn as nn

from monai import transforms

from dataloader.amosloader import AMOSDataset
from models.diff_unet import DiffUNet
from models.smooth_diff_unet import SmoothDiffUNet


def load_model(model_name, **kwargs):
    if model_name == "diff_unet":
        model = DiffUNet(**kwargs)
    elif model_name == "smooth_diff_unet":
        model = SmoothDiffUNet(**kwargs)
    else:
        raise ValueError(f"Invalid model_type: {model_name}")

    return model

def save_model(model, optimizer, scheduler, epoch, global_step, best_mean_dice, project_name, id, save_path, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    # if delete_last_model is not None:
    #     delete_last_model(save_dir, delete_symbol)
    
    if isinstance(model, nn.DataParallel):
        model = model.module
        
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch+1,
        'global_step': global_step,
        'best_mean_dice': best_mean_dice,
        'project_name': project_name,
        'id': id,
    }
    
    torch.save(state, save_path)

    print(f"model is saved in {save_path}")

def get_class_names(dataset="amos"):
    if dataset == "amos":
        return OrderedDict({0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney", 
                            4: "gall bladder", 5: "esophagus", 6: "liver", 7: "stomach", 
                            8: "arota", 9: "postcava", 10: "pancreas", 11: "right adrenal gland", 
                            12: "left adrenal gland", 13: "duodenum", 14: "bladder", 15: "prostate,uterus"})

def get_amosloader(data_dir, image_size=256, spatial_size=96, num_samples=1, mode="train", use_cache=True):
    data = {
        "train" : 
            {"images" : sorted(glob.glob(f"{data_dir}/imagesTr/*.nii.gz")),
             "labels" : sorted(glob.glob(f"{data_dir}/labelsTr/*.nii.gz"))},
        "val" : 
            {"images" : sorted(glob.glob(f"{data_dir}/imagesVa/*.nii.gz")),
             "labels" : sorted(glob.glob(f"{data_dir}/labelsVa/*.nii.gz"))},
    }
    
    paired = []

    for image_path, label_path in zip(data["train"]["images"], data["train"]["labels"]):
        pair = [image_path, label_path]
        paired.append(pair)
    
    data["train"]["files"] = paired
    
    paired = []

    for image_path, label_path in zip(data["val"]["images"], data["val"]["labels"]):
        pair = [image_path, label_path]
        paired.append(pair)
        
    data["val"]["files"] = paired
    train_transform = transforms.Compose(
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
            # transforms.Resized(
            #     keys=["image", "label"],
            #     spatial_size=(spatial_size, image_size, image_size),
            # ),
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
            
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),

            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    val_transform = transforms.Compose(
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

    test_transform = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),            
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )

    if mode == "train":
        train_dataset = AMOSDataset(data["train"]["files"], 
                                transform=train_transform, 
                                data_dir=data_dir, 
                                data_dict=data,
                                mode="train",
                                use_cache=use_cache)
    
        val_dataset = AMOSDataset(data["val"]["files"], 
                            transform=val_transform, 
                            data_dir=data_dir, 
                            data_dict=data, 
                            mode="val",
                            use_cache=use_cache)

        loader = [train_dataset, val_dataset]
        
    elif mode == "test":
        test_dataset = AMOSDataset(data["val"]["files"], 
                            transform=test_transform, 
                            data_dir=data_dir, 
                            data_dict=data,
                            mode="test",
                            use_cache=use_cache)
        
        return test_dataset
    else:
        train_dataset = AMOSDataset(data["train"]["files"], 
                                transform=train_transform, 
                                data_dir=data_dir, 
                                data_dict=data,
                                mode="train",
                                use_cache=use_cache)
    
        val_dataset = AMOSDataset(data["val"]["files"], 
                            transform=val_transform, 
                            data_dir=data_dir, 
                            data_dict=data, 
                            mode="val",
                            use_cache=use_cache)
        
        test_dataset = AMOSDataset(data["val"]["files"], 
                            transform=test_transform, 
                            data_dir=data_dir, 
                            mode="test",
                            data_dict=data,
                            use_cache=use_cache)
        
        loader = [train_dataset, val_dataset, test_dataset]
    
    return loader

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
    