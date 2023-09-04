import os
import glob 
import argparse
from prettytable import PrettyTable
from collections import OrderedDict

import torch
import torch.nn as nn

from monai import transforms

from dataloader.amosloader import AMOSDataset
from models.diff_unet import DiffUNet
from models.smooth_diff_unet import SmoothDiffUNet


def parse_args(mode="train", project_name="diff-unet"):
    parser = argparse.ArgumentParser()
    
    # Common settings
    parser.add_argument("--model_name", type=str, default="smooth_diff_unet",
                        help="Name of the model type")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="Image size")
    parser.add_argument("--spatial_size", type=int, default=96, 
                        help="Spatial size")
    parser.add_argument("--num_classes", type=int, default=16, 
                        help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--model_path", type=str, default="logs/smoothing-encoder/weights/epoch_474.pt",
                        help="Path to the checkpoint for pretrained weights")
    parser.add_argument("--pretrained", action="store_true", 
                        help="Use pretrained model")
    parser.add_argument("--project_name", type=str, default=project_name, 
                        help="Project name for WandB logging")
    parser.add_argument("--wandb_name", type=str, default="smoothing-encoder", 
                        help="Name for WandB logging")
    parser.add_argument("--use_wandb", action="store_true", default=True, # default=False,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--use_amp", action="store_true", default=True, # default=False,
                        help="Enable Automatic Mixed Precision (AMP)")
    
    if mode == "train":
        # Training settings
        parser.add_argument("--log_dir", type=str,
                            help="Directory to store log files")
        parser.add_argument("--max_epochs", type=int, default=2000,
                            help="Maximum number of training epochs")
        parser.add_argument("--batch_size", type=int, default=10,
                            help="Batch size for training")
        parser.add_argument("--num_workers", type=int, default=2,
                            help="Number of parallel workers for dataloader")
        parser.add_argument("--loss_combine", type=str, default='plus',
                            help="Method for combining multiple losses")
        parser.add_argument("--val_freq", type=int, default=20000,
                            help="Validation frequency (number of iterations)")
        parser.add_argument("--num_gpus", type=int, default=5,
                            help="Number of GPUs to use for training")
        parser.add_argument("--use_cache", action="store_true", default=False, # default=False,
                            help="Enable caching")
    
    args = parser.parse_args()
    
    table = PrettyTable(["Argument", "Value"])
    for arg, value in args.__dict__.items():
        table.add_row([arg, value])
    
    print(table)
    
    return args

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
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
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
            transforms.ToTensord(keys=["image", "label"],),
        ]
    )
    
    val_transform = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [   
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
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),            
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )

    # train_ds = PretrainDataset(data["train"]["files"], transform=train_transform, data_dir=data_dir, cache=cache)
    # val_ds = PretrainDataset(data["val"]["files"], transform=val_transform, data_dir=data_dir, cache=cache)
    # test_ds = PretrainDataset(data["val"]["files"], transform=test_transform, data_dir=data_dir)
    
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