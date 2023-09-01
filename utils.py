import glob 
import argparse

from monai import transforms

from dataset.amosloader import AMOSDataset

def parse_args():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument("--log_dir", type=str,
                        help="Directory to store log files")
    parser.add_argument("--model_name", type=str, default="smooth_diff_unet",
                        help="Name of the model type")
    parser.add_argument("--max_epoch", type=int, default=50000,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of parallel workers for dataloader")
    parser.add_argument("--loss_combine", type=str, default='plus',
                        help="Method for combining multiple losses")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--val_freq", type=int, default=400,
                        help="Validation frequency (number of iterations)")
    parser.add_argument("--num_gpus", type=int, default=5,
                        help="Number of GPUs to use for training")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to the checkpoint for resuming training")
    parser.add_argument("--pretrained", action="store_true", default=False, # default=False,
                        help="Use pretrained weights")
    parser.add_argument("--use_wandb", action="store_true", default=True, # default=False,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--use_cache", action="store_true", default=True, # default=False,
                        help="Enable caching")

    args = parser.parse_args()
    return args


def get_amosloader(data_dir, spatial_size=96, num_samples=1, mode="train", use_cache=True):
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
                spatial_size=(spatial_size, spatial_size, spatial_size),
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