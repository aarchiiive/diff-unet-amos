import glob 

from monai import transforms

from dataset.amosloader import AMOSDataset


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