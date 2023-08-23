# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import (
    List,
    Dict,
    Tuple,
    Optional, 
    Sequence, 
    Union
)

import os
import glob 
import time
import pickle
import numpy as np
from tqdm import tqdm 
from collections import OrderedDict

import nibabel
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset 

from monai import transforms


def resample_img(
    image: sitk.Image,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    is_label: bool = False,
    pad_value = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


class AMOSDataset(Dataset):
    def __init__(self, 
                 data_list : list, 
                 image_size : int = 256,
                 depth : int = 96,
                 padding : bool = True,
                 transform : transforms = None, 
                 data_dir : Optional[str] = None, 
                 data_dict : Optional[dict] = None, 
                 key : Optional[str] = "train") -> None:
        super().__init__()
        
        self.transform = transform
        self.data_list = data_list
        self.image_size = image_size
        self.depth = depth
        self.padding = padding
        self.data_dir = data_dir
        self.data_dict = data_dict
        self.tensor_dir = os.path.join(data_dir, "tensor")
        self.cache_dir = os.path.join(data_dir, "cache")
        self.cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        assert key != "train" or  key != "val" or  key != "test", \
            "key must be one of these keywords : train / val / test"
            
        self.key = "Tr" if key == "train" else "Va"
        # self.cache = cache
        
        self.resize = transforms.Compose([transforms.Resize((self.image_size, self.image_size))])

        os.makedirs(self.tensor_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache = {}
        
        print("Caching....")
        self.save_cache(key)
        
    
    def load_cache(self, key):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            return False
        else:
            self.cache = {}
            return True
 
    def save_cache(self, key):
        with open(self.cache_path, 'wb') as f:
            for d in tqdm(self.data_list):
                _ = self.read_data(d)
            pickle.dump(self.cache, f)
            
    def read_data(self, data_path):
        if data_path[0] in self.cache.keys():
            return self.cache[data_path[0]]
        else:
            image_path = data_path[0]
            label_path = data_path[1]

            # image = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path))).astype(np.float32)
            # label = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(label_path), is_label=True)).astype(np.float32)
            # raw_label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)
            
            image = nibabel.load(image_path).get_fdata()
            label = nibabel.load(label_path).get_fdata()
            raw_label = nibabel.load(label_path).get_fdata()

            image = torch.tensor(image)
            label = torch.tensor(label)
            raw_label = torch.tensor(raw_label)
            
            # # (D, W, H) -> (H, W, D)
            # image = torch.transpose(image, 0, 2).contiguous()
            # label = torch.transpose(label, 0, 2).contiguous()
            # raw_label = torch.transpose(raw_label, 0, 2).contiguous()
            
            
            if self.padding:
                _, _, d = image.shape
                
                if self.depth > d: # add padding
                    p = (self.depth - d) // 2
                    pad = (p, p) if d % 2 == 0 else (p, p+1)
                    image = F.pad(image, pad, "constant")
                    label = F.pad(label, pad, "constant")
                elif self.depth < d: # resize -> reducing depth
                    image = F.interpolate(image, size=(self.depth), mode='nearest')
                    label = F.interpolate(label, size=(self.depth), mode='nearest')
                    
                _, _, d = raw_label.shape
                 
                if self.depth > d: # add padding
                    p = (self.depth - d) // 2
                    pad = (p, p) if d % 2 == 0 else (p, p+1)
                    raw_label = F.pad(raw_label, pad, "constant")
                elif self.depth < d: # resize -> reducing depth
                    raw_label = F.interpolate(raw_label, size=(self.depth), mode='nearest')

            # # (H, W, D) -> (D, W, H)
            image = torch.transpose(image, 0, 2).contiguous()
            label = torch.transpose(label, 0, 2).contiguous()
            raw_label = torch.transpose(raw_label, 0, 2).contiguous()
            
            if self.resize:
                image = self.resize(image)
                label = self.resize(label)
                raw_label = self.resize(raw_label)
                
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            raw_label = raw_label.unsqueeze(0)
            
            self.cache[data_path[0]] = {
                "image": image,
                "label": label,
                "raw_label": raw_label
            } 
            
            return self.cache[data_path[0]]
        
    def nii2tensor(self):
        assert self.data_dict != None, "data_dict has not been assigned"
        for phase in ["train", "val"]:
            p = "Tr" if phase == "train" else "Va"
            image_dir = os.path.join(self.tensor_dir, f"images{p}")
            label_dir = os.path.join(self.tensor_dir, f"labels{p}")
            
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            for img, label in zip(tqdm(self.data_dict[phase]["images"]), self.data_dict[phase]["labels"]):
                name = os.path.basename(img)
                patient = name.split(".")[0]
                
                img = nibabel.load(img).get_fdata()
                label = nibabel.load(label).get_fdata()
                
                img = torch.from_numpy(img).permute(2, 1, 0)
                label = torch.from_numpy(label).permute(2, 1, 0)
                
                torch.save(img, os.path.join(image_dir, f"{patient}.pt"))
                torch.save(label, os.path.join(label_dir, f"{patient}.pt"))
    
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        image = self.read_data(self.data_list[i])
        # try:
        #     image = self.read_data(self.data_list[i])
        # except:
        #     with open("./bugs.txt", "a+") as f:
        #         f.write(f"Bugs reported, {self.data_list[i]}\n")
        #     if i != len(self.data_list)-1:
        #         return self.__getitem__(i+1)
        #     else :
        #         return self.__getitem__(i-1)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image


def get_amosloader(data_dir, spatial_size=96, num_samples=1, mode="train"):
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
    
    train_dataset = AMOSDataset(data["train"]["files"], 
                                transform=train_transform, 
                                data_dir=data_dir, 
                                data_dict=data,
                                key="train")
    
    # train_dataset.nii2tensor()
    
    val_dataset = AMOSDataset(data["val"]["files"], 
                         transform=val_transform, 
                         data_dir=data_dir, 
                         data_dict=data, 
                         key="val")
    
    if mode != "train":
        test_dataset = AMOSDataset(data["val"]["files"], 
                            transform=test_transform, 
                            data_dir=data_dir, 
                            key="test",
                            data_dict=data)
    
        loader = [train_dataset, val_dataset, test_dataset]
    else:
        loader = [train_dataset, val_dataset]
    
    # t0 = time.time()
    # for data in train_dataset:
    #     # print(data[0].keys())
    #     print(data[0]['image'].shape)
    #     print(f"{time.time() - t0:.4f}s")
        
    #     t0 = time.time()

    return loader