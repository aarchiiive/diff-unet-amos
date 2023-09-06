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


from typing import Optional

import os
import pickle
import numpy as np
from tqdm import tqdm 

import nibabel
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset 
from monai import transforms


class AMOSDataset(Dataset):
    def __init__(self, 
                 data_list: list, 
                 image_size: int = 256,
                 spatial_size: int = 96,
                 pad: int = 2,
                 padding: bool = True,
                 transform: transforms = None, 
                 data_path: Optional[str] = None, 
                 data_dict: Optional[dict] = None, 
                 mode: Optional[str] = "train",
                 use_cache: Optional[bool] = True) -> None:
        super().__init__()
        
        self.transform = transform
        self.data_list = data_list
        self.image_size = image_size
        self.spatial_size = spatial_size
        self.padding = padding
        self.data_path = data_path
        self.data_dict = data_dict
        self.mode = mode
        self.use_cache = use_cache
        
        self.pad = (pad, pad)
        
        assert mode != "train" or  mode != "val" or  mode != "test", \
            "Key must be one of these keywords : train / val / test"
            
        self.key = "Tr" if mode == "train" else "Va"
        self.resize = transforms.Compose([transforms.Resize((spatial_size, image_size, image_size))])
        self.cache = {}
        
        if use_cache:
            print("Caching....")
            self.save_cache()
    
    def save_cache(self):
        for d in tqdm(self.data_list):
            _ = self.read_data(d)
            
    def read_data(self, data_path):
        if data_path[0] in self.cache.keys():
            return self.cache[data_path[0]]
        else:
            image_path = data_path[0]
            label_path = data_path[1]

            image = nibabel.load(image_path).get_fdata()
            label = nibabel.load(label_path).get_fdata()
            raw_label = nibabel.load(label_path).get_fdata()

            image = torch.tensor(image)
            label = torch.tensor(label)
            raw_label = torch.tensor(raw_label)
            
            # if self.padding:
            #     _, _, d = image.shape
                
            #     if self.spatial_size > d: # add padding
            #         p = (self.spatial_size - d) // 2
            #         pad = (p, p) if d % 2 == 0 else (p, p+1)
            #         image = F.pad(image, pad, "constant")
            #         label = F.pad(label, pad, "constant")
            #     elif self.spatial_size < d: # resize -> reducing depth
            #         image = F.interpolate(image, size=(self.spatial_size), mode='nearest')
            #         label = F.interpolate(label, size=(self.spatial_size), mode='nearest')
            
            image = F.pad(image, self.pad, "constant", 0)
            label = F.pad(label, self.pad, "constant", 0)
            
            # (H, W, D) -> (D, W, H)
            image = torch.transpose(image, 0, 2).contiguous()
            label = torch.transpose(label, 0, 2).contiguous()
            raw_label = torch.transpose(raw_label, 0, 2).contiguous()
            
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            raw_label = raw_label.unsqueeze(0)
            
            if self.resize:
                image = self.resize(image)
                label = self.resize(label)
            
            self.cache[data_path[0]] = {
                "image": image,
                "label": label
            } 
            
            if self.mode == "test": self.cache[data_path[0]]["raw_label"] = raw_label
            
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
        data = self.read_data(self.data_list[i])
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data, self.data_list[i][0]
            


