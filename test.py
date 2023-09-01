import os
import glob
import wandb
import natsort
from tqdm import tqdm
from collections import OrderedDict

import numpy as np

# from medpy import metric

import torch 
import torch.nn as nn 
from torchvision import transforms

from monai.data import DataLoader
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer

from utils import get_amosloader
from unet.diff_unet import DiffUNet
from metric import *
from dataset_path import data_dir

set_determinism(123)

        
class AMOSTester:
    def __init__(self, 
                 model_path,
                 image_size=256,
                 depth=96,
                 class_names=None,
                 num_classes=16,
                 device="cpu", 
                 wandb_name=None,
                 use_wandb=True
                ):
        
        self.model_path = model_path
        self.device = torch.device(device)
        self.use_wandb = use_wandb
        
        if class_names is not None:
            self.class_names = class_names
            self.num_classes = len(class_names)
        else:
            self.num_classes = num_classes
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
        
        if self.use_wandb:
            wandb.init(project="diff-unet-test", name=wandb_name, config=self.__dict__)
            self.table = wandb.Table(columns=["patient", "image", "dice"]+[n for n in self.class_names.values()])
            
        self.tensor2pil = transforms.ToPILImage()
            
        self.window_infer = SlidingWindowInferer(roi_size=[depth, image_size, image_size],
                                                 sw_batch_size=1,
                                                 overlap=0.6,
                                                 device=device,
                                                 sw_device=device)
        
        self.model = DiffUNet(image_size=image_size,
                              depth=depth,
                              num_classes=num_classes,
                              device=device).to(self.device)
        
        self.best_mean_dice = 0.0
        
        self.load_checkpoint(model_path)

    def load_checkpoint(self, model_path):
        print(f"Checkpoint loaded from {model_path}.....")
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        
    def get_input(self, batch):
        image = batch["image"]
        label = batch["raw_label"]
        
        label = self.convert_labels(label)

        label = label.float()
        return image, label

    def convert_labels(self, labels):
        labels_new = []
        for i in range(self.num_classes):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def validation_step(self, batch, filename):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

        d, w, h = label.shape[2], label.shape[3], label.shape[4]

        if self.use_wandb:
            vis_data = self.tensor2images(image, label, output, int(d * 0.75)) # put appropriate index

        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h))
        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = OrderedDict({v : 0 for v in self.class_names.values()})
        # hd = []
        for i in range(self.num_classes-1):
            pred = output[:, i]
            gt = target[:, i]

            if pred.sum() > 0 and gt.sum() > 0:
                dice = dice_coef(pred, gt)
                # hd95 = hd95(pred, gt)
                # dice = metric.binary.dc(pred, gt)
                # hd95 = metric.binary.hd95(pred, gt)
            elif pred.sum() > 0 and gt.sum()==0:
                dice = 1
                # hd95 = 0
            else:
                dice = 0
                # hd95 = 0

            dices[self.class_names[i]] = dice
            # hd.append(hd95)
        
        all_m = []
        for d in dices:
            all_m.append(d)
        # for h in hd:
        #     all_m.append(h)
        
        mean_dice = sum(dices.values()) / self.num_classes
        # print(f"mean dice : {mean_dice:.4f}", )
        
        if self.use_wandb:
            self.log_plot(vis_data, mean_dice, dices, filename)
            self.log("mean_dice", mean_dice)
        
        return all_m 
    
    def test(self, val_dataset):
        try:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            val_outputs = []
            
            self.model.eval()
            
            for idx, (batch, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
                
                with torch.no_grad():
                    val_out = self.validation_step(batch, filename[0])
                    assert val_out is not None 

                return_list = False
                val_outputs.append(val_out)
                
            if isinstance(val_out, list) or isinstance(val_out, tuple):
                return_list = True

            val_outputs = torch.tensor(val_outputs)
            if not return_list:
                length = 0
                v_sum = 0.0
                for v in val_outputs:
                    if not torch.isnan(v):
                        v_sum += v
                        length += 1

                if length == 0:
                    v_sum = 0
                else :
                    v_sum = v_sum / length             
            else :
                num_val = len(val_outputs[0])
                length = [0.0 for i in range(num_val)]
                v_sum = [0.0 for i in range(num_val)]

                for v in val_outputs:
                    for i in range(num_val):
                        if not torch.isnan(v[i]):
                            v_sum[i] += v[i]
                            length[i] += 1

                for i in range(num_val):
                    if length[i] == 0:
                        v_sum[i] = 0
                    else :
                        v_sum[i] = v_sum[i] / length[i]
            
            if self.use_wandb: wandb.log({"table": self.table})    
    
            return v_sum, val_outputs
        except:
            if self.use_wandb: wandb.log({"table": self.table})    
    
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
    
    def log(self, k, v, step=None):
        wandb.log({k: v}, step=step)
        
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
        
if __name__ == "__main__":
    class_names = OrderedDict({0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney", 
                               4: "gall bladder", 5: "esophagus", 6: "liver", 7: "stomach", 
                               8: "arota", 9: "postcava", 10: "pancreas", 11: "right adrenal gland", 
                               12: "left adrenal gland", 13: "duodenum", 14: "bladder", 15: "prostate,uterus"})
    
    model_path = natsort.natsorted(glob.glob("logs/amos_ver2/weights/*.pt"))[-1]
    wandb_name = "amos_test"
    use_cache = False
    
    tester = AMOSTester(model_path=model_path,
                        device="cuda:0",
                        class_names=class_names,
                        wandb_name=wandb_name,
                        use_wandb=True)
    
    test_ds = get_amosloader(data_dir=data_dir, mode="test", use_cache=use_cache)
    v_mean, v_out = tester.test(test_ds)

    print(f"v_mean is {v_mean}")