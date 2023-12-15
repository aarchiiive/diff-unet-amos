import os
import wandb
import pickle
import warnings

from tqdm import tqdm
from prettytable import PrettyTable
from collections import OrderedDict

import numpy as np

import torch 
from monai.utils import set_determinism

from engine import Engine
from metric import dice_coeff
from utils import parse_args, get_dataloader

set_determinism(123)
warnings.filterwarnings("ignore")

class Tester(Engine):
    def __init__(
        self, 
        model_path: str = None,
        model_name: str = "diff_unet",
        data_name: str = "amos",
        data_path: str = None,
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        image_size: int = 256,
        spatial_size: int = 96,
        classes: str = None,
        epoch: int = 800,
        device: str = "cpu", 
        project_name: str = "diff-unet-test",
        wandb_name: str = None,
        include_background: bool = False,
        use_amp: bool = True,
        use_cache: bool = False,
        use_wandb: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            data_name=data_name,
            data_path=data_path, 
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            image_size=image_size,
            spatial_size=spatial_size,
            classes=classes, 
            device=device,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            include_background=include_background,
            use_amp=use_amp,
            use_cache=use_cache,
            use_wandb=use_wandb,
            mode="test",
        )
        self.epoch = epoch
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.model = self.load_model()
        self.log_dir = os.path.dirname(os.path.dirname(model_path))
        self.save_name = os.path.basename(model_path).split(".")[0]
        self.dices = []
        self.images = []
        self.outputs = []
        self.labels = []
        
        self.patient_index = 0
        
        self.load_checkpoint(model_path)
        self.set_dataloader()    
        
        if use_wandb:
            wandb.init(project=self.project_name,
                       name=self.wandb_name,
                       config=self.__dict__)
            self.table = wandb.Table(columns=["patient", "image", "dice", "hd95", "iou"]+
                                             [n for n in self.class_names.values()])

    def load_checkpoint(self, model_path):
        if self.epoch is not None:
            model_path = os.path.join(os.path.dirname(model_path), f"epoch_{self.epoch}.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict['model'])
            
        print(f"Checkpoint loaded from {model_path}.....")

    def set_dataloader(self):
        self.dataloader = get_dataloader(data_path=self.data_path,
                                         image_size=self.image_size,
                                         spatial_size=self.spatial_size,
                                         num_workers=self.num_workers,
                                         batch_size=self.batch_size,
                                         mode=self.mode)

    def test(self):
        self.model.eval()
        dices = []
        with torch.cuda.amp.autocast(self.use_amp):
            for batch in tqdm(self.dataloader["val"], total=len(self.dataloader["val"])):
                with torch.no_grad():
                    dices.append(self.validation_step(batch))
                
                self.global_step += 1
        
        if self.use_wandb: wandb.log({"table": self.table})  

        print("="*100)
        print(f"results : {np.mean(dices):.4f}")
        
        self.save_results()
    
    def validation_step(self, batch):
        with torch.cuda.amp.autocast(self.use_amp):
            images, outputs, labels = self.infer(batch)
            
        # _, _, d, w, h = labels.shape
        # images = torch.nn.functional.interpolate(images, mode="nearest", size=(d, w, h))
        # output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h))
        
        if self.use_wandb:
            vis_data = self.tensor2images(images, labels, outputs, images.shape) # put appropriate index
            
        dices = OrderedDict({v : 0 for v in self.class_names.values()})
        hds = OrderedDict({v : 0 for v in self.class_names.values()})
        ious = OrderedDict({v : 0 for v in self.class_names.values()})
        
        classes = list(self.class_names.values())
        
        # for i in range(self.num_classes):
        #     output = outputs[:, i]
        #     label = labels[:, i]
        #     if output.sum() > 0 and label.sum() == 0:
        #         dice = 1
        #     else:
        #         self.dice_metric(output, label)
        #         dice = self.dice_metric.aggregate().item()
            
        #     dices[classes[i]] = dice
        #     print(f"{classes[i]} : {dice:.4f}")

        # self.dice_metric.reset()
        
        # medpy ver.
        for i in range(self.num_classes):
            output = outputs[:, i]
            label = labels[:, i]
            if output.sum() > 0 and label.sum() == 0:
                dice = 1
            else:
                dice = dice_coeff(output, label).to(outputs.device).item()
            
            dices[classes[i]] = dice
            print(f"{classes[i]} : {dice:.4f}")
            
        self.dices.append(dices)
        self.images.append(images)
        self.outputs.append(outputs)
        self.labels.append(labels)
        
        mean_dice = np.mean(list(dices.values()))
        print(f"mean dice : {mean_dice:.4f}")
        
        # if self.use_wandb:
        #     self.log_plot(vis_data, mean_dice, dices, filename=str(self.patient_index))
        #     self.log("mean_dice", mean_dice)
        
        self.patient_index += 1
        
        return mean_dice
        
    def save_results(self):
        results = {
            "images": self.images,
            "dices": self.dices,
            "labels": self.labels,
            "outputs": self.outputs,
        }
        
        with open(os.path.join(self.log_dir, f'results.pkl'), 'wb') as file:
            pickle.dump(results, file)

      

if __name__ == "__main__":
    args = parse_args()
    tester = Tester(**vars(args))
    tester.test()