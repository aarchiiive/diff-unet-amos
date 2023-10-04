import os
import wandb
import warnings

from tqdm import tqdm
from prettytable import PrettyTable
from collections import OrderedDict

import numpy as np

import torch 
from monai.utils import set_determinism

from engine import Engine
from utils import parse_args, get_dataloader

set_determinism(123)
warnings.filterwarnings("ignore")

class Tester(Engine):
    def __init__(
        self, 
        model_path,
        model_name="diff_unet",
        data_name="amos",
        data_path=None,
        sw_batch_size=4,
        overlap=0.25,
        image_size=256,
        spatial_size=96,
        classes=None,
        epoch=800,
        device="cpu", 
        project_name="diff-unet-test",
        wandb_name=None,
        include_background=False,
        use_amp=True,
        use_cache=False,
        use_wandb=True,
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
        self.model = self.load_model()
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
        state_dict = torch.load(model_path)
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
    
    def validation_step(self, batch):
        with torch.cuda.amp.autocast(self.use_amp):
            image, outputs, labels = self.infer(batch)
        
        if self.use_wandb:
            vis_data = self.tensor2images(image, labels, outputs, image.shape) # put appropriate index
            
        dices = OrderedDict({v : 0 for v in self.class_names.values()})
        hds = OrderedDict({v : 0 for v in self.class_names.values()})
        ious = OrderedDict({v : 0 for v in self.class_names.values()})
        
        classes = list(self.class_names.values())
        # output = torch.argmax(output)
        # label = torch.argmax(label)
        dices = []
        for i in range(self.num_classes):
            output = outputs[:, i]
            label = labels[:, i]
            if output.sum() > 0 and label.sum() > 0:
                self.dice_metric(output, label)
                dice = self.dice_metric.aggregate().item()
            elif output.sum() > 0 and label.sum() == 0:
                dice = 1
            
            print(f"{classes[i]} : {dice:.4f}")
            dices.append(dice)
        self.dice_metric.reset()
        print(f"mean dice : {np.mean(dices):.4f}")
        
        return np.mean(dices)
        
        # dice = self.dice_metric(output, label)
        # print(dice)
        # print(torch.mean(dice))
        # for i in range(self.num_classes):
        #     pred = output[..., i].unsqueeze(0)
        #     gt = label[..., i].unsqueeze(0)
            
        #     if torch.sum(pred) > 0 and torch.sum(gt) > 0:
        #         # dice = dc(pred, gt)
        #         dice = self.dice_metric(pred, gt)[0, 1]
        #         # hd = hd95(pred, gt)
        #         # iou = iou_score(pred, gt)
        #     elif torch.sum(pred) > 0 and torch.sum(gt) == 0:
        #         dice = 1
        #         hd = 0
        #         iou = 1
        #     else:
        #         dice = 0
        #         hd = 0
        #         iou = 0
            
        #     if self.remove_bg: i += 1
            
        #     dices[self.class_names[i]] = dice
        #     # hds[self.class_names[i]] = hd
        #     # ious[self.class_names[i]] = iou
            
        #     table = PrettyTable()
        #     table.title = self.class_names[i]
        #     table.field_names = ["metric", "score"]
        #     table.add_row(["dice", f"{dice:.4f}"])
        #     # table.add_row(["hd95", f"{hd:.4f}"])
        #     # table.add_row(["iou", f"{iou:.4f}"])
        #     print(table)
        
        # mean_dice = sum(dices.values()) / self.num_classes
        # # mean_hd95 = sum(hds.values()) / self.num_classes
        # # mean_iou = sum(ious.values()) / self.num_classes
        
        # print(f"mean_dice : {mean_dice:.4f}")
        # print(f"mean_hd95 : {mean_hd95:.4f}")
        # print(f"mean_iou : {mean_iou:.4f}")
        
        # if self.use_wandb:
        #     self.log_plot(vis_data, mean_dice, mean_hd95, mean_iou, dices, filename)
        #     self.log("mean_dice", mean_dice)
        #     self.log("mean_hd95", mean_hd95)
        #     self.log("mean_iou", mean_iou)
        
      

if __name__ == "__main__":
    args = parse_args()
    tester = Tester(**vars(args))
    tester.test()