import os
import wandb
import torch 
from tqdm import tqdm
from prettytable import PrettyTable
from collections import OrderedDict

from medpy.metric.binary import dc, hd95

from monai.data import DataLoader
from monai.utils import set_determinism

from metric import iou_score
from engine import Engine
from utils import parse_args, get_data_path, get_dataloader

set_determinism(123)
        
class Tester(Engine):
    def __init__(
        self, 
        model_path,
        model_name="diff_unet",
        data_name="amos",
        image_size=256,
        spatial_size=96,
        classes=None,
        epoch=800,
        device="cpu", 
        project_name="diff-unet-test",
        wandb_name=None,
        pretrained=True,
        remove_bg=False,
        use_amp=True,
        use_cache=False,
        use_wandb=True,
    ):
        super().__init__(
            model_name=model_name,
            data_name=data_name, 
            image_size=image_size,
            spatial_size=spatial_size,
            classes=classes, 
            device=device,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            remove_bg=remove_bg,
            use_amp=use_amp,
            use_cache=use_cache,
            use_wandb=use_wandb,
            mode="test",
        )
        self.epoch = epoch
        self.pretrained = pretrained
        self.model = self.load_model()
        self.load_checkpoint(model_path)
        self.set_dataloader()    
        
        if use_wandb:
            wandb.init(project=self.project_name,
                       name=self.wandb_name,
                       config=self.__dict__)
            self.table = wandb.Table(columns=["patient", "image", "dice", "hd95", "iou"]+[n for n in self.class_names.values()])

    def load_checkpoint(self, model_path):
        if self.epoch is not None:
            model_path = os.path.join(os.path.dirname(model_path), f"epoch_{self.epoch}.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict['model'])
            
        print(f"Checkpoint loaded from {model_path}.....")

    def set_dataloader(self):
        dataset = get_dataloader(data_path=get_data_path(self.data_name),
                                 data_name=self.data_name,
                                 image_size=self.image_size,
                                 spatial_size=self.spatial_size,
                                 mode=self.mode, 
                                 use_cache=self.use_cache)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    def validation_step(self, batch, filename):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
            
        _, _, d, w, h = label.shape
        image = torch.nn.functional.interpolate(image, mode="nearest", size=(d, w, h))
        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h)) # idea

        if self.use_wandb:
            vis_data = self.tensor2images(image, label, output, image.shape) # put appropriate index
            
        dices = OrderedDict({v : 0 for v in self.class_names.values()})
        hds = OrderedDict({v : 0 for v in self.class_names.values()})
        ious = OrderedDict({v : 0 for v in self.class_names.values()})
        
        for i in range(self.num_classes):
            pred = output[:, i]
            gt = label[:, i]
            
            if torch.sum(pred) > 0 and torch.sum(gt) > 0:
                dice = dc(pred, gt)
                hd = hd95(pred, gt)
                iou = iou_score(pred, gt)
            elif torch.sum(pred) > 0 and torch.sum(gt) == 0:
                dice = 1
                hd = 0
                iou = 1
            else:
                dice = 0
                hd = 0
                iou = 0
            
            dices[self.class_names[i]] = dice
            hds[self.class_names[i]] = hd
            ious[self.class_names[i]] = iou
            
            table = PrettyTable()
            table.title = self.class_names[i]
            table.field_names = ["metric", "score"]
            table.add_row(["dice", f"{dice:.4f}"])
            table.add_row(["hd95", f"{hd:.4f}"])
            table.add_row(["iou", f"{iou:.4f}"])
            print(table)
        
        mean_dice = sum(dices.values()) / self.num_classes
        mean_hd95 = sum(hds.values()) / self.num_classes
        mean_iou = sum(ious.values()) / self.num_classes
        
        # print(f"mean_dice : {mean_dice:.4f}")
        # print(f"mean_hd95 : {mean_hd95:.4f}")
        # print(f"mean_iou : {mean_iou:.4f}")
        
        if self.use_wandb:
            self.log_plot(vis_data, mean_dice, mean_hd95, mean_iou, dices, filename)
            self.log("mean_dice", mean_dice)
            self.log("mean_hd95", mean_hd95)
            self.log("mean_iou", mean_iou)
        
    def test(self):
        self.model.eval()
        
        with torch.cuda.amp.autocast(self.use_amp):
            for batch, filename in tqdm(self.dataloader, total=len(self.dataloader)):
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
                
                with torch.no_grad():
                    self.validation_step(batch, filename[0])
                
                self.global_step += 1
                
        if self.use_wandb: wandb.log({"table": self.table})    

if __name__ == "__main__":
    args = parse_args()
    tester = Tester(**vars(args))
    tester.test()