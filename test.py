import os
import glob
import wandb
import torch 
from tqdm import tqdm
from collections import OrderedDict

from medpy.metric import dc, hd95
from monai.utils import set_determinism

from metric import *
from engine import Engine
from utils import parse_args, get_class_names, get_amosloader
from dataset_path import data_dir

set_determinism(123)
        
class AMOSTester(Engine):
    def __init__(
        self, 
        model_path,
        model_name="smooth_diff_unet",
        image_size=256,
        spatial_size=96,
        class_names=None,
        num_classes=16,
        device="cpu", 
        project_name="diff-unet-test",
        wandb_name=None,
        pretrained=True,
        use_amp=True,
        use_wandb=True,
    ):
        super().__init__(
            model_name=model_name, 
            image_size=image_size,
            spatial_size=spatial_size,
            class_names=class_names,
            num_classes=num_classes, 
            device=device,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            pretrained=pretrained,
            use_amp=use_amp,
            use_wandb=use_wandb,
            mode="test",
        )
        if use_wandb:
            wandb.init(project=self.project_name, name=self.wandb_name)

        self.model = self.load_model()
        self.load_checkpoint(model_path)

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        print(f"Checkpoint loaded from {model_path}.....")

    def validation_step(self, batch, filename):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        _, _, d, w, h = label.shape

        if self.use_wandb:
            vis_data = self.tensor2images(image, label, output, int(d * 0.75)) # put appropriate index

        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h)) # idea
        # output = output.cpu().numpy()
        # target = label.cpu().numpy()

        dices = OrderedDict({v : 0 for v in self.class_names.values()})
        # hd = []
        
        for i in range(self.num_classes):
            pred = output[:, i]
            gt = label[:, i]

            if torch.sum(pred) > 0 and torch.sum(gt) > 0:
                dice = dice_score(pred, gt, i)
                # hd95 = hd95_score(pred, gt, i)
            elif torch.sum(pred) > 0 and torch.sum(gt) == 0:
                dice = 1
                # hd95 = 0
            else:
                dice = 0
                # hd95 = 0
            
            if isinstance(dice, int):
                dices[self.class_names[i]] = dice
            else:
                dices[self.class_names[i]] = dice.item()
            # hd.append(hd95)
        
        all_m = []
        for d in dices:
            all_m.append(d)
        # for h in hd:
        #     all_m.append(h)
        
        mean_dice = sum(dices.values()) / self.num_classes
        # mean_hd95 = sum(hd) / self.num_classes
        
        print(f"mean_dice : {mean_dice:.4f}")
        # print(f"mean_hd95 : {mean_hd95:.4f}")
        
        if self.use_wandb:
            self.log_plot(vis_data, mean_dice, dices, filename)
            self.log("mean_dice", mean_dice)
        
        return all_m 
    
    def test(self, val_dataset):
        # try:
        val_loader = self.get_dataloader(val_dataset, batch_size=1, shuffle=False)
        val_outputs = []
        self.model.eval()
        
        with torch.cuda.amp.autocast(self.use_amp):
            for idx, (batch, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = batch[0]
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
                
                with torch.no_grad():
                    val_out = self.validation_step(batch, filename[0])
                    assert val_out is not None 
                
                self.global_step += 1
            #     return_list = False
            #     val_outputs.append(val_out)
                
            # if isinstance(val_out, list) or isinstance(val_out, tuple):
            #     return_list = True

            # val_outputs = torch.tensor(val_outputs)
            # if not return_list:
            #     length = 0
            #     v_sum = 0.0
            #     for v in val_outputs:
            #         if not torch.isnan(v):
            #             v_sum += v
            #             length += 1

            #     if length == 0:
            #         v_sum = 0
            #     else :
            #         v_sum = v_sum / length             
            # else :
            #     num_val = len(val_outputs[0])
            #     length = [0.0 for i in range(num_val)]
            #     v_sum = [0.0 for i in range(num_val)]

            #     for v in val_outputs:
            #         for i in range(num_val):
            #             if not torch.isnan(v[i]):
            #                 v_sum[i] += v[i]
            #                 length[i] += 1

            #     for i in range(num_val):
            #         if length[i] == 0:
            #             v_sum[i] = 0
            #         else :
            #             v_sum[i] = v_sum[i] / length[i]
                
            
        
        if self.use_wandb: wandb.log({"table": self.table})    

        return v_sum, val_outputs
        
        # except Exception as e:
        #     print(e)
        #     if self.use_wandb: wandb.log({"table": self.table})    
    
        
if __name__ == "__main__":
    class_names = get_class_names("amos")
    args = parse_args("test", project_name="diff-unet-test")
    
    tester = AMOSTester(**vars(args), class_names=class_names)
    test_ds = get_amosloader(data_dir=data_dir, 
                             image_size=args.image_size, 
                             spatial_size=args.spatial_size, 
                             mode="test", 
                             use_cache=False)
    v_mean, v_out = tester.test(test_ds)