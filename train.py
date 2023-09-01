import os
import wandb
from tqdm import tqdm
import numpy as np

import torch 
import torch.nn as nn 
from torch.nn.parallel import DataParallel

from monai.data import DataLoader
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from monai.inferers import SlidingWindowInferer

from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_model

from metric import *
from unet.diff_unet import DiffUNet
from unet.smooth_diff_unet import SmoothDiffUNet
from utils import get_amosloader, parse_args
from dataset_path import data_dir

set_determinism(123)

        
class AMOSTrainer:
    def __init__(
        self, 
        model_name="diff_unet",
        max_epochs=5000,
        batch_size=10, 
        image_size=256,
        depth=96,
        num_classes=16,
        val_freq=1, 
        save_freq=5,
        device="cpu", 
        num_gpus=1, 
        num_workers=2,
        loss_combine='plus',
        log_dir="logs", 
        resume_path=None,
        pretrained=True,
        use_cache=True,
        use_wandb=True
    ):
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.depth = depth
        self.num_classes = num_classes
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.device = torch.device(device)
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.loss_combine = loss_combine
        self.log_dir = os.path.join("logs", log_dir)
        self.resume_path = resume_path
        self.pretrained = pretrained
        self.use_cache = use_cache
        self.use_wandb = use_wandb
        self.start_epoch = 0
        self.global_step = 0
        self.wandb_id = None
        self.local_rank = 0
        self.weights_path = os.path.join(self.log_dir, "weights")
        self.auto_optim = True
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size

        self.window_infer = SlidingWindowInferer(roi_size=[depth, image_size, image_size],
                                                 sw_batch_size=1,
                                                 overlap=0.5)

        if model_name == "diff_unet":
            _model = DiffUNet
        elif model_name == "smooth_diff_unet":
            _model = SmoothDiffUNet
        else:
            raise ValueError(f"Invalid model_type: {model_name}")
        
        self.model = _model(image_size=image_size,
                            depth=depth,
                            num_classes=num_classes,
                            device=device,
                            pretrained=pretrained,
                            mode="train").to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=100,
                                                       max_epochs=max_epochs)

        if resume_path is not None:
            self.load_checkpoint(resume_path)
            
        if self.use_wandb:
            if resume_path is None:
                wandb.init(project="diff-unet", 
                           name=f"{log_dir}",
                           config=self.__dict__)
            else:
                wandb.init(project="diff-unet", 
                           id=self.wandb_id, 
                           resume=True)
                
            # wandb.init(project="diff-unet", 
            #             name=f"{log_dir}",
            #             config=self.__dict__)
                
        if self.num_gpus > 1:
            self.model = DataParallel(self.model)
            
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_mean_dice = checkpoint['best_mean_dice']
        self.wandb_id = checkpoint['id']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        
    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers)

    def train(self, train_dataset, val_dataset=None):
        set_determinism(1234 + self.local_rank)
        print(f"check model parameter: {next(self.model.parameters()).sum()}")
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        if self.local_rank == 0:
            print(f"model parameters is {para * 4 / 1000 / 1000}M ")
        
        os.makedirs(self.log_dir, exist_ok=True)

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
            
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch 
            self.train_epoch(train_loader, epoch)
            
            val_outputs = []
            
            if (epoch + 1) % self.val_freq == 0:
                self.model.eval()
                for idx, (batch, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list) :
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    
                    else:
                        print("not support data type")
                        exit(0)

                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None 

                    val_outputs.append(val_out)

                val_outputs = torch.tensor(val_outputs)

                if self.local_rank == 0:
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

                    self.validation_end(mean_val_outputs=v_sum, epoch=epoch)
            
            self.model.train()

    def train_epoch(self, loader, epoch):
        self.model.train()
        with tqdm(total=len(loader)) as t:
            for idx, (batch, filename) in enumerate(loader):
                self.global_step += 1
                t.set_description('Epoch %i' % epoch)
                
                batch = batch[0]
                
                if isinstance(batch, dict):
                    batch = {
                        x: batch[x].contiguous().to(self.device)
                        for x in batch if isinstance(batch[x], torch.Tensor)
                    }
                elif isinstance(batch, list) :
                    batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                
                else :
                    print("not support data type")
                    exit(0)
                
                for param in self.model.parameters(): param.grad = None
                loss = self.training_step(batch)

                if self.auto_optim:
                    loss.backward()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                    t.set_postfix(loss=loss.item(), lr=lr)
                t.update(1)
                    
        if (epoch + 1) % self.save_freq == 0:
            save_model(self.model,
                                           self.optimizer,
                                           self.scheduler, 
                                           self.epoch,
                                           self.global_step,
                                           self.best_mean_dice,
                                           wandb.run.id,
                                           os.path.join(self.weights_path, f"epoch_{epoch}.pt"))
        
    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        if self.loss_combine == 'plus':
            loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss.item(), step=self.global_step)

        return loss 
    
    def validation_end(self, mean_val_outputs, epoch):
        dices = mean_val_outputs
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_model(self.model,
                                           self.optimizer,
                                           self.scheduler, 
                                           self.epoch,
                                           self.global_step,
                                           self.best_mean_dice,
                                           wandb.run.id,
                                           os.path.join(self.weights_path, f"best_{mean_dice:.4f}.pt"))

        print(f"mean_dice : {mean_dice}")
        

    def validation_step(self, batch):
        image, label = self.get_input(batch)  
        
        output = self.window_infer(image, self.model.module, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()
        target = label.cpu().numpy()

        dices = []
        hd = []
        for i in range(self.num_classes):
            pred_c = output[:, i]
            target_c = target[:, i]

            dices.append(dice_coef(pred_c, target_c))
            # hd.append(hausdorff_distance_95(pred_c, target_c))
        
        return dices

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        
        label = self.convert_labels(label)
        label = label.float()
        
        return image, label

    def convert_labels(self, labels):
        labels_new = []
        for i in range(self.num_classes):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def log(self, k, v, step):
        if self.use_wandb:
            wandb.log({k: v}, step=step)
    

if __name__ == "__main__":
    args = parse_args()

    log_dir = args.log_dir
    model_name = args.model_name
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    loss_combine = args.loss_combine
    device = args.device
    val_freq = args.val_freq
    num_gpus = args.num_gpus
    resume_path = args.resume_path
    pretrained = args.pretrained
    use_wandb = args.use_wandb
    use_cache = args.use_cache
    # resume_path = "logs/amos_remastered/model/epoch_24.pt"

    trainer = AMOSTrainer(model_name=model_name,
                          max_epochs=max_epoch,
                          batch_size=batch_size,
                          device=device,
                          val_freq=val_freq,
                          num_gpus=num_gpus,
                          num_workers=num_workers,
                          loss_combine=loss_combine,
                          log_dir=log_dir,
                          resume_path=resume_path,
                          pretrained=pretrained,
                          use_wandb=use_wandb,
                          use_cache=use_cache)

    train_ds, val_ds = get_amosloader(data_dir=data_dir, mode="train", use_cache=use_cache)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
