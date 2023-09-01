import os
import wandb
from tqdm import tqdm
import numpy as np

import torch 
import torch.nn as nn 
from torch.nn.parallel import DataParallel

from monai.utils import set_determinism
from monai.losses.dice import DiceLoss

from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from metric import *
from engine import Engine
from utils import parse_args, save_model, get_amosloader
from dataset_path import data_dir

set_determinism(123)

        
class AMOSTrainer(Engine):
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
        model_path=None,
        wandb_name=None,
        pretrained=True,
        use_amp=True,
        use_cache=True,
        use_wandb=True
    ):
        super().__init__(
            model_name=model_name, 
            image_size=image_size,
            depth=depth,
            num_classes=num_classes, 
            device=device,
            num_workers=num_workers,
            model_path=model_path,
            wandb_name=wandb_name,
            pretrained=pretrained,
            use_amp=use_amp,
            use_wandb=use_wandb,
            mode="train",
        )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.loss_combine = loss_combine
        self.log_dir = os.path.join("logs", log_dir)
        self.use_cache = use_cache
        
        self.start_epoch = 0
        self.wandb_id = None
        self.local_rank = 0
        self.weights_path = os.path.join(self.log_dir, "weights")
        self.auto_optim = True
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)
                
        self.model = self.load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=100,
                                                       max_epochs=max_epochs)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        
        if model_path is not None:
            self.load_checkpoint(model_path)
                
        if self.num_gpus > 1:
            self.model = DataParallel(self.model)
            
        if use_wandb:
            if model_path is None:
                if wandb_name is None: wandb_name = log_dir
                wandb.init(project="diff-unet", 
                           name=f"{wandb_name}",
                           config=self.__dict__)
            else:
                wandb.init(project="diff-unet", 
                           id=self.wandb_id, 
                           resume=True)
        
    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_mean_dice = checkpoint['best_mean_dice']
        self.wandb_id = checkpoint['id']
        
        print(f"Checkpoint loaded from {model_path}")

    def train(self, train_dataset, val_dataset=None):
        set_determinism(1234 + self.local_rank)
        print(f"check model parameter: {next(self.model.parameters()).sum()}")
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        print(f"model parameters is {para * 4 / 1000 / 1000}M ")
        
        os.makedirs(self.log_dir, exist_ok=True)

        train_loader = self.get_dataloader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = self.get_dataloader(val_dataset, batch_size=1, shuffle=False)
            
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch 
            
            with torch.cuda.amp.autocast(self.use_amp):
                self.train_epoch(train_loader, epoch)
                
                val_outputs = []
                
                if (epoch + 1) % self.val_freq == 0:
                    self.model.eval()
                    for idx, (batch, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }

                        with torch.no_grad():
                            val_out = self.validation_step(batch)
                            assert val_out is not None 

                        val_outputs.append(val_out)

                    val_outputs = torch.tensor(val_outputs)

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
            running_loss = 0
            for idx, (batch, filename) in enumerate(loader):
                self.global_step += 1
                self.optimizer.zero_grad()
                t.set_description('Epoch %i' % epoch)
                
                batch = batch[0]
                batch = {
                    x: batch[x].contiguous().to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
                
                for param in self.model.parameters(): param.grad = None
                loss = self.training_step(batch)
                running_loss += loss.item()
                
                if self.auto_optim:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                        
                    self.scheduler.step()
                    
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                    t.set_postfix(loss=loss.item(), lr=lr)
                t.update(1)
                
            self.log("loss", running_loss / len(loader), epoch)
                    
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

        return loss 
    
    def validation_end(self, mean_val_outputs, epoch):
        dices = mean_val_outputs
        mean_dice = sum(dices) / len(dices)

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
        self.log("mean_dice", mean_dice)

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

if __name__ == "__main__":
    args = parse_args("train")

    trainer = AMOSTrainer(**vars(args))
    train_ds, val_ds = get_amosloader(data_dir=data_dir, mode="train", use_cache=args.use_cache)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
