from collections import OrderedDict

import os
import wandb
from tqdm import tqdm
import numpy as np
from medpy.metric.binary import dc, hd95

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel

from monai.data import DataLoader
from monai.utils import set_determinism

from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from engine import Engine
from models.model_type import ModelType
from utils import parse_args, get_data_path, get_dataloader
from losses.utils import dist_map_transform
from losses.hub import *
set_determinism(123)

        
class Trainer(Engine):
    def __init__(
        self, 
        model_name="diff_unet",
        data_name="amos",
        max_epochs=5000,
        batch_size=10, 
        image_size=256,
        spatial_size=96,
        classes=None,
        val_freq=1, 
        save_freq=5,
        device="cpu", 
        num_gpus=1, 
        num_workers=2,
        losses="mse,bce,dice",
        loss_combine='plus',
        log_dir="logs", 
        model_path=None,
        pretrained_path=None,
        project_name="diff-unet",
        wandb_name=None,
        remove_bg=False,
        use_amp=True,
        use_cache=True,
        use_wandb=True,
    ):
        super().__init__(
            model_name=model_name, 
            data_name=data_name,
            image_size=image_size,
            spatial_size=spatial_size,
            classes=classes,
            device=device,
            num_workers=num_workers,
            losses=losses,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            remove_bg=remove_bg,
            use_amp=use_amp,
            use_cache=use_cache,
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
        self.pretrained = pretrained_path is not None
        self.use_cache = use_cache
        
        self.local_rank = 0
        self.start_epoch = 0
        self.wandb_id = None
        self.weights_path = os.path.join(self.log_dir, "weights")
        self.auto_optim = True
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)
        
        self.set_dataloader()        
        self.model = self.load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=10,
                                                       max_epochs=max_epochs)
        self.dist_transform = dist_map_transform([1.0, 1.0, 1.0, 1.0], self.num_classes)
        
        if model_path is not None:
            self.load_checkpoint(model_path)
        elif self.pretrained:
            self.load_pretrained_weights(pretrained_path)
                
        if self.num_gpus > 1:
            self.model = DataParallel(self.model)
            
        if use_wandb:
            if model_path is None:
                if wandb_name is None: wandb_name = log_dir
                wandb.init(project=self.project_name, 
                           name=f"{wandb_name}",
                           config=self.__dict__)
            else:
                assert self.wandb_id != 0
                wandb.init(project=self.project_name, 
                           id=self.wandb_id, 
                           resume=True)
        
    def load_checkpoint(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.start_epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.best_mean_dice = state_dict['best_mean_dice']
        self.wandb_id = state_dict['id']
        
        print(f"Checkpoint loaded from {model_path}")
        
    def load_pretrained_weights(self, pretrained_path):
        self.model.embed_model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        print(f"Pretrained weights...")
            
    def set_dataloader(self):
        train_ds, val_ds = get_dataloader(data_path=get_data_path(self.data_name),
                                          data_name=self.data_name,
                                          image_size=self.image_size,
                                          spatial_size=self.spatial_size,
                                          mode="train", 
                                          one_hot=self.model_type == ModelType.Diffusion,
                                          remove_bg=self.remove_bg,
                                          use_cache=self.use_cache)
        self.dataloader = {"train": DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
                           "val": DataLoader(val_ds, batch_size=1, shuffle=False)}
        
    def train(self):
        set_determinism(1234 + self.local_rank)
        print(f"check model parameter: {next(self.model.parameters()).sum()}")
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        print(f"model parameters is {para * 4 / 1000 / 1000}M ")
        
        os.makedirs(self.log_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch 
            
            with torch.cuda.amp.autocast(self.use_amp):
                self.train_epoch(epoch)
                
                val_outputs = []
                
                if (epoch + 1) % self.val_freq == 0:
                    self.model.eval()
                    for batch, _ in tqdm(self.dataloader["val"], total=len(self.dataloader["val"])):
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
                    length = [0.0 for _ in range(num_val)]
                    v_sum = [0.0 for _ in range(num_val)]

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

    def train_epoch(self, epoch):
        self.model.train()
        with tqdm(total=len(self.dataloader["train"])) as t:
            running_loss = 0
            for batch, _ in self.dataloader["train"]:
                self.global_step += 1
                self.optimizer.zero_grad()
                t.set_description('Epoch %i' % epoch)
                
                batch = batch[0]
                batch = {
                    x: batch[x].contiguous().to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
                
                for param in self.model.parameters(): param.grad = None
                loss = self.training_step(batch).float()
                running_loss += loss.item()
                
                if self.auto_optim:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                    t.set_postfix(loss=loss.item(), lr=lr)
                t.update(1)
                
            self.scheduler.step()
            
            self.loss = running_loss / len(self.dataloader["train"])
            # self.log("loss", running_loss / len(self.dataloader["train"]))
            self.log("loss", self.loss, step=epoch)
                    
        if (epoch + 1) % self.save_freq == 0:
            self.save_model(self.model,
                            self.optimizer,
                            self.scheduler, 
                            self.epoch,
                            os.path.join(self.weights_path, f"epoch_{epoch+1}.pt"))

    def training_step(self, batch):
        image, label = self.get_input(batch)
        
        if self.model_type == ModelType.Diffusion:
            x_start = label
            x_start = (x_start) * 2 - 1
            x_t, t, _ = self.model(x=x_start, pred_type="q_sample")
            pred = self.model(x=x_t, step=t, image=image, pred_type="denoise")
        elif self.model_type == ModelType.SwinUNETR:
            pred = self.model(image)

        return self.compute_loss(pred, label) 
    
    def compute_loss(self, preds, labels):
        loss = []
        for _loss in self.losses:
            if isinstance(_loss, MSELoss):
                loss.append(_loss(torch.sigmoid(preds), labels))
            elif isinstance(_loss, CrossEntropyLoss):
                loss.append(_loss(preds, labels))
            elif isinstance(_loss, BCEWithLogitsLoss):
                loss.append(_loss(preds, labels))
            elif isinstance(_loss, DiceLoss):
                loss.append(_loss(preds, labels))
            elif isinstance(_loss, BoundaryLoss):
                loss.append(_loss(F.softmax(preds, dim=1), self.dist_transform(labels)))
                
        if self.loss_combine == 'plus':
            return sum(loss)
        else:
            # TODO : implement loss combinations
            pass
    
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

            dices.append(dc(pred_c, target_c))
            hd.append(hd95(pred_c, target_c))
        
        return dices
    
    def validation_end(self, mean_val_outputs, epoch):
        dices = mean_val_outputs
        mean_dice = sum(dices) / len(dices)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            self.save_model(self.model,
                            self.optimizer,
                            self.scheduler, 
                            self.epoch,
                            os.path.join(self.weights_path, f"best_{mean_dice:.4f}.pt"))

        print(f"mean_dice : {mean_dice}")
        self.log("mean_dice", mean_dice, epoch)

# def main():
#     dist_url = "env://"
#     rank = int(os.environ['RANK'])
#     world_size = int(os.environ(['WORLD_SIZE']))
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.distributed.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
#     torch.cuda.set_device(local_rank)
#     torch.distributed

if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(**vars(args))
    trainer.train()
