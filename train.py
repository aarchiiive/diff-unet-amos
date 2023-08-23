import os
import wandb
import datetime
import argparse
from tqdm import tqdm
import numpy as np

import torch 
import torch.nn as nn 
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from monai.inferers import SlidingWindowInferer

from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe

from light_training.trainer import Trainer
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

from dataset.amosloader import get_amosloader
from dataset_path import dataset_dir


set_determinism(123)

import os
import sys

class DiffUNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, num_classes+1, num_classes, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 16, 96, 256, 256), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
        
class AMOSTrainer:
    def __init__(self, 
                 env_type, 
                 max_epochs,
                 batch_size, 
                 image_size=256,
                 depth=96,
                 num_classes=16,
                 device="cpu", 
                 val_every=1, 
                 num_gpus=1, 
                 logdir="./logs/", 
                 resume_path=None,
                 master_ip='localhost', 
                 master_port=17750, 
                 training_script="train.py",
                 use_wandb=True
                 ):
        
        self.env_type = env_type
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.depth = depth
        self.num_classes = num_classes
        self.ddp = False
        self.num_gpus = num_gpus
        self.device = torch.device(device)
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.not_call_launch = True
        self.logdir = logdir
        self.model_save_path = os.path.join(logdir, "model")
        self.scheduler = None 
        self.auto_optim = True
        self.use_wandb = use_wandb  # Change this to False if you don't want to use WandB
        self.start_epoch = 0
        
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        if self.use_wandb:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            wandb.init(project="diff-unet", name=f"{current_time}", config=self.__dict__)

        
        self.window_infer = SlidingWindowInferer(roi_size=[depth, image_size, image_size],
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.model = DiffUNet(num_classes=num_classes).to(self.device)
        
        if self.num_gpus > 1:
            self.model = DataParallel(self.model)
            
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=100,
                                                  max_epochs=max_epochs)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        
        if resume_path is not None:
            self.load_checkpoint(resume_path)
            
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        # 추가적인 필요한 상태 로드

        print(f"Checkpoint loaded from {checkpoint_path}")
        
    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=2)

    def train(self,
                train_dataset,
                optimizer=None,
                model=None,
                val_dataset=None,
                scheduler=None,
              ):
        
        if scheduler is not None:
            self.scheduler = scheduler

        set_determinism(1234 + self.local_rank)
        if self.model is not None:
            print(f"check model parameter: {next(self.model.parameters()).sum()}")
            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            if self.local_rank == 0:
                print(f"model parameters is {para * 4 / 1000 / 1000}M ")
                
        self.global_step = 0
        
        os.makedirs(self.logdir, exist_ok=True)
        # self.writer = SummaryWriter(self.logdir)

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        
        if val_dataset is not None:
            val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
        else :
            val_loader = None 
            
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch 
            
            self.train_epoch(
                            train_loader,
                            epoch,
                            )
            
            val_outputs = []
            
            if (epoch + 1) % self.val_every == 0 \
                    and val_loader is not None :
                if self.model is not None:
                    self.model.eval()
                if self.ddp:
                    torch.distributed.barrier()
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list) :
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    
                    else :
                        print("not support data type")
                        exit(0)

                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None 

                    return_list = False
                    val_outputs.append(val_out)
                    if isinstance(val_out, list) or isinstance(val_out, tuple):
                        return_list = True

                val_outputs = torch.tensor(val_outputs)

                if self.local_rank == 0:
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
                        self.validation_end(mean_val_outputs=v_sum)
                    
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

                        self.validation_end(mean_val_outputs=v_sum)
            
            if self.model is not None:
                self.model.train()


    def train_epoch(self, 
                    loader,
                    epoch,
                    ):
        if self.model is not None:
            self.model.train()
            
        if self.local_rank == 0:
            with tqdm(total=len(loader)) as t:

                for idx, batch in enumerate(loader):
                    self.global_step += 1
                    t.set_description('Epoch %i' % epoch)
                    
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
                    
                    if self.model is not None:
                        for param in self.model.parameters(): param.grad = None
                    loss = self.training_step(batch)

                    if self.auto_optim:
                        loss.backward()
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                        t.set_postfix(loss=loss.item(), lr=lr)
                    t.update(1)
                    
                    # save_new_model_and_delete_last(self.model,
                    #                                loss.item(),
                    #                                self.optimizer,
                    #                                self.scheduler, 
                    #                                epoch,
                    #                                os.path.join(self.model_save_path, f"epoch_{epoch}.pt"))
        else:
            for idx, batch in enumerate(loader):
                self.global_step += 1
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

            for param in self.model.parameters() : param.grad = None

        save_new_model_and_delete_last(self.model,
                                       loss.item(),
                                       self.optimizer,
                                       self.scheduler, 
                                       epoch,
                                       os.path.join(self.model_save_path, f"epoch_{epoch}.pt"))
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

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss.item(), step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        
        label = self.convert_labels(label)
        label = label.float()

        return image, label

    def convert_labels(self, labels):
        labels_new = []
        for i in range(16):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def validation_end(self, mean_val_outputs):
        dices = mean_val_outputs
        print(dices)
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice.item(), step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f" mean_dice is {mean_dice}")

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        dices = []
        hd = []
        c = 16
        for i in range(c):
            pred_c = output[:, i]
            target_c = target[:, i]

            dices.append(dice(pred_c, target_c))
            # hd.append(hausdorff_distance_95(pred_c, target_c))
        
        return dices
    
    def log(self, k, v, step):
        if self.use_wandb:
            wandb.log({k: v}, step=step)
    
if __name__ == "__main__":
    # data_dir = "./RawData/Training/"

    logdir = "logs/amos"
    model_save_path = os.path.join(logdir, "model")

    max_epoch = 50000
    batch_size = 10
    val_every = 40000
    env = "DDP"
    num_gpus = 5
    
    # or
    # env = "pytorch"
    # num_gpus = 1

    device = "cuda:0"

    trainer = AMOSTrainer(env_type=env,
                          max_epochs=max_epoch,
                          batch_size=batch_size,
                          device=device,
                          logdir=logdir,
                          val_every=val_every,
                          num_gpus=num_gpus,
                          master_port=17751,
                          training_script=__file__,
                          use_wandb=True)

    train_ds, val_ds = get_amosloader(data_dir=dataset_dir, mode="train")

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
