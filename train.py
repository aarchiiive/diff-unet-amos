import os
import pytz
import wandb
import argparse
import datetime
from tqdm import tqdm
import numpy as np

import torch 
import torch.nn as nn 
from torch.nn.parallel import DataParallel

from monai.data import DataLoader
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedEvaluator
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe

from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

from metric import *
from utils import get_amosloader
from dataset_path import data_dir

set_determinism(123)


class DiffUNet(nn.Module):
    def __init__(self, 
                 image_size,
                 depth,
                 num_classes,
                 device,
                 ):
        super().__init__()
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
            
        self.depth = depth
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, num_classes, num_classes-1, [64, 64, 128, 256, 512, 64], 
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
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, 
                                                                (1, self.num_classes-1, self.depth, self.width,  self.height), 
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"].to(self.device)
            return sample_out
        
class AMOSTrainer:
    def __init__(self, 
                 max_epochs,
                 batch_size, 
                 image_size=256,
                 depth=96,
                 num_classes=16,
                 device="cpu", 
                 val_every=1, 
                 save_freq=5,
                 num_gpus=1, 
                 logdir="logs", 
                 resume_path=None,
                 use_cache=True,
                 use_wandb=True
                ):
        
        self.val_every = val_every
        self.save_freq = save_freq
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.depth = depth
        self.num_classes = num_classes
        self.ddp = num_gpus > 1
        self.num_gpus = num_gpus
        self.device = torch.device(device)
        self.local_rank = 0
        self.batch_size = batch_size
        self.logdir = os.path.join("logs", logdir)
        self.weights_path = os.path.join(self.logdir, "weights")
        self.auto_optim = True
        self.use_cache = use_cache
        self.use_wandb = use_wandb
        self.start_epoch = 0
        
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)
        
        if isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        elif isinstance(image_size, int):
            self.width = self.height = image_size
        
        if self.use_wandb:
            wandb.init(project="diff-unet", 
                       name=f"{logdir}", 
                       config=self.__dict__, 
                       resume=resume_path is not None)

        self.window_infer = SlidingWindowInferer(roi_size=[depth, image_size, image_size],
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        
        self.model = DiffUNet(image_size=image_size,
                              depth=depth,
                              num_classes=num_classes,
                              device=device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=100,
                                                  max_epochs=max_epochs)

        if resume_path is not None:
            self.load_checkpoint(resume_path)
            
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
        if 'best_mean_dice' in checkpoint.keys():
            self.best_mean_dice = checkpoint['best_mean_dice']

        print(f"Checkpoint loaded from {checkpoint_path}")
        
    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=2)

    def train(self, train_dataset, val_dataset=None):
        set_determinism(1234 + self.local_rank)
        print(f"check model parameter: {next(self.model.parameters()).sum()}")
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        if self.local_rank == 0:
            print(f"model parameters is {para * 4 / 1000 / 1000}M ")
                
        self.global_step = 0
        
        os.makedirs(self.logdir, exist_ok=True)

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
            
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch 
            self.train_epoch(train_loader, epoch)
            
            val_outputs = []
            
            if (epoch + 1) % self.val_every == 0:
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
            
        if self.local_rank == 0:
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
            save_new_model_and_delete_last(self.model,
                                           self.optimizer,
                                           self.scheduler, 
                                           epoch,
                                           self.best_mean_dice,
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
        for i in range(1, self.num_classes):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def validation_end(self, mean_val_outputs, epoch):
        dices = mean_val_outputs
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           self.optimizer,
                                           self.scheduler, 
                                           epoch,
                                           self.best_mean_dice,
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
        for i in range(self.num_classes-1):
            pred_c = output[:, i]
            target_c = target[:, i]

            dices.append(dice_coef(pred_c, target_c))
            # hd.append(hausdorff_distance_95(pred_c, target_c))
        
        return dices
    
    def log(self, k, v, step):
        if self.use_wandb:
            wandb.log({k: v}, step=step)
    
def parse_args():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument("--logdir", type=str,
                        help="Directory to store log files")
    parser.add_argument("--max_epoch", type=int, default=50000,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--val_every", type=int, default=100,
                        help="Validation frequency (number of iterations)")
    parser.add_argument("--num_gpus", type=int, default=5,
                        help="Number of GPUs to use for training")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to the checkpoint for resuming training")
    parser.add_argument("--use_wandb", action="store_true", default=True, # default=False,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--use_cache", action="store_true", default=True, # default=False,
                        help="Enable caching")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    logdir = args.logdir
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    device = args.device
    val_every = args.val_every
    num_gpus = args.num_gpus
    resume_path = args.resume_path
    use_wandb = args.use_wandb
    use_cache = args.use_cache
    # resume_path = "logs/amos_remastered/model/epoch_24.pt"

    trainer = AMOSTrainer(max_epochs=max_epoch,
                          batch_size=batch_size,
                          device=device,
                          logdir=logdir,
                          val_every=val_every,
                          num_gpus=num_gpus,
                          resume_path=resume_path,
                          use_wandb=use_wandb,
                          use_cache=use_cache)

    train_ds, val_ds = get_amosloader(data_dir=data_dir, mode="train", use_cache=use_cache)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
