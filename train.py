import os
import wandb
import warnings
from tqdm import tqdm

import numpy as np

import torch 
from torch.nn.parallel import DataParallel

from monai.utils import set_determinism
from monai.transforms import AsDiscrete

from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from engine import Engine
from models.model_type import ModelType
from utils import parse_args, get_dataloader

set_determinism(123)
warnings.filterwarnings("ignore")
        
        
class Trainer(Engine):
    def __init__(
        self, 
        model_name="diff_unet",
        data_name="amos",
        data_path=None,
        max_epochs=5000,
        batch_size=10, 
        sw_batch_size=4,
        overlap=0.25,
        image_size=256,
        spatial_size=96,
        lr=1e-4,
        weight_decay=1e-3,
        scheduler=None,
        warmup_epochs=100,
        timesteps=1000,
        classes=None,
        val_freq=1, 
        save_freq=5,
        device="cpu", 
        device_ids=[0, 1, 2, 3],
        num_workers=2,
        losses="mse,bce,dice",
        loss_combine='sum',
        log_dir="logs", 
        model_path=None,
        pretrained_path=None,
        project_name="diff-unet",
        wandb_name=None,
        include_background=False,
        use_amp=True,
        use_cache=True,
        use_wandb=True,
    ):
        super().__init__(
            model_name=model_name, 
            data_name=data_name,
            data_path=data_path,
            batch_size=batch_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            image_size=image_size,
            spatial_size=spatial_size,
            timesteps=timesteps,
            classes=classes,
            device=device,
            num_workers=num_workers,
            losses=losses,
            loss_combine=loss_combine,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            include_background=include_background,
            use_amp=use_amp,
            use_cache=use_cache,
            use_wandb=use_wandb,
            mode="train",
        )
        self.max_epochs = max_epochs
        self.lr = float(lr)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.num_workers = num_workers
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        if scheduler is not None:
            print("Training with scheduler...")
            self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                           warmup_epochs=warmup_epochs,
                                                           max_epochs=max_epochs)
            
        if model_path is not None:
            self.load_checkpoint(model_path)
        elif self.pretrained:
            self.load_pretrained_weights(pretrained_path)
                
        if device_ids:
            self.model = DataParallel(self.model, device_ids=list(map(int, device_ids.split(','))))
            
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
        for k in ['model', 'optimizer', 'scheduler']:
            if state_dict[k] is not None:
                getattr(self, k).load_state_dict(state_dict[k])
        self.start_epoch = state_dict['epoch']
        self.project_name = state_dict['project_name']
        self.global_step = state_dict['global_step']
        self.best_mean_dice = state_dict['best_mean_dice']
        self.wandb_id = state_dict['id']
        
        print(f"Checkpoint loaded from {model_path}")
        
    def load_pretrained_weights(self, pretrained_path):
        if self.model_type == ModelType.Diffusion:
            self.model.embed_model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        elif self.model_type == ModelType.SwinUNETR:
            self.model.load_from(weights=torch.load(pretrained_path, map_location="cpu"))
        print(f"Load pretrained weights from {pretrained_path}")
            
    def set_dataloader(self):
        self.dataloader = get_dataloader(data_path=self.data_path,
                                         image_size=self.image_size,
                                         spatial_size=self.spatial_size,
                                         num_workers=self.num_workers,
                                         batch_size=self.batch_size,
                                         mode="train")
    
    def train(self):
        set_determinism(1234 + self.local_rank)
        print(f"check model parameter: {next(self.model.parameters()).sum():.4f}")
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        print(f"model parameters is {para * 4 / 1000 / 1000:.2f}M ")
        
        os.makedirs(self.log_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch 
            self.train_epoch(epoch)
            
            if (epoch + 1) % self.val_freq == 0:
                self.model.eval()
                dices = []
                for batch in tqdm(self.dataloader["val"], total=len(self.dataloader["val"])):
                    with torch.no_grad():
                        dices.append(self.validation_step(batch))

                self.validation_end(dices, epoch)
            
    def train_epoch(self, epoch):
        running_loss = 0
        self.model.train()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
        with tqdm(total=len(self.dataloader["train"])) as t:
            for batch in self.dataloader["train"]:
                self.global_step += 1
                self.optimizer.zero_grad()
                t.set_description('Epoch %i' % epoch)
                
                for param in self.model.parameters(): param.grad = None
                with torch.cuda.amp.autocast(self.use_amp):
                    loss = self.training_step(batch).float()
                
                t.set_postfix(loss=loss.item(), lr=lr)
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                running_loss += loss.item()
                t.update(1)
                
            if self.scheduler is not None: self.scheduler.step()
            
            self.loss = running_loss / len(self.dataloader["train"])
            self.log("loss", self.loss, step=epoch)
                    
        if (epoch + 1) % self.save_freq == 0:
            self.save_model(model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=self.epoch,
                            save_path=os.path.join(self.weights_path, f"epoch_{epoch+1}.pt"))

    def training_step(self, batch):
        image, label = self.get_input(batch)
        
        if self.model_type == ModelType.Diffusion:
            x_start = (label) * 2 - 1
            x_t, t, _ = self.model(x=x_start, pred_type="q_sample")
            pred = self.model(x=x_t, step=t, image=image, pred_type="denoise")
        else:
            pred = self.model(image)

        return self.compute_loss(pred, label) 
    
    def compute_loss(self, preds, labels):
        return self.criterion(preds, labels) 
    
    def validation_step(self, batch):
        with torch.cuda.amp.autocast(self.use_amp):
            _, outputs, labels = self.infer(batch)
            
        dices = []
        for i in range(self.num_classes):
            output = outputs[:, i]
            label = labels[:, i]
            if output.sum() > 0 and label.sum() > 0:
                self.dice_metric(output, label)
                dice = self.dice_metric.aggregate()
            elif output.sum() > 0 and label.sum() == 0:
                dice = torch.Tensor([1.0]).to(outputs.device)
            
            dices.append(dice)
            
        self.dice_metric.reset()
        return torch.mean(torch.stack(dices))
    
    def validation_end(self, dices, epoch):
        dices = torch.stack(dices)
        mean_dice = torch.mean(dices)
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            if mean_dice > 0.5:
                self.save_model(model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                epoch=self.epoch,
                                save_path=os.path.join(self.weights_path, f"best_{mean_dice:.4f}.pt"))

        print(f"mean_dice : {mean_dice:.4f}")
        self.log("mean_dice", mean_dice.item(), epoch)

if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(**vars(args))
    trainer.train()
