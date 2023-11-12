import torch
from models.attention_unet.attention_unet import AttentionUNet
from models import DiffSwinUNETR


device = torch.device("cuda:0")
model = DiffSwinUNETR().to(device)
print(int(sum(p.numel() for p in model.parameters())))
x = torch.ones((1, 13, 96, 96, 96)).to(device)
x = torch.ones((1, 1, 96, 96, 96)).to(device)

with torch.no_grad():
    x_t, t, _ = model(x=x, pred_type="q_sample")
    pred = model(x=x_t, step=t, image=x, pred_type="denoise")