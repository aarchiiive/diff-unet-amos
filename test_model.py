import torch
import torch.nn as nn   

from models.swin_unetr import SwinUNETRDenoiser
from models.swin_diff_unetr import SwinDiffUNETR

device = torch.device("cuda:0")

# x = torch.randn((1, 1, 96, 96, 96))
t = torch.randn((2, 1)).to(device)
image = torch.randn((1, 1, 96, 96, 96)).to(device)
label = torch.randn((1, 13, 96, 96, 96)).to(device)
embeddings = torch.randn((1, 13, 96, 96, 96)).to(device)

# model = SwinUNETRDenoiser(img_size=96, in_channels=1, out_channels=13)
model = SwinDiffUNETR(out_channels=13, feature_size=48).to(device)

with torch.no_grad():   
    x_start = (label) * 2 - 1
    x_t, t, _ = model(x=x_start, pred_type="q_sample")
    pred = model(x=x_t, step=t, image=image, pred_type="denoise")

# out = model(x, t, embeddings)