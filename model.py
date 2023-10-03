from models.attention_diff_unet import AttentionDiffUNet
from models.diff_unet import DiffUNet

import torch

model = AttentionDiffUNet(3, 1, 13)
# model = DiffUNet(3, 13, 1000, "train")
image = torch.randn((1, 1, 96, 96, 96))
label = torch.randn((1, 13, 96, 96, 96))
x_start = (label) * 2 - 1
x_t, t, _ = model(x=x_start, pred_type="q_sample")
pred = model(x=x_t, step=t, image=image, pred_type="denoise")

# pred = model(image, pred_type="ddim_sample")

# print(f"x_t : {x_t.shape}")
# print(f"t : {t.shape}")
print(f"pred : {pred.shape}")