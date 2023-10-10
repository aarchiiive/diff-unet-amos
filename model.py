import torch
from models.attention_unet import AttentionUNet

device = torch.device("cpu")
model = AttentionUNet().to(device)
print(int(sum(p.numel() for p in model.parameters())))
x = torch.ones((1, 1, 96, 96, 96)).to(device)

with torch.no_grad():
    out = model(x)