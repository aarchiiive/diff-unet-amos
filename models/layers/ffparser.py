import torch
import torch.nn as nn

class FFParser(nn.Module):
    def __init__(self, 
                 dim: int, 
                 d: int, 
                 w: int, 
                 h: int):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, d, w, h//2+1, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, D, H, W = x.shape
        assert H == W, "height and width are not equal"
        
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(3, 4), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(3, 4), norm='ortho')
        x = x.reshape(B, C, D, H, W)
        
        return x