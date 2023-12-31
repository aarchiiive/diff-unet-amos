import torch 
import torch.nn as nn 
import torch.nn.functional as F


class SmoothLayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 spatial_size: int = 96,
                 width: int = 256,
                 height: int = 256,
                 ndims: int = 5, 
                 p: int = 1, 
                 k: float = 1.0) -> None:
        super().__init__()
        assert ndims == 4 or ndims == 5
        if ndims == 4: # (B, D, W, H) or (B, D, H, W)
            self.dims = (1, 2, 3)
        if ndims == 5: # (B, C, D, W, H) or (B, C, D, H, W)
            self.dims = (2, 3, 4)
            
        # for short code
        w, h = width, height
        d = spatial_size
        
        self.p = p
        self.k = k
        self.padding = (p,) * len(self.dims) * 2
        self.shifts = [(p, 0, 0),
                       (-p, 0, 0),
                       (0, p, 0),
                       (0, -p, 0),
                       (0, 0, p),
                       (0, 0, -p)]
        self.weights = nn.Parameter(torch.randn(in_features, d, w, h) * 0.5)
        
    def forward(self, x: torch.Tensor):
        p = self.p
        _x = F.pad(x.clone(), self.padding, "constant", 0)
        laplacian = -6 * _x
        for shift in self.shifts:
            laplacian = laplacian + torch.roll(_x, shifts=shift, dims=self.dims)
        
        laplacian = laplacian[..., p:-p, p:-p, p:-p] * self.weights
        x = x + laplacian
        
        return x
    
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

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        assert H == W, "height and width are not equal"
        
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(3, 4), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(3, 4), norm='ortho')
        x = x.reshape(B, C, D, H, W)
        
        return x