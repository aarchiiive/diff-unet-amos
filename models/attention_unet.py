from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# References : https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def conv_bn(spatial_dims: int) -> Sequence[nn.Module]:
    if spatial_dims == 2:
        return nn.Conv2d, nn.BatchNorm2d
    elif spatial_dims == 3:
        return nn.Conv3d, nn.BatchNorm3d
    else:
        raise NotImplementedError(f"spatial_dims value of {spatial_dims} is not supported. Please use ndim 2 or 3.")


class Conv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
        super(Conv,self).__init__()
        conv, batch_norm = conv_bn(spatial_dims)
        
        self.conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            batch_norm(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            batch_norm(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
        super(UpConv, self).__init__()
        conv, batch_norm = conv_bn(spatial_dims)
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            batch_norm(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float):
        super(AttentionBlock,self).__init__()
        f_g = f_l = out_channels
        f_int = out_channels // 2
        
        conv, batch_norm = conv_bn(spatial_dims)
            
        self.up = UpConv(spatial_dims, in_channels, out_channels, dropout)
        self.w_g = nn.Sequential(
            conv(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            batch_norm(f_int)
        )
        
        self.w_x = nn.Sequential(
            conv(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            batch_norm(f_int)
        )

        self.psi = nn.Sequential(
            conv(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            batch_norm(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.out = Conv(spatial_dims, in_channels, out_channels, dropout)
        
    def forward(self, x1, x2):
        g = self.up(x1)
        
        # attention
        psi = self.relu(self.w_g(g) + self.w_x(x2))
        psi = self.psi(psi)
        x = x2 * psi
        # concatenate
        x = torch.cat((x, g),dim=1)        

        return self.out(x)

class AttentionUNet(nn.Module):
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 3, 
                 out_channels: int = 1,
                 features: Sequence[int] = [64, 128, 256, 512, 1024],
                 kernel_size: int = 1,
                 stride: int = 1,
                 pool_size: int = 2,
                 dropout: float = 0.2):
        super(AttentionUNet,self).__init__()
        features = [64, 128, 256, 512, 1024]
        
        self.head = Conv(spatial_dims, in_channels, features[0], dropout)
        self.down = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.down.append(nn.Sequential(
                nn.MaxPool3d(kernel_size=pool_size, stride=pool_size),
                Conv(spatial_dims, features[i], features[i+1], dropout)
            ))
            
        features.reverse()
        self.up = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.up.append(AttentionBlock(spatial_dims, features[i], features[i+1], dropout))
        
        self.out = nn.Conv3d(features[-1], out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsampling
        embeddings = self.downsample(x)
        
        # upsampling
        x = self.upsample(embeddings)
        
        return self.out(x)
    
    def downsample(self, x: torch.Tensor) -> List[torch.Tensor]:
        _x = [self.head(x)]
        for i in range(len(self.down)):
            _x.append(self.down[i](_x[-1]))
        
        _x.reverse()
        
        return _x
    
    def upsample(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        x = None
        for i in range(len(self.up)):
            x = self.up[i](embeddings[i], embeddings[i+1]) if x is None else self.up[i](x, embeddings[i+1])
        
        return x
        
    
if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, out_channels=13)
    print(model)
    x = torch.randn((4, 1, 96, 96, 96))
    print(model(x).shape)