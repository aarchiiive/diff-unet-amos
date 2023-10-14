import torch
import torch.nn as nn

"""
[Code References]
https://github.com/milesial/Pytorch-UNet/tree/master

"""


class Conv(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: int,
    ) -> None:
        assert 0 < spatial_dims <= 3
        if spatial_dims == 1:
            conv_type = nn.Conv1d
        elif spatial_dims == 2:
            conv_type = nn.Conv2d
        elif spatial_dims == 3:
            conv_type = nn.Conv3d
            
        self.layer = conv_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNext(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dropout: int) -> None:
        super().__init__()
        
        
        
        