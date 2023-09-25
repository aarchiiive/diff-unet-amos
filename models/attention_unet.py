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

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, ndim=5):
        super(Conv,self).__init__()
        if ndim == 4:
            _conv = nn.Conv2d
            _batch_norm = nn.BatchNorm2d
        elif ndim == 5:
            _conv = nn.Conv3d
            _batch_norm = nn.BatchNorm3d
        else:
            raise NotImplementedError(f"ndim value of {ndim} is not supported. Please use ndim 4 or 5.")

        self.conv = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            _batch_norm(out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            _batch_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, ndim=5):
        super(UpConv, self).__init__()
        if ndim == 4:
            _conv = nn.Conv2d
            _batch_norm = nn.BatchNorm2d
        elif ndim == 5:
            _conv = nn.Conv3d
            _batch_norm = nn.BatchNorm3d
        else:
            raise NotImplementedError(f"ndim value of {ndim} is not supported. Please use ndim 4 or 5.")
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            _batch_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1
        
class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCNNBlock, self).__init__()
        self.rcnn = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t)
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.conv(x)
        x1 = self.rcnn(x)
        return x + x1


class SingleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SingleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ndim=5):
        super(AttentionBlock,self).__init__()
        F_g = F_l = out_channels
        F_int = out_channels // 2
        
        if ndim == 4:
            _conv = nn.Conv2d
            _batch_norm = nn.BatchNorm2d
        elif ndim == 5:
            _conv = nn.Conv3d
            _batch_norm = nn.BatchNorm3d
        
        self.up = UpConv(in_channels=in_channels, out_channels=out_channels)
        self.wg = nn.Sequential(
            _conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            _batch_norm(F_int)
            )
        
        self.wx = nn.Sequential(
            _conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            _batch_norm(F_int)
        )

        self.psi = nn.Sequential(
            _conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            _batch_norm(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.out = Conv(in_channels=in_channels, out_channels=out_channels)
        
    def forward(self, x1, x2):
        g = self.up(x1)
        
        # attention
        psi = self.relu(self.wg(g) + self.wx(x2))
        psi = self.psi(psi)
        x = x2 * psi
        # concatenate
        x = torch.cat((x, g),dim=1)        

        return self.out(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, ndim=5):
        super(AttentionUNet,self).__init__()
        features = [64, 128, 256, 512, 1024]
        
        self.input_layer = Conv(in_channels, features[0], ndim)
        self.down = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.down.append(nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                Conv(features[i], features[i+1])
            ))
            
        features.reverse()
        self.up = nn.ModuleList()
        
        for i in range(len(features[:-1])):
            self.up.append(AttentionBlock(features[i], features[i+1], ndim))
        
        if ndim == 4:
            self.output_layer = nn.Conv2d(features[-1], out_channels, kernel_size=1, stride=1, padding=0)
        elif ndim == 5:
            self.output_layer = nn.Conv3d(features[-1], out_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x):
        # downsampling
        _x = [self.input_layer(x)]
        for i in range(len(self.down)):
            _x.append(self.down[i](_x[-1]))
        
        _x.reverse()
        
        # upsampling
        d = None
        for i in range(len(self.up)):
            print(_x[i].shape)
            if d is None:
                d = self.up[i](_x[i], _x[i+1])
            else:
                d = self.up[i](d, _x[i+1])
        
        return self.output_layer(d)
    
    
# class R2AttU_Net(nn.Module):
#     def __init__(self,in_channels=3,out_channels=1,t=2):
#         super(R2AttU_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(in_channels=in_channels,out_channels=64,t=t)

#         self.RRCNN2 = RRCNN_block(in_channels=64,out_channels=128,t=t)
        
#         self.RRCNN3 = RRCNN_block(in_channels=128,out_channels=256,t=t)
        
#         self.RRCNN4 = RRCNN_block(in_channels=256,out_channels=512,t=t)
        
#         self.RRCNN5 = RRCNN_block(in_channels=512,out_channels=1024,t=t)
        

#         self.Up5 = up_conv(in_channels=1024,out_channels=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_RRCNN5 = RRCNN_block(in_channels=1024, out_channels=512,t=t)
        
#         self.Up4 = up_conv(in_channels=512,out_channels=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_RRCNN4 = RRCNN_block(in_channels=512, out_channels=256,t=t)
        
#         self.Up3 = up_conv(in_channels=256,out_channels=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_RRCNN3 = RRCNN_block(in_channels=256, out_channels=128,t=t)
        
#         self.Up2 = up_conv(in_channels=128,out_channels=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_RRCNN2 = RRCNN_block(in_channels=128, out_channels=64,t=t)

#         self.Conv_1x1 = nn.Conv2d(64,out_channels,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.RRCNN1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.RRCNN2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.RRCNN3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.RRCNN4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.RRCNN5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_RRCNN5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4,x=x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_RRCNN4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3,x=x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_RRCNN3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2,x=x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_RRCNN2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1