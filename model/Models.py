from .ResNet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .senet import *
from .Module import Seg

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialGate2d(nn.Module):

    def __init__(self, in_channels):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.conv1(x)
        cal = self.sigmoid(cal)
        return cal * x

class ChannelGate2d(nn.Module):

    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate2d(channels)
        self.channel_gate = ChannelGate2d(channels, reduction=reduction)

    def  forward(self, x, z=None):
        XsSE = self.spatial_gate(x)
        XcSe = self.channel_gate(x)
        return XsSE + XcSe

class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, SE=False):
        super(CenterBlock, self).__init__()
        self.SE = SE
        self.pool = pool
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if SE:
            self.se = scSqueezeExcitationGate(out_channels)

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = self.conv_res(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.SE:
            x = self.se(x)

        x += residual
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, convT_channels, out_channels, convT_ratio=2, SE=False):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.SE = SE
        ## out = (in-1)*S + k, 如(8-1)*2 + 2 = 16
        self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)
        self.conv1 = ConvBn2d(in_channels  + convT_channels // convT_ratio, out_channels)
        self.conv2 = ConvBn2d(out_channels, out_channels)
        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

        self.conv_res = nn.Conv2d(convT_channels // convT_ratio, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip):
        x = self.convT(x)
        residual = x
        x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.SE:
            x = self.scSE(x)
        ## 这一步使得x与residual的out_channels相等，才能相加
        x += self.conv_res(residual)
        x = self.relu(x)
        return x


class UNetResNet34(Seg):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet34, self).__init__(**kwargs)

        self.resnet = resnet34(pretrained=pretrained, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder(256, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder(128, 64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder(64,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder(64,  64,  64, convT_ratio=1,  SE=True)

        # only no-empty
        self.fuse_pixel = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )


        ## 将（512， 1， 1）->(64, 1, 1)
        self.fuse_image = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True)
        )

        ## 分类
        self.logit_image = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 1))
        self.logit = nn.Sequential(
            ConvBn2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        # self.logit = nn.Sequential(
        #     ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 1, kernel_size=1, padding=0),
        # )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        batch_size = x.shape[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        c = self.center(e4)  # 8

        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)   # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)
        f = F.dropout2d(f, p=0.50)
        # for non-empty, 64*128*128
        fuse_pixel = self.fuse_pixel(f)
        # 1*128*128
        logit_pixel = self.logit_pixel(fuse_pixel)


        e = F.adaptive_avg_pool2d(e4, output_size=1).view(batch_size, -1) 
        e = F.dropout(e, p=0.50)
        # 
        fuse_image = self.fuse_image(e)   # 64*8*8
        # classify
        logit_image = self.logit_image(e).view(-1)  # *1

        fuse = torch.cat([ #fuse
            fuse_pixel,
            F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=128, mode='nearest')
        ],1)   # (64+64)*128*128
        logit = self.logit(fuse)
        # all, non-empty, logit_image
        return logit, logit_pixel, logit_image

class UnetSeNext50(Seg):
    def __init__(self, pretrained=True,  **kwargs):
        super(UnetSeNext50, self).__init__(**kwargs)
        self.resnet = se_resnext50_32x4d()
        
        self.conv1 = nn.Sequential(
            self.resnet.layer0.conv1,
            self.resnet.layer0.bn1,
            self.resnet.layer0.relu1,
        )  # 64
        
        self.convT = nn.ConvTranspose2d(64, 64 // 1, kernel_size=2, stride=2)
        
        self.encoder1 = self.resnet.layer1  # 256
        self.encoder2 = self.resnet.layer2  # 512
        self.encoder3 = self.resnet.layer3  # 1024
        self.encoder4 = self.resnet.layer4  # 2048
        
        self.center = CenterBlock(2048, 64, pool=False, SE=True)
        
        self.decoder4 = Decoder(1024, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder(512,  64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder(256,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder(64,   64,  64, convT_ratio=1,  SE=True)
        
        # only no-empty
        self.fuse_pixel = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )


        ## 将（512， 1， 1）->(64, 1, 1)
        self.fuse_image = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True)
        )

        ## 分类
        self.logit_image = nn.Sequential(
            nn.Linear(2048, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 1))
        self.logit = nn.Sequential(
            ConvBn2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
        
        x = self.conv1(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        c = self.center(e4)  # 8
        
        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        x = self.convT(x)
        d1 = self.decoder1(d2, x)   # 128
        
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)
        f = F.dropout2d(f, p=0.50)
        
        # for non-empty, 64*128*128
        fuse_pixel = self.fuse_pixel(f)
        # 1*128*128
        logit_pixel = self.logit_pixel(fuse_pixel)
        
        # 2048*1*1
        e = F.adaptive_avg_pool2d(e4, output_size=1).view(batch_size, -1) 
        e = F.dropout(e, p=0.50)
        # 
        fuse_image = self.fuse_image(e)   # 64*8*8
        # classify
        logit_image = self.logit_image(e).view(-1)  # *1
        
        fuse = torch.cat([ #fuse
            fuse_pixel,
            F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=128, mode='nearest')
        ],1)   # (64+64)*128*128
        
        logit = self.logit(fuse)
        return logit, logit_pixel, logit_image

class UnetSeNext50_salt(Seg):
    def __init__(self, pretrained=True,  **kwargs):
        super(UnetSeNext50, self).__init__(**kwargs)
        self.resnet = se_resnext50_32x4d()
        
        self.conv1 = nn.Sequential(
            self.resnet.layer0.conv1,
            self.resnet.layer0.bn1,
            self.resnet.layer0.relu1,
        )  # 64
        
        self.convT = nn.ConvTranspose2d(64, 64 // 1, kernel_size=2, stride=2)
        
        self.encoder1 = self.resnet.layer1  # 256
        self.encoder2 = self.resnet.layer2  # 512
        self.encoder3 = self.resnet.layer3  # 1024
        self.encoder4 = self.resnet.layer4  # 2048
        
        self.center = CenterBlock(2048, 64, pool=False, SE=True)
        
        self.decoder4 = Decoder(1024, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder(512,  64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder(256,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder(64,   64,  64, convT_ratio=1,  SE=True)
        
        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
        
        x = self.conv1(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        c = self.center(e4)  # 8
        
        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        x = self.convT(x)
        d1 = self.decoder1(d2, x)   # 128
        
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)

        f = F.dropout2d(f, p=0.50)
        
        logit = self.logit(f)
        return logit
##########################################################################
##########################################################################