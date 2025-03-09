"""UNet模型实现"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """双卷积块
    
    包含两个3x3卷积层，每个卷积后跟BN和ReLU
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样块
    
    包含最大池化和双卷积
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块
    
    包含转置卷积/上采样和双卷积
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        bilinear: 是否使用双线性插值上采样
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理输入尺寸不匹配的情况
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                  diff_y // 2, diff_y - diff_y // 2])
        
        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层
    
    1x1卷积，将通道数映射到类别数
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数（类别数）
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet模型
    
    参数:
        n_channels: 输入通道数
        n_classes: 类别数
        bilinear: 是否使用双线性插值上采样
    """
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 初始双卷积
        self.inc = DoubleConv(n_channels, 64)
        
        # 下采样路径
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 上采样路径
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出层
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出层
        logits = self.outc(x)
        
        return logits