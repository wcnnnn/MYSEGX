"""简单的CNN分割网络实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSegNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        """初始化CNNSegNet
        
        参数:
            n_channels (int): 输入通道数，默认为3（RGB图像）
            n_classes (int): 分割类别数，默认为2（二分类）
        """
        super(CNNSegNet, self).__init__()
        
        # 编码器
        self.enc_conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 解码器
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor): 输入图像，形状为 (B, C, H, W)
            
        返回:
            out (tensor): 分割结果，形状为 (B, n_classes, H, W)
        """
        # 编码器部分
        x1 = F.relu(self.bn1(self.enc_conv1(x)))
        x1 = F.max_pool2d(x1, 2)
        
        x2 = F.relu(self.bn2(self.enc_conv2(x1)))
        x2 = F.max_pool2d(x2, 2)
        
        x3 = F.relu(self.bn3(self.enc_conv3(x2)))
        x3 = F.max_pool2d(x3, 2)
        
        # 解码器部分（使用反卷积上采样）
        x = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn4(self.dec_conv1(x)))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn5(self.dec_conv2(x)))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dec_conv3(x)
        
        return x
        
    def get_loss(self, outputs, targets):
        """计算损失函数
        
        参数:
            outputs (tensor): 模型输出
            targets (tensor): 目标掩码
            
        返回:
            loss (tensor): 损失值
        """
        return F.cross_entropy(outputs, targets)
