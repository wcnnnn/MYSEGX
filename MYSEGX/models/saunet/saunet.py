"""SA-UNet模型实现
这是一个基于UNet架构的改进版本，加入了空间注意力机制。
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class DropBlock(nn.Module):
    """DropBlock正则化层
    
    参数:
        block_size (int): 丢弃块的大小
        p (float): 丢弃概率
    """
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class DoubleConv(nn.Sequential):
    """双卷积块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        mid_channels: 中间通道数（可选）
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, use_dropblock=True):
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropblock:
            layers.insert(4, DropBlock(7, 0.1))
        super(DoubleConv, self).__init__(*layers)


class Down(nn.Sequential):
    """下采样块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Last_Down(nn.Sequential):
    """最后的下采样块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super(Last_Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.1),  # 降低DropBlock概率
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    """上采样块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        bilinear: 是否使用双线性插值上采样
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels * 3 // 2, out_channels, use_dropblock=False)
        else:  
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_dropblock=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    """输出卷积层
    
    参数:
        in_channels: 输入通道数
        num_classes: 输出类别数
    """
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Attention(nn.Module):
    """空间注意力模块
    
    使用平均池化和最大池化的特征来计算空间注意力图
    """
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, 1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x = x4 * x
        return x


class SA_UNet(nn.Module):
    """SA-UNet模型
    
    一个带有空间注意力机制的UNet变体
    
    参数:
        n_channels (int): 输入通道数
        n_classes (int): 类别数
        bilinear (bool): 是否使用双线性插值上采样
        base_c (int): 基础通道数
    """
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 1,
                 bilinear: bool = False,
                 base_c: int = 32):
        super(SA_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Last_Down(base_c * 8, base_c * 16)

        self.attn = Attention()
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_c * 16, base_c * 16, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.1),
            nn.BatchNorm2d(base_c * 16),
            nn.ReLU(inplace=True)
        )

        self.up1 = Up(base_c * 16, base_c * 8, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.out_conv = OutConv(base_c, n_classes)

    def forward(self, x):
        # 输入信息调试
        print(f"\n[DEBUG] 输入图像形状: {x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        # 编码器路径
        x1 = self.conv1(x)
        print(f"\n[DEBUG] 初始特征提取:")
        print(f"初始双卷积输出: shape={x1.shape}, 范围=[{x1.min():.3f}, {x1.max():.3f}]")
        
        # 下采样路径
        print(f"\n[DEBUG] 下采样路径输出:")
        x2 = self.down1(x1)
        print(f"down1输出: shape={x2.shape}, 范围=[{x2.min():.3f}, {x2.max():.3f}]")
        
        x3 = self.down2(x2)
        print(f"down2输出: shape={x3.shape}, 范围=[{x3.min():.3f}, {x3.max():.3f}]")
        
        x4 = self.down3(x3)
        print(f"down3输出: shape={x4.shape}, 范围=[{x4.min():.3f}, {x4.max():.3f}]")
        
        x5 = self.down4(x4)
        print(f"down4输出: shape={x5.shape}, 范围=[{x5.min():.3f}, {x5.max():.3f}]")
        
        # 注意力模块
        print(f"\n[DEBUG] 注意力模块输出:")
        x6 = self.attn(x5)
        print(f"注意力输出: shape={x6.shape}, 范围=[{x6.min():.3f}, {x6.max():.3f}]")
        
        x7 = self.conv2(x6)
        print(f"注意力后卷积: shape={x7.shape}, 范围=[{x7.min():.3f}, {x7.max():.3f}]")
        
        # 上采样路径
        print(f"\n[DEBUG] 上采样路径输出:")
        x = self.up1(x7, x4)
        print(f"up1输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        x = self.up2(x, x3)
        print(f"up2输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        x = self.up3(x, x2)
        print(f"up3输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        x = self.up4(x, x1)
        print(f"up4输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        # 输出层
        logits = self.out_conv(x)
        print(f"\n[DEBUG] 最终输出:")
        print(f"logits: shape={logits.shape}, 范围=[{logits.min():.3f}, {logits.max():.3f}]")

        if not self.training:
            # 在推理模式下，返回softmax结果
            outputs = torch.softmax(logits, dim=1)
            print(f"softmax后: shape={outputs.shape}, 范围=[{outputs.min():.3f}, {outputs.max():.3f}]")
            return outputs
            
        # 在训练模式下，直接返回logits
        return logits

