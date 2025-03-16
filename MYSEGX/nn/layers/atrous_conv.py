"""空洞卷积层模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# 配置日志
logger = logging.getLogger(__name__)

class AtrousConv2d(nn.Module):
    """空洞卷积层
    
    使用不同膨胀率的卷积来扩大感受野，同时保持特征图分辨率。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充大小
        dilation (int): 膨胀率
        bias (bool): 是否使用偏置
        norm_layer (nn.Module): 归一化层类型
        activation (nn.Module): 激活函数类型
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU, name=""):
        super(AtrousConv2d, self).__init__()
        
        # 计算膨胀卷积的填充值，确保输出尺寸不变
        if padding == 'same':
            padding = (kernel_size - 1) // 2 * dilation
        
        self.name = name or f"atrous_conv_d{dilation}"
        self.dilation = dilation
            
        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride,
                padding=padding, 
                dilation=dilation, 
                bias=bias
            )
        ]
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        if activation is not None:
            layers.append(activation(inplace=True))
            
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            Tensor: 输出特征图 [B, C', H, W]
        """
        # 记录输入统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输入: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any():
                logger.warning(f"[{self.name}] 输入中包含NaN值!")
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                x = torch.nan_to_num(x, posinf=1.0, neginf=-1.0)
            
            # 如果输入值范围过大，进行裁剪
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e4 or min_val < -1e4:
                x = torch.clamp(x, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 输入值范围过大，已裁剪到[-1e4, 1e4]")
        
        out = self.conv(x)
        
        # 检查输出是否有异常值
        if torch.is_grad_enabled():
            if torch.isnan(out).any():
                logger.warning(f"[{self.name}] 输出中包含NaN值!")
                out = torch.nan_to_num(out, nan=0.0)
            if torch.isinf(out).any():
                logger.warning(f"[{self.name}] 输出中包含Inf值!")
                out = torch.nan_to_num(out, posinf=1e4, neginf=-1e4)
            
            # 如果输出值范围过大，进行裁剪
            max_val = out.max().item()
            min_val = out.min().item()
            if max_val > 1e4 or min_val < -1e4:
                out = torch.clamp(out, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 输出值范围过大，已裁剪到[-1e4, 1e4]")
            
            logger.debug(f"[{self.name}] 输出: shape={out.shape}, 范围=[{out.min().item():.3f}, {out.max().item():.3f}]")
        
        return out


class SeparableAtrousConv2d(nn.Module):
    """深度可分离空洞卷积
    
    将标准卷积分解为深度卷积和逐点卷积，减少参数量和计算量。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充大小
        dilation (int): 膨胀率
        bias (bool): 是否使用偏置
        norm_layer (nn.Module): 归一化层类型
        activation (nn.Module): 激活函数类型
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU, name=""):
        super(SeparableAtrousConv2d, self).__init__()
        
        # 计算膨胀卷积的填充值，确保输出尺寸不变
        if padding == 'same':
            padding = (kernel_size - 1) // 2 * dilation
        
        self.name = name or f"sep_atrous_conv_d{dilation}"
        self.dilation = dilation
        
        # 深度卷积 - 每个通道单独卷积
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation,
            groups=in_channels, 
            bias=bias
        )
        
        # 逐点卷积 - 1x1卷积用于通道混合
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)]
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        if activation is not None:
            layers.append(activation(inplace=True))
            
        self.pointwise = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            Tensor: 输出特征图 [B, C', H, W]
        """
        # 记录输入统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输入: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any():
                logger.warning(f"[{self.name}] 输入中包含NaN值!")
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                x = torch.nan_to_num(x, posinf=1.0, neginf=-1.0)
            
            # 如果输入值范围过大，进行裁剪
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e4 or min_val < -1e4:
                x = torch.clamp(x, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 输入值范围过大，已裁剪到[-1e4, 1e4]")
        
        x = self.depthwise(x)
        
        # 检查深度卷积后的值
        if torch.is_grad_enabled():
            if torch.isnan(x).any():
                logger.warning(f"[{self.name}] 深度卷积后包含NaN值!")
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 深度卷积后包含Inf值!")
                x = torch.nan_to_num(x, posinf=1e4, neginf=-1e4)
            
            # 如果深度卷积后值范围过大，进行裁剪
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e4 or min_val < -1e4:
                x = torch.clamp(x, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 深度卷积后值范围过大，已裁剪到[-1e4, 1e4]")
            
            logger.debug(f"[{self.name}] 深度卷积后: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        out = self.pointwise(x)
        
        # 检查输出是否有异常值
        if torch.is_grad_enabled():
            if torch.isnan(out).any():
                logger.warning(f"[{self.name}] 输出中包含NaN值!")
                out = torch.nan_to_num(out, nan=0.0)
            if torch.isinf(out).any():
                logger.warning(f"[{self.name}] 输出中包含Inf值!")
                out = torch.nan_to_num(out, posinf=1e4, neginf=-1e4)
            
            # 如果输出值范围过大，进行裁剪
            max_val = out.max().item()
            min_val = out.min().item()
            if max_val > 1e4 or min_val < -1e4:
                out = torch.clamp(out, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 输出值范围过大，已裁剪到[-1e4, 1e4]")
            
            logger.debug(f"[{self.name}] 输出: shape={out.shape}, 范围=[{out.min().item():.3f}, {out.max().item():.3f}]")
        
        return out 