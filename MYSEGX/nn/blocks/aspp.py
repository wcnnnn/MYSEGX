"""空洞空间金字塔池化(ASPP)模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import List, Tuple, Dict, Optional
from ..layers.atrous_conv import AtrousConv2d, SeparableAtrousConv2d

# 配置日志
logger = logging.getLogger(__name__)

class ASPPConv(nn.Sequential):
    """ASPP卷积模块
    
    使用不同膨胀率的卷积捕获多尺度上下文信息。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        dilation (int): 膨胀率
        use_separable (bool): 是否使用深度可分离卷积
    """
    def __init__(self, in_channels, out_channels, dilation, use_separable=False, name=None):
        self.name = name or f"ASPPConv_d{dilation}"
        
        if use_separable:
            conv = SeparableAtrousConv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation,
                bias=False,
                name=f"{self.name}_sep"
            )
        else:
            conv = AtrousConv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation,
                bias=False,
                name=f"{self.name}_std"
            )
            
        modules = [
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)
        
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
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                
            # 如果输入值范围过大，进行归一化
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e4 or min_val < -1e4:
                # 使用更稳定的缩放方法
                if max_val != min_val:
                    scale_factor = 1.0 / max(1.0, max_val)  # 确保缩放因子不会太小
                    x = x * scale_factor
                    logger.warning(f"[{self.name}] 输入值过大，已缩放 (factor={scale_factor:.6f})")
                else:
                    # 如果所有值相等且过大，重置为较小值
                    x = torch.ones_like(x) * 0.1
                    logger.warning(f"[{self.name}] 输入值全部相等且过大，已重置为0.1")
                logger.debug(f"[{self.name}] 缩放后输入: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 应用卷积操作
        for module in self:
            x = module(x)
            
            # 在每个模块后检查数值
            if torch.is_grad_enabled() and isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if torch.isnan(x).any():
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后出现NaN值!")
                if torch.isinf(x).any():
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后出现Inf值!")
                
                # 如果值范围过大，进行裁剪
                max_val = x.max().item()
                min_val = x.min().item()
                if max_val > 1e4 or min_val < -1e4:
                    x = torch.clamp(x, min=-1e4, max=1e4)
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后值范围过大，已裁剪到[-1e4, 1e4]")
        
        # 记录输出统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        return x


class ASPPPooling(nn.Sequential):
    """ASPP池化模块
    
    使用全局平均池化捕获全局上下文信息。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    def __init__(self, in_channels, out_channels, name=None):
        self.name = name or "ASPPPooling"
        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPPooling, self).__init__(*modules)
        
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
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                
            # 如果输入值范围过大，进行归一化
            max_val = x.max().item()
            if max_val > 1e4:
                scale_factor = 1.0 / math.sqrt(max_val)
                x = x * scale_factor
                logger.warning(f"[{self.name}] 输入值过大，已缩放 (factor={scale_factor:.6f})")
                logger.debug(f"[{self.name}] 缩放后输入: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        size = x.shape[-2:]
        
        # 应用池化和卷积操作
        for i, module in enumerate(self):
            x = module(x)
            
            # 在每个模块后检查数值
            if torch.is_grad_enabled() and i > 0 and isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if torch.isnan(x).any():
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后出现NaN值!")
                if torch.isinf(x).any():
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后出现Inf值!")
                
                # 如果值范围过大，进行裁剪
                max_val = x.max().item()
                min_val = x.min().item()
                if max_val > 1e4 or min_val < -1e4:
                    x = torch.clamp(x, min=-1e4, max=1e4)
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后值范围过大，已裁剪到[-1e4, 1e4]")
        
        # 上采样到原始大小
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        # 记录输出统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        return x


class ASPP(nn.Module):
    """空洞空间金字塔池化模块
    
    使用不同膨胀率的卷积和全局池化捕获多尺度上下文信息。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        atrous_rates (List[int]): 膨胀率列表
        use_separable (bool): 是否使用深度可分离卷积
        dropout_rate (float): Dropout比率
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18], 
                 use_separable=False, dropout_rate=0.5, name=None):
        super(ASPP, self).__init__()
        
        self.name = name or "ASPP"
        
        # 根据output_stride调整膨胀率
        if isinstance(atrous_rates, int):
            output_stride = atrous_rates
            if output_stride == 8:
                atrous_rates = [12, 24, 36]
            else:  # output_stride == 16 或其他值
                atrous_rates = [6, 12, 18]
            logger.info(f"ASPP: 根据output_stride={output_stride}设置膨胀率为{atrous_rates}")
        
        modules = []
        
        # 1x1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 多尺度空洞卷积
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, use_separable, name=f"{self.name}_conv{rate}"))
        
        # 全局池化分支
        modules.append(ASPPPooling(in_channels, out_channels, name=f"{self.name}_pool"))
        
        self.convs = nn.ModuleList(modules)
        
        # 投影层
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 初始化权重
        self._init_weight()
        
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
                # 替换NaN值为0
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                # 替换Inf值
                x = torch.nan_to_num(x, posinf=1.0, neginf=-1.0)
                
            # 如果输入值范围过大，进行归一化
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e4 or min_val < -1e4:
                # 使用更稳定的缩放方法
                if max_val != min_val:
                    scale_factor = 1.0 / max(1.0, max_val)  # 确保缩放因子不会太小
                    x = x * scale_factor
                    logger.warning(f"[{self.name}] 输入值过大，已缩放 (factor={scale_factor:.6f})")
                else:
                    # 如果所有值相等且过大，重置为较小值
                    x = torch.ones_like(x) * 0.1
                    logger.warning(f"[{self.name}] 输入值全部相等且过大，已重置为0.1")
                logger.debug(f"[{self.name}] 缩放后输入: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 应用各个分支
        res = []
        for conv in self.convs:
            branch_output = conv(x)
            
            # 检查分支输出是否有异常值
            if torch.is_grad_enabled():
                if torch.isnan(branch_output).any():
                    logger.warning(f"[{self.name}] 分支输出中包含NaN值!")
                    branch_output = torch.nan_to_num(branch_output, nan=0.0)
                if torch.isinf(branch_output).any():
                    logger.warning(f"[{self.name}] 分支输出中包含Inf值!")
                    branch_output = torch.nan_to_num(branch_output, posinf=1.0, neginf=-1.0)
                
                # 如果分支输出值范围过大，进行裁剪
                max_val = branch_output.max().item()
                min_val = branch_output.min().item()
                if max_val > 1e4 or min_val < -1e4:
                    branch_output = torch.clamp(branch_output, min=-1e4, max=1e4)
                    logger.warning(f"[{self.name}] 分支输出值范围过大，已裁剪到[-1e4, 1e4]")
            
            res.append(branch_output)
        
        # 拼接各个分支的输出
        res = torch.cat(res, dim=1)
        
        # 记录拼接后的统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 拼接后: shape={res.shape}, 范围=[{res.min().item():.3f}, {res.max().item():.3f}]")
            
            # 处理异常值
            if torch.isnan(res).any() or torch.isinf(res).any():
                res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
                logger.warning(f"[{self.name}] 拼接后包含异常值，已替换")
            
            # 如果拼接后的值范围过大，进行裁剪
            max_val = res.max().item()
            min_val = res.min().item()
            if max_val > 1e6 or min_val < -1e6:
                res = torch.clamp(res, min=-1e6, max=1e6)
                logger.warning(f"[{self.name}] 拼接后值范围异常，已裁剪到[-1e6, 1e6]")
                logger.debug(f"[{self.name}] 裁剪后: 范围=[{res.min().item():.3f}, {res.max().item():.3f}]")
        
        # 应用投影层
        res = self.project(res)
        
        # 记录输出统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输出: shape={res.shape}, 范围=[{res.min().item():.3f}, {res.max().item():.3f}]")
            
            # 检查输出是否有异常值
            if torch.isnan(res).any():
                logger.warning(f"[{self.name}] 输出中包含NaN值!")
                res = torch.nan_to_num(res, nan=0.0)
            if torch.isinf(res).any():
                logger.warning(f"[{self.name}] 输出中包含Inf值!")
                res = torch.nan_to_num(res, posinf=1e4, neginf=-1e4)
            
            # 最终输出值范围检查
            max_val = res.max().item()
            min_val = res.min().item()
            if max_val > 1e4 or min_val < -1e4:
                res = torch.clamp(res, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 最终输出值范围过大，已裁剪到[-1e4, 1e4]")
        
        return res
    
    def _init_weight(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用标准的kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Sequential):
    """DeepLabV3分割头
    
    使用ASPP模块捕获多尺度上下文信息，生成语义分割掩码。
    
    参数:
        in_channels (int): 输入通道数
        num_classes (int): 类别数量
        hidden_dim (int): 隐藏层维度
    """
    def __init__(self, in_channels, num_classes, hidden_dim=256, name=None):
        self.name = name or "DeepLabHead"
        modules = [
            ASPP(in_channels, hidden_dim, name=f"{self.name}_aspp"),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        ]
        super(DeepLabHead, self).__init__(*modules)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            Tensor: 输出特征图 [B, num_classes, H, W]
        """
        # 记录输入统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输入: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any():
                logger.warning(f"[{self.name}] 输入中包含NaN值!")
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                
            # 如果输入值范围过大，进行归一化
            max_val = x.max().item()
            if max_val > 1e4:
                scale_factor = 1.0 / math.sqrt(max_val)
                x = x * scale_factor
                logger.warning(f"[{self.name}] 输入值过大，已缩放 (factor={scale_factor:.6f})")
                logger.debug(f"[{self.name}] 缩放后输入: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 应用各个模块
        for i, module in enumerate(self):
            x = module(x)
            
            # 在每个模块后检查数值
            if torch.is_grad_enabled() and i > 0 and isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if torch.isnan(x).any():
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后出现NaN值!")
                if torch.isinf(x).any():
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后出现Inf值!")
                
                # 如果值范围过大，进行裁剪
                max_val = x.max().item()
                min_val = x.min().item()
                if max_val > 1e4 or min_val < -1e4:
                    x = torch.clamp(x, min=-1e4, max=1e4)
                    logger.warning(f"[{self.name}] {module.__class__.__name__}后值范围过大，已裁剪到[-1e4, 1e4]")
        
        # 记录输出统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        return x


class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+解码器
    
    融合低层特征和ASPP输出，生成更精细的分割掩码。
    
    参数:
        in_channels (int): 输入通道数
        low_level_channels (int): 低层特征通道数
        num_classes (int): 类别数量
        low_level_channels_proj (int): 低层特征投影通道数
        hidden_dim (int): 隐藏层维度
    """
    def __init__(self, in_channels, low_level_channels, num_classes, 
                 low_level_channels_proj=48, hidden_dim=256, name=None):
        super(DeepLabV3PlusDecoder, self).__init__()
        
        self.name = name or "DeepLabV3PlusDecoder"
        
        # 低层特征投影
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_channels_proj, 1, bias=False),
            nn.BatchNorm2d(low_level_channels_proj),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels + low_level_channels_proj, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
        
        # 初始化权重
        self._init_weight()
        
    def forward(self, x, low_level_feat):
        """前向传播
        
        参数:
            x (Tensor): ASPP输出特征图 [B, C, H, W]
            low_level_feat (Tensor): 低层特征 [B, C_low, H_low, W_low]
            
        返回:
            Tensor: 输出特征图 [B, num_classes, H_low, W_low]
        """
        # 记录输入统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] ASPP输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            logger.debug(f"[{self.name}] 低层特征: shape={low_level_feat.shape}, 范围=[{low_level_feat.min().item():.3f}, {low_level_feat.max().item():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any() or torch.isnan(low_level_feat).any():
                logger.warning(f"[{self.name}] 输入中包含NaN值!")
            if torch.isinf(x).any() or torch.isinf(low_level_feat).any():
                logger.warning(f"[{self.name}] 输入中包含Inf值!")
                
            # 如果输入值范围过大，进行归一化
            max_val_x = x.max().item()
            if max_val_x > 1e4:
                scale_factor = 1.0 / math.sqrt(max_val_x)
                x = x * scale_factor
                logger.warning(f"[{self.name}] ASPP输出值过大，已缩放 (factor={scale_factor:.6f})")
                logger.debug(f"[{self.name}] 缩放后ASPP输出: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
                
            max_val_low = low_level_feat.max().item()
            if max_val_low > 1e4:
                scale_factor = 1.0 / math.sqrt(max_val_low)
                low_level_feat = low_level_feat * scale_factor
                logger.warning(f"[{self.name}] 低层特征值过大，已缩放 (factor={scale_factor:.6f})")
                logger.debug(f"[{self.name}] 缩放后低层特征: 范围=[{low_level_feat.min().item():.3f}, {low_level_feat.max().item():.3f}]")
        
        # 上采样ASPP输出
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 上采样后ASPP输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 投影低层特征
        low_level_feat = self.low_level_projection(low_level_feat)
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 投影后低层特征: shape={low_level_feat.shape}, 范围=[{low_level_feat.min().item():.3f}, {low_level_feat.max().item():.3f}]")
        
        # 拼接特征
        x = torch.cat([x, low_level_feat], dim=1)
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 拼接后特征: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 只在极端情况下进行裁剪
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e6 or min_val < -1e6 or torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.clamp(x, min=-1e6, max=1e6)
                logger.warning(f"[{self.name}] 拼接后值范围异常，已裁剪到[-1e6, 1e6]")
                logger.debug(f"[{self.name}] 裁剪后: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 解码
        x = self.decoder(x)
        
        # 记录输出统计信息
        if torch.is_grad_enabled():
            logger.debug(f"[{self.name}] 解码器输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 检查输出是否有异常值
            if torch.isnan(x).any():
                logger.warning(f"[{self.name}] 输出中包含NaN值!")
            if torch.isinf(x).any():
                logger.warning(f"[{self.name}] 输出中包含Inf值!")
                
            # 如果输出值范围过大，进行裁剪
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e4 or min_val < -1e4:
                x = torch.clamp(x, min=-1e4, max=1e4)
                logger.warning(f"[{self.name}] 输出值范围过大，已裁剪到[-1e4, 1e4]")
                logger.debug(f"[{self.name}] 裁剪后输出: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        return x
    
    def _init_weight(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用标准的kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 