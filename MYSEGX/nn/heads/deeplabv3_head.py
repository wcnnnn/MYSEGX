"""DeepLabV3分割头模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Tuple

from ..blocks.aspp import ASPP
from .semantic_mask_head import FPNBlock

# 配置日志
logger = logging.getLogger(__name__)

class DeepLabV3Head(nn.Module):
    """DeepLabV3分割头
    
    使用ASPP模块捕获多尺度上下文信息，生成语义分割掩码。
    
    参数:
        in_channels (int): 输入特征通道数
        hidden_dim (int): 隐藏层维度
        num_classes (int): 类别数量
        backbone_type (str): 骨干网络类型
        output_stride (int): 输出步长，可选8或16
    """
    def __init__(self, in_channels, hidden_dim, num_classes, backbone_type="resnet50", output_stride=16):
        super().__init__()
        
        self.backbone_type = backbone_type.lower()
        self.output_stride = output_stride
        self.num_classes = num_classes
        
        # 根据output_stride调整膨胀率
        if output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:  # output_stride == 16
            atrous_rates = [6, 12, 18]
            
        # 创建ASPP模块
        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=hidden_dim,
            atrous_rates=atrous_rates,
            dropout_rate=0.1,
            name="DeepLabV3Head_ASPP"
        )
        
        # 解码器 - 增加一个额外的卷积层以提高表达能力
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),  # 添加一个额外的卷积层
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
        
        # 初始化权重
        self._init_weight()
        
        logger.info(f"初始化DeepLabV3Head: in_channels={in_channels}, hidden_dim={hidden_dim}, num_classes={num_classes}")
        
    def forward(self, features):
        """前向传播
        
        参数:
            features (List[Tensor]): 主干网络输出的多尺度特征列表
                - features[0]: layer1输出 (1/4)
                - features[1]: layer2输出 (1/8)
                - features[2]: layer3输出 (1/16)
                - features[3]: layer4输出 (1/32)
        
        返回:
            Tensor: 预测的分割掩码 (B, C, H, W)
        """
        # 记录输入特征
        if torch.is_grad_enabled():
            logger.debug(f"DeepLabV3Head.forward - 特征列表长度: {len(features)}")
            for i, feat in enumerate(features):
                if feat is not None:
                    logger.debug(f"  特征[{i}]: shape={feat.shape}, 范围=[{feat.min().item():.3f}, {feat.max().item():.3f}]")
                    
                    # 检查特征是否有异常值
                    if torch.isnan(feat).any():
                        logger.warning(f"特征[{i}]中包含NaN值!")
                    if torch.isinf(feat).any():
                        logger.warning(f"特征[{i}]中包含Inf值!")
                    
                    # 如果特征值范围过大，进行归一化
                    max_val = feat.max().item()
                    if max_val > 1e6:
                        scale_factor = 1.0 / math.sqrt(max_val)
                        features[i] = feat * scale_factor
                        logger.warning(f"特征[{i}]值过大，已缩放 (factor={scale_factor:.6f})")
                        logger.debug(f"  特征[{i}]缩放后: 范围=[{features[i].min().item():.3f}, {features[i].max().item():.3f}]")
        
        # 使用最深层特征作为ASPP输入
        # 确保我们有有效的特征
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            raise ValueError("没有有效的特征输入")
            
        # 使用最深层特征（通常是最后一个）
        x = valid_features[-1]
        
        # 应用ASPP模块
        x = self.aspp(x)
        
        if torch.is_grad_enabled():
            logger.debug(f"ASPP输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 只在极端情况下进行裁剪
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 1e6 or min_val < -1e6 or torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.clamp(x, min=-1e6, max=1e6)
                logger.warning(f"ASPP输出值异常，已裁剪到[-1e6, 1e6]")
                logger.debug(f"裁剪后ASPP输出: 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 应用解码器
        for i, module in enumerate(self.decoder):
            x = module(x)
            
            # 在每个模块后检查数值
            if torch.is_grad_enabled() and isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if torch.isnan(x).any():
                    logger.warning(f"解码器[{i}] {module.__class__.__name__}后出现NaN值!")
                if torch.isinf(x).any():
                    logger.warning(f"解码器[{i}] {module.__class__.__name__}后出现Inf值!")
                
                # 如果值范围过大，进行裁剪
                max_val = x.max().item()
                min_val = x.min().item()
                if max_val > 1e4 or min_val < -1e4:
                    x = torch.clamp(x, min=-1e4, max=1e4)
                    logger.warning(f"解码器[{i}] {module.__class__.__name__}后值范围过大，已裁剪到[-1e4, 1e4]")
        
        if torch.is_grad_enabled():
            logger.debug(f"解码器输出: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 上采样到原始分辨率
        # 使用第一个有效特征的尺寸作为目标尺寸
        if valid_features:
            input_shape = valid_features[0].shape[-2:]
            masks = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        else:
            # 如果没有有效特征，保持原样
            masks = x
        
        if torch.is_grad_enabled():
            logger.debug(f"最终掩码: shape={masks.shape}, 范围=[{masks.min().item():.3f}, {masks.max().item():.3f}]")
        
        return masks
    
    def _init_weight(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用标准的kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 移除权重限制代码
                # with torch.no_grad():
                #     m.weight.data.clamp_(-0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabV3PlusHead(nn.Module):
    """DeepLabV3+分割头
    
    DeepLabV3+的分割头部分，包含ASPP模块和解码器部分，用于融合低层特征。
    
    参数:
        in_channels (int): 输入通道数
        hidden_dim (int): 中间特征维度
        num_classes (int): 类别数量
        backbone_type (str): 骨干网络类型
        output_stride (int): 输出步长
    """
    def __init__(self, in_channels=2048, hidden_dim=128, num_classes=21, backbone_type='resnet101', output_stride=16):
        super(DeepLabV3PlusHead, self).__init__()
        self.backbone_type = backbone_type
        self.output_stride = output_stride
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # ASPP模块
        self.aspp = ASPP(
            in_channels, 
            hidden_dim, 
            atrous_rates=[6, 12, 18] if output_stride == 16 else [12, 24, 36],
            name="DeepLabV3PlusHead_ASPP"
        )
        
        # 低层特征的通道数设置
        if 'resnet' in backbone_type.lower():
            if backbone_type.lower() in ['resnet18', 'resnet34']:
                low_level_channels = 64
            else:
                low_level_channels = 256
        elif 'vgg' in backbone_type.lower():
            low_level_channels = 64
        elif 'mobilenet' in backbone_type.lower():
            low_level_channels = 24
        else:
            low_level_channels = 64
            
        logger.info(f"初始化DeepLabV3PlusHead:")
        logger.info(f"- backbone类型: {backbone_type}")
        logger.info(f"- 输入通道数: {in_channels}")
        logger.info(f"- 低层特征通道数: {low_level_channels}")
        logger.info(f"- 隐藏层维度: {hidden_dim}")
        logger.info(f"- 类别数: {num_classes}")
            
        # 减少低层特征的通道数
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 32, 1, bias=False),  # 从48减到32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 优化解码器结构
        reduced_dim = hidden_dim // 2  # 减少中间层通道数
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 32, reduced_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, num_classes, 1)
        )
        
        # 初始化权重
        self._init_weight()
        
    def forward(self, features):
        """前向传播"""
        # 检查输入特征
        if torch.is_grad_enabled():
            logger.debug(f"DeepLabV3PlusHead输入特征数量: {len(features)}")
            for i, feat in enumerate(features):
                if feat is None:
                    logger.warning(f"特征[{i}]为None!")
                    continue
                feat_max = feat.max()
                feat_min = feat.min()
                logger.debug(f"特征[{i}]: shape={feat.shape}, 范围=[{feat_min:.3f}, {feat_max:.3f}]")
                
                # 处理异常值
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    features[i] = torch.nan_to_num(feat, nan=0.0, posinf=100.0, neginf=-100.0)
                
                # 值范围过大时进行归一化
                if torch.max(torch.abs(features[i])) > 100.0:
                    features[i] = F.normalize(features[i], p=2, dim=1)
        
        # 获取特征
        valid_features = [f for f in features if f is not None]
        if len(valid_features) < 2:
            raise ValueError(f"DeepLabV3Plus需要至少两个特征层，但只提供了{len(valid_features)}个")
        
        low_level_feat = valid_features[0]
        high_level_feat = valid_features[-1]
        
        # 应用ASPP模块
        x = self.aspp(high_level_feat)
        
        # 处理低层特征 - 先降采样再处理
        low_level_feat = F.interpolate(low_level_feat, scale_factor=0.5, mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # ASPP输出只上采样2倍，而不是完整的上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 拼接特征
        x = torch.cat((x, low_level_feat), dim=1)
        
        # 应用解码器
        x = self.decoder(x)
        
        # 最后再进行上采样到原始尺寸
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 检查输出
        if torch.is_grad_enabled():
            logger.debug(f"最终输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
                x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def _init_weight(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用标准的kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 移除权重限制代码
                # with torch.no_grad():
                #     m.weight.data.clamp_(-0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 