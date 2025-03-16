"""DeepLabV3模型实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Tuple

# 导入骨干网络
from ...nn.backbones.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from ...nn.backbones.vgg import VGG16, VGG19
from ...nn.backbones.mobilenet import create_mobilenet_v2, create_mobilenet_v3_small, create_mobilenet_v3_large

# 导入ASPP模块和分割头
from ...nn.blocks.aspp import ASPP
from ...nn.heads.deeplabv3_head import DeepLabV3Head

# 配置日志
logger = logging.getLogger(__name__)

class DeepLabV3(nn.Module):
    """DeepLabV3模型
    
    使用ASPP模块捕获多尺度上下文信息，生成语义分割掩码。
    
    参数:
        num_classes (int): 类别数量
        backbone (str): 骨干网络类型，可选 'resnet50', 'resnet101', 'vgg16', 'mobilenetv2'等
        output_stride (int): 输出步长，可选8或16
        pretrained_backbone (bool): 是否使用预训练的骨干网络
    """
    def __init__(self, num_classes=21, backbone='resnet101', output_stride=16, pretrained_backbone=True):
        super(DeepLabV3, self).__init__()
        
        self.backbone_type = backbone
        self.output_stride = output_stride
        self.num_classes = num_classes
        
        # 验证参数
        if output_stride not in [8, 16]:
            raise ValueError("output_stride必须是8或16")
            
        # 选择骨干网络
        if backbone == 'resnet50':
            # 检查ResNet50是否支持output_stride参数
            try:
                self.backbone = ResNet50(output_stride=output_stride, pretrained=pretrained_backbone)
            except TypeError:
                # 如果不支持，则使用默认参数
                self.backbone = ResNet50(pretrained=pretrained_backbone)
            backbone_channels = 2048
        elif backbone == 'resnet101':
            # 检查ResNet101是否支持output_stride参数
            try:
                self.backbone = ResNet101(output_stride=output_stride, pretrained=pretrained_backbone)
            except TypeError:
                # 如果不支持，则使用默认参数
                self.backbone = ResNet101(pretrained=pretrained_backbone)
            backbone_channels = 2048
        elif backbone == 'resnet18':
            self.backbone = ResNet18(pretrained=pretrained_backbone)
            backbone_channels = 512
        elif backbone == 'resnet34':
            self.backbone = ResNet34(pretrained=pretrained_backbone)
            backbone_channels = 512
        elif backbone == 'vgg16':
            self.backbone = VGG16(pretrained=pretrained_backbone)
            backbone_channels = 512
        elif backbone == 'vgg19':
            self.backbone = VGG19(pretrained=pretrained_backbone)
            backbone_channels = 512
        elif backbone == 'mobilenetv2':
            self.backbone = create_mobilenet_v2(pretrained=pretrained_backbone)
            backbone_channels = 1280
        elif backbone == 'mobilenetv3_small':
            self.backbone = create_mobilenet_v3_small(pretrained=pretrained_backbone)
            backbone_channels = 576
        elif backbone == 'mobilenetv3_large':
            self.backbone = create_mobilenet_v3_large(pretrained=pretrained_backbone)
            backbone_channels = 960
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
            
        # 分割头
        self.head = DeepLabV3Head(
            in_channels=backbone_channels,
            hidden_dim=256,
            num_classes=num_classes,
            backbone_type=backbone,
            output_stride=output_stride
        )
        
        # 初始化权重
        self._init_weight()
        
        logger.info(f"初始化DeepLabV3模型: backbone={backbone}, output_stride={output_stride}, num_classes={num_classes}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入图像 [B, 3, H, W]
            
        返回:
            Dict[str, Tensor]: 包含预测结果的字典
                - out (Tensor): 预测的分割掩码 [B, num_classes, H, W]
        """
        # 记录输入统计信息
        if torch.is_grad_enabled():
            logger.debug(f"DeepLabV3输入: shape={x.shape}, 范围=[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any():
                logger.warning(f"DeepLabV3输入中包含NaN值!")
                # 替换NaN值为0
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning(f"DeepLabV3输入中包含Inf值!")
                # 替换Inf值
                x = torch.nan_to_num(x, posinf=1.0, neginf=-1.0)
                
            # 只检查值范围，不进行归一化
            max_val = x.max().item()
            min_val = x.min().item()
            if max_val > 10.0 or min_val < -10.0:
                logger.warning(f"DeepLabV3输入值范围异常: [{min_val:.3f}, {max_val:.3f}]，请检查数据加载器的归一化步骤")
        
        input_shape = x.shape[-2:]
        
        # 提取特征
        features = self.backbone(x)
        
        # 统一处理不同骨干网络的输出
        if not isinstance(features, list):
            # 如果骨干网络没有返回多尺度特征，则将输出包装为列表
            features = [features]
        
        # 确保特征列表至少有一个元素
        if len(features) == 0:
            raise ValueError("骨干网络未返回任何特征")
            
        # 记录骨干网络输出的特征
        if torch.is_grad_enabled():
            logger.debug(f"DeepLabV3骨干网络输出特征数量: {len(features)}")
            for i, feat in enumerate(features):
                if feat is None:
                    logger.warning(f"特征[{i}]为None!")
                    continue
                    
                # 使用torch.tensor操作而不是item()
                feat_max = feat.max()
                feat_min = feat.min()
                logger.debug(f"  特征[{i}]: shape={feat.shape}, 范围=[{feat_min:.3f}, {feat_max:.3f}]")
                
                # 检查特征是否有异常值
                if torch.isnan(feat).any():
                    logger.warning(f"特征[{i}]中包含NaN值!")
                    # 替换NaN值为0
                    features[i] = torch.nan_to_num(feat, nan=0.0)
                if torch.isinf(feat).any():
                    logger.warning(f"特征[{i}]中包含Inf值!")
                    # 替换Inf值
                    features[i] = torch.nan_to_num(feat, posinf=100.0, neginf=-100.0)
                
                # 如果特征值范围过大，使用更稳定的归一化方法
                feat_abs_max = torch.max(torch.abs(features[i]))
                if feat_abs_max > 100.0:  # 使用更合理的阈值
                    # 使用L2归一化
                    norm = torch.norm(features[i], p=2, dim=1, keepdim=True)
                    features[i] = features[i] / (norm + 1e-7)  # 添加小的epsilon防止除零
                    logger.warning(f"特征[{i}]值过大，已使用L2归一化")
                    # 记录归一化后的范围
                    feat_max = features[i].max()
                    feat_min = features[i].min()
                    logger.debug(f"  特征[{i}]归一化后: 范围=[{feat_min:.3f}, {feat_max:.3f}]")
        
        # 应用分割头
        out = self.head(features)
        
        # 记录分割头输出
        if torch.is_grad_enabled():
            out_max = out.max()
            out_min = out.min()
            logger.debug(f"DeepLabV3分割头输出: shape={out.shape}, 范围=[{out_min:.3f}, {out_max:.3f}]")
            
            # 如果分割头输出值范围过大，进行归一化
            if out_max > 100.0 or out_min < -100.0 or torch.isnan(out).any() or torch.isinf(out).any():
                # 先处理NaN和Inf
                out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0)
                # 对每个通道分别归一化
                out = F.normalize(out, p=2, dim=1)
                logger.warning(f"分割头输出值范围过大或包含异常值，已进行归一化")
                logger.debug(f"归一化后分割头输出: 范围=[{out.min():.3f}, {out.max():.3f}]")
        
        # 确保输出与输入大小一致
        if out.shape[-2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
            
            # 记录上采样后的输出
            if torch.is_grad_enabled():
                logger.debug(f"DeepLabV3上采样后输出: shape={out.shape}, 范围=[{out.min():.3f}, {out.max():.3f}]")
        
        # 检查输出中是否有异常值
        if torch.is_grad_enabled():
            if torch.isnan(out).any():
                logger.warning("DeepLabV3输出中包含NaN值!")
                out = torch.nan_to_num(out, nan=0.0)
            if torch.isinf(out).any():
                logger.warning("DeepLabV3输出中包含Inf值!")
                out = torch.nan_to_num(out, posinf=100.0, neginf=-100.0)
            
            # 对输出进行最终的归一化
            out = F.normalize(out, p=2, dim=1)
            logger.debug(f"最终输出: 范围=[{out.min():.3f}, {out.max():.3f}]")
        
        return {'out': out}
    
    def _init_weight(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用标准的kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 添加权重限制，防止初始值过大
                with torch.no_grad():
                    m.weight.data.clamp_(-0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 