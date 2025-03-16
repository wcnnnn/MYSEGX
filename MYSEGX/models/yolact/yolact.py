"""YOLACT模型实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple

from ...nn.backbones.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from ...nn.backbones.mobilenet import create_mobilenet_v2
from ...nn.blocks.protonet import ProtoNet
from ...nn.blocks.fpn import FPN
from ...nn.heads.yolact_mask_head import YOLACTMaskHead
from ...nn.modules.ops.nms import FastNMSModule

logger = logging.getLogger(__name__)

class YOLACT(nn.Module):
    """YOLACT模型
    
    将ResNet主干网络、FPN、ProtoNet和预测头组合成完整的实例分割模型。
    
    参数:
        num_classes (int): 类别数量
        backbone_type (str): 主干网络类型
        hidden_dim (int): 隐藏层维度
        num_protos (int): prototype的数量
        use_gn (bool): 是否使用GroupNorm
        top_k (int): NMS保留的最大检测数
        score_threshold (float): 分数阈值
        nms_threshold (float): NMS的IoU阈值
    """
    def __init__(self, num_classes=80, backbone_type="resnet50", hidden_dim=256,
                 num_protos=32, use_gn=False, top_k=200, score_threshold=0.05,
                 nms_threshold=0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_type = backbone_type.lower()
        self.hidden_dim = hidden_dim
        self.num_protos = num_protos
        
        # 主干网络
        backbone_mapping = {
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "mobilenetv2": create_mobilenet_v2
        }
        if backbone_type.lower() not in backbone_mapping:
            raise ValueError(f"不支持的backbone类型: {backbone_type}")
            
        self.backbone = backbone_mapping[backbone_type.lower()]()
        
        # 获取backbone的输出通道数
        if backbone_type.lower() in ["resnet50", "resnet101"]:
            in_channels = [512, 1024, 2048]  # C3, C4, C5
            backbone_channels = 2048
        elif backbone_type.lower() == "mobilenetv2":
            in_channels = [32, 96, 320]  # 根据MobileNetV2的结构调整
            backbone_channels = 320
        else:  # resnet18, resnet34
            in_channels = [128, 256, 512]  # C3, C4, C5
            backbone_channels = 512
            
        # FPN
        self.fpn = FPN(
            in_channels_list=in_channels,
            out_channels=hidden_dim,
            use_gn=use_gn
        )
        
        # ProtoNet
        self.proto_net = ProtoNet(
            in_channels=hidden_dim,
            proto_channels=hidden_dim,
            num_protos=num_protos
        )
        
        # 预测头
        self.prediction_head = YOLACTMaskHead(
            in_channels=hidden_dim,
            num_protos=num_protos,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            use_gn=use_gn
        )
        
        # NMS模块
        self.nms = FastNMSModule(
            iou_threshold=nms_threshold,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化YOLACT: backbone={backbone_type}, num_classes={num_classes}, num_protos={num_protos}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入图像 [B, 3, H, W]
            
        返回:
            dict: 包含以下键值对的字典:
                - pred_cls: 预测的类别分数 [B, N, num_classes]
                - pred_boxes: 预测的边界框 [B, N, 4]
                - pred_masks: 预测的实例掩码 [B, N, H, W]
                - proto_out: prototype masks [B, num_protos, H/4, W/4]
        """
        if torch.is_grad_enabled():
            logger.debug(f"YOLACT输入: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any():
                logger.warning("输入图像包含NaN值!")
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning("输入图像包含Inf值!")
                x = torch.nan_to_num(x, posinf=100.0, neginf=-100.0)
        
        # 提取特征
        features = self.backbone(x)  # 返回多尺度特征 [C3, C4, C5]
        
        # FPN特征融合
        fpn_features = self.fpn(features)
        
        # 生成prototype masks
        proto_out = self.proto_net(fpn_features[0])  # 使用最高分辨率的特征
        
        # 预测头
        pred_dict = self.prediction_head(fpn_features)
        cls_logits = pred_dict['cls_logits']
        bbox_reg = pred_dict['bbox_reg']
        mask_coefs = pred_dict['mask_coefs']
        
        if self.training:
            return {
                'pred_cls': cls_logits,
                'pred_boxes': bbox_reg,
                'pred_masks': mask_coefs,
                'proto_out': proto_out
            }
        else:
            # 测试时进行后处理
            results = []
            for cls_per_level, box_per_level, coef_per_level in zip(
                cls_logits, bbox_reg, mask_coefs):
                # 应用sigmoid/softmax
                cls_per_level = torch.sigmoid(cls_per_level)
                
                # 应用NMS
                keep_idx, scores, classes = self.nms(box_per_level, cls_per_level)
                
                # 获取保留的预测
                boxes = box_per_level[keep_idx]
                coeffs = coef_per_level[keep_idx]
                
                # 生成实例掩码
                masks = torch.matmul(coeffs, proto_out.view(proto_out.size(0), -1))
                masks = masks.view(-1, proto_out.size(2), proto_out.size(3))
                masks = torch.sigmoid(masks)
                
                results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'masks': masks
                })
            
            return results
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
