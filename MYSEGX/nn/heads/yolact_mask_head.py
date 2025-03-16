"""YOLACT Mask Head模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class YOLACTMaskHead(nn.Module):
    """YOLACT Mask Head
    
    用于预测mask系数,这些系数将与prototype masks组合生成实例掩码。
    
    参数:
        in_channels (int): 输入特征通道数
        num_protos (int): prototype的数量
        num_classes (int): 类别数量
        hidden_dim (int): 隐藏层通道数
        use_gn (bool): 是否使用GroupNorm
    """
    def __init__(self, in_channels=256, num_protos=32, num_classes=80, 
                 hidden_dim=256, use_gn=False):
        super().__init__()
        
        self.num_protos = num_protos
        self.num_classes = num_classes
        
        # 分类分支
        cls_tower = []
        for i in range(4):
            cls_tower.append(
                nn.Conv2d(in_channels if i == 0 else hidden_dim,
                         hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
            )
            if use_gn:
                cls_tower.append(nn.GroupNorm(32, hidden_dim))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        
        # mask系数分支
        mask_tower = []
        for i in range(4):
            mask_tower.append(
                nn.Conv2d(in_channels if i == 0 else hidden_dim,
                         hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
            )
            if use_gn:
                mask_tower.append(nn.GroupNorm(32, hidden_dim))
            mask_tower.append(nn.ReLU(inplace=True))
        self.mask_tower = nn.Sequential(*mask_tower)
        
        # 预测层
        self.cls_logits = nn.Conv2d(hidden_dim, num_classes, kernel_size=3, stride=1, padding=1)
        self.mask_coeff = nn.Conv2d(hidden_dim, num_protos, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(hidden_dim, 4, kernel_size=3, stride=1, padding=1)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化YOLACTMaskHead: in_channels={in_channels}, num_protos={num_protos}, num_classes={num_classes}")
        
    def forward(self, features):
        """前向传播
        
        参数:
            features (list[Tensor]): 输入特征图列表 [B, C, H, W]
            
        返回:
            dict: 包含分类分数、边界框回归和mask系数的字典
        """
        if torch.is_grad_enabled():
            for i, feat in enumerate(features):
                logger.debug(f"MaskHead输入{i}: shape={feat.shape}, 范围=[{feat.min():.3f}, {feat.max():.3f}]")
                
                # 检查输入是否有异常值
                if torch.isnan(feat).any():
                    logger.warning(f"输入特征{i}包含NaN值!")
                    features[i] = torch.nan_to_num(feat, nan=0.0)
                if torch.isinf(feat).any():
                    logger.warning(f"输入特征{i}包含Inf值!")
                    features[i] = torch.nan_to_num(feat, posinf=100.0, neginf=-100.0)
        
        cls_logits = []
        bbox_reg = []
        mask_coefs = []
        
        for feature in features:
            cls_tower_out = self.cls_tower(feature)
            mask_tower_out = self.mask_tower(feature)
            
            # 分类预测
            cls_pred = self.cls_logits(cls_tower_out)
            cls_logits.append(cls_pred)
            
            # 边界框预测
            bbox_pred = self.bbox_pred(cls_tower_out)
            bbox_reg.append(bbox_pred)
            
            # mask系数预测
            coef_pred = self.mask_coeff(mask_tower_out)
            mask_coefs.append(coef_pred)
            
        # 检查输出
        if torch.is_grad_enabled():
            for i, (cls_pred, bbox_pred, coef_pred) in enumerate(zip(cls_logits, bbox_reg, mask_coefs)):
                logger.debug(f"输出{i}:")
                logger.debug(f"  分类: shape={cls_pred.shape}, 范围=[{cls_pred.min():.3f}, {cls_pred.max():.3f}]")
                logger.debug(f"  边界框: shape={bbox_pred.shape}, 范围=[{bbox_pred.min():.3f}, {bbox_pred.max():.3f}]")
                logger.debug(f"  mask系数: shape={coef_pred.shape}, 范围=[{coef_pred.min():.3f}, {coef_pred.max():.3f}]")
                
                # 检查输出是否有异常值
                for name, pred in [("分类", cls_pred), ("边界框", bbox_pred), ("mask系数", coef_pred)]:
                    if torch.isnan(pred).any():
                        logger.warning(f"{name}预测包含NaN值!")
                        pred = torch.nan_to_num(pred, nan=0.0)
                    if torch.isinf(pred).any():
                        logger.warning(f"{name}预测包含Inf值!")
                        pred = torch.nan_to_num(pred, posinf=100.0, neginf=-100.0)
        
        return {
            "cls_logits": cls_logits,
            "bbox_reg": bbox_reg,
            "mask_coefs": mask_coefs
        }
        
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
                
        # 为分类层使用先验概率初始化
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias_value) 