"""DETR模型模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...nn.backbones.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from ...nn.backbones.vgg import VGG16, VGG19
from ...nn.backbones.mobilenet import create_mobilenet_v2, create_mobilenet_v3_small, create_mobilenet_v3_large
from ...nn.blocks.transformer_block import TransformerEncoderBlock, TransformerDecoderBlock
from ...nn.layers.attention import PositionalEncoding
from ...nn.heads.detr_mask_head import MaskHead
from ...nn.heads.semantic_mask_head import SemanticMaskHead
from ...nn.heads.panoptic_mask_head import PanopticMaskHead
from ...nn.modules.assigners.hungarian_assigner import HungarianAssigner

class DETR(nn.Module):
    """DETR模型
    
    将ResNet主干网络、Transformer编码器和解码器、掩码头以及匈牙利匹配器组合成完整的分割模型。
    
    参数:
        num_classes (int): 类别数量
        num_queries (int): 目标查询数量
        hidden_dim (int): 隐藏层维度
        nhead (int): 注意力头数
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        dim_feedforward (int): 前馈网络维度
        dropout (float): dropout比率
        backbone_type (str): 主干网络类型
        task_type (str): 任务类型
        output_size (tuple): 输出尺寸，如果为None则保持输入尺寸
    """
    def __init__(self, num_classes=20, num_queries=20, hidden_dim=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, backbone_type="resnet50", task_type="instance", output_size=None):
        super().__init__()
        
        # 验证任务类型
        valid_task_types = ["semantic", "instance", "panoptic"]
        if task_type not in valid_task_types:
            raise ValueError(f"不支持的任务类型: {task_type}，可用选项: {valid_task_types}")
        self.task_type = task_type
        self.backbone_type = backbone_type.lower()
        
        # 主干网络
        backbone_mapping = {
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "vgg16": VGG16,
            "vgg19": VGG19,
            "mobilenetv2": create_mobilenet_v2,
            "mobilenetv3small": create_mobilenet_v3_small,
            "mobilenetv3large": create_mobilenet_v3_large
        }
        if backbone_type.lower() not in backbone_mapping:
            raise ValueError(f"不支持的backbone类型: {backbone_type}，可用选项: {list(backbone_mapping.keys())}")
            
        self.backbone = backbone_mapping[backbone_type.lower()]()
        
        # 获取backbone的输出通道数
        if backbone_type.lower() in ["resnet50", "resnet101"]:
            backbone_out_channels = 2048
        elif backbone_type.lower() in ["vgg16", "vgg19"]:
            backbone_out_channels = 512
        elif backbone_type.lower() in ["mobilenetv2", "mobilenetv3small", "mobilenetv3large"]:
            backbone_out_channels = 1280
        else:  # resnet18, resnet34
            backbone_out_channels = 512
        
        # 投影层
        self.conv = nn.Conv2d(backbone_out_channels, hidden_dim, 1)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderBlock(
            hidden_dim, nhead, dim_feedforward, dropout)
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])
        
        # 根据任务类型初始化不同组件
        if task_type in ["instance", "panoptic"]:
            # Transformer解码器 - 仅用于实例分割和全景分割
            decoder_layer = TransformerDecoderBlock(
                hidden_dim, nhead, dim_feedforward, dropout)
            self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])
            
            # 目标查询
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            
            # 分类头
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1表示背景类
            
            # 匈牙利匹配器
            self.matcher = HungarianAssigner()
        else:  # semantic
            self.decoder = None
            self.query_embed = None
            self.class_embed = None
            self.matcher = None
        
        # 掩码头 - 根据任务类型使用不同的掩码头
        if task_type == "semantic":
            self.mask_head = SemanticMaskHead(
                backbone_out_channels, 
                hidden_dim, 
                num_classes, 
                backbone_type=self.backbone_type,
                output_size=output_size
            )
        elif task_type == "instance":
            self.mask_head = MaskHead(
                backbone_out_channels, 
                hidden_dim, 
                num_classes, 
                backbone_type=self.backbone_type,
                output_size=output_size
            )
        else:  # panoptic
            self.mask_head = PanopticMaskHead(
                backbone_out_channels, 
                hidden_dim, 
                num_classes, 
                backbone_type=self.backbone_type,
                output_size=output_size
            )
        
        # 匈牙利匹配器
        self.matcher = HungarianAssigner()
        
    def forward(self, images):
        """前向传播
        
        参数:
            images (Tensor): 输入图像, shape (B, 3, H, W)
            
        返回:
            outputs (dict): 包含以下键值对的字典
                - pred_logits: 预测的类别logits, shape (B, N, C) [仅实例分割和全景分割]
                - pred_masks: 预测的分割掩码
                    - 语义分割: shape (B, C, H/4, W/4)
                    - 实例分割: shape (B, N, H/4, W/4)
                    - 全景分割: shape (B, N, H/4, W/4)
        """
        print(f"\n[DEBUG] 输入图像形状: {images.shape}")
        
        # 提取特征
        features = self.backbone(images)  # 返回多尺度特征 [x1, x2, x3, x4]
        print("\n[DEBUG] Backbone特征金字塔输出:")
        for i, feat in enumerate(features):
            print(f"Level {i+1}: shape={feat.shape}, range=[{feat.min():.3f}, {feat.max():.3f}]")
        
        # 投影最后一层特征到hidden_dim维度
        x = self.conv(features[-1])
        print(f"\n[DEBUG] 特征投影后: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
        
        # 准备序列输入
        h, w = x.shape[-2:]
        x = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        print(f"\n[DEBUG] Transformer输入序列: shape={x.shape}")
        
        # 添加位置编码
        pos = self.pos_encoder(x)
        print(f"[DEBUG] 位置编码: shape={pos.shape}, range=[{pos.min():.3f}, {pos.max():.3f}]")
        
        # Transformer编码器
        memory = x
        for i, layer in enumerate(self.encoder):
            memory = layer(memory + pos)
            print(f"[DEBUG] 编码器层{i+1}输出: shape={memory.shape}, range=[{memory.min():.3f}, {memory.max():.3f}]")
        
        # 根据任务类型处理输出
        if self.task_type == "semantic":
            print("\n[DEBUG] 执行语义分割分支")
            # 语义分割 - 直接使用编码器输出
            outputs_mask = self.mask_head(features)
            print(f"[DEBUG] 语义分割掩码输出: shape={outputs_mask.shape}")
            return {'pred_masks': outputs_mask}
            
        else:  # instance or panoptic
            print(f"\n[DEBUG] 执行{self.task_type}分割分支")
            # 准备目标查询
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.shape[1], 1)
            tgt = torch.zeros_like(query_embed)
            print(f"[DEBUG] 目标查询: shape={query_embed.shape}")
            
            # Transformer解码器
            hs = tgt
            for i, layer in enumerate(self.decoder):
                hs = layer(hs + query_embed, memory)
                print(f"[DEBUG] 解码器层{i+1}输出: shape={hs.shape}, range=[{hs.min():.3f}, {hs.max():.3f}]")
            
            # 预测类别
            outputs_class = self.class_embed(hs)  # (N, B, C)
            outputs_class = outputs_class.transpose(0, 1)  # (B, N, C)
            print(f"\n[DEBUG] 类别预测输出: shape={outputs_class.shape}")
            
            # 预测掩码 - 确保掩码尺寸正确
            outputs_mask = self.mask_head(features, hs.permute(1, 0, 2))  # 已经是 (B, N, 640, 640)
            print(f"[DEBUG] 掩码预测输出: shape={outputs_mask.shape}, range=[{outputs_mask.min():.3f}, {outputs_mask.max():.3f}]")
            
            return {
                'pred_logits': outputs_class,
                'pred_masks': outputs_mask
            }