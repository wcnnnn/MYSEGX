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
        activation (str): 激活函数类型
    """
    def __init__(self, num_classes=20, num_queries=100, hidden_dim=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", backbone_type="resnet50"):
        super().__init__()
        
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
        
        # 投影层 - 使用动态获取的输出通道数
        self.conv = nn.Conv2d(backbone_out_channels, hidden_dim, 1)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderBlock(
            hidden_dim, nhead, dim_feedforward, dropout)
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])
        
        # Transformer解码器
        decoder_layer = TransformerDecoderBlock(
            hidden_dim, nhead, dim_feedforward, dropout)
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])
        
        # 目标查询
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 分类头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1表示背景类
        
        # 掩码头
        self.mask_head = MaskHead(backbone_out_channels, hidden_dim, num_classes, backbone_type=backbone_type)
        
        # 匈牙利匹配器
        self.matcher = HungarianAssigner()
        
    def forward(self, images):
        """前向传播
        
        参数:
            images (Tensor): 输入图像, shape (B, 3, H, W)
            
        返回:
            outputs (dict): 包含以下键值对的字典
                - pred_logits: 预测的类别logits, shape (B, N, C)
                - pred_masks: 预测的分割掩码, shape (B, N, H, W)
        """
        # 打印输入形状
        # print(f"\nDETR forward:")
        # print(f"Input images shape: {images.shape}")
        
        # 提取特征
        features = self.backbone(images)  # 返回多尺度特征 [x1, x2, x3, x4]
        for i, feat in enumerate(features):
            # print(f"Backbone feature {i} shape: {feat.shape}")
            pass
            
        # 投影最后一层特征到hidden_dim维度
        x = self.conv(features[-1])
        # print(f"Projected feature shape: {x.shape}")
        
        # 准备序列输入
        h, w = x.shape[-2:]
        x = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        # print(f"Flattened feature shape: {x.shape}")
        
        # 添加位置编码
        pos = self.pos_encoder(x)
        
        # Transformer编码器
        memory = x
        for layer in self.encoder:
            memory = layer(memory + pos)
        # print(f"Encoder output shape: {memory.shape}")
        
        # 准备目标查询
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.shape[1], 1)
        tgt = torch.zeros_like(query_embed)
        # print(f"Query embeddings shape: {query_embed.shape}")
        
        # Transformer解码器
        hs = tgt
        for layer in self.decoder:
            hs = layer(hs + query_embed, memory)
        # print(f"Decoder output shape: {hs.shape}")
        
        # 预测类别
        outputs_class = self.class_embed(hs)  # (N, B, C)
        outputs_class = outputs_class.transpose(0, 1)  # (B, N, C)
        # print(f"Class predictions shape: {outputs_class.shape}")
        
        # 预测掩码 - 使用所有尺度的特征
        outputs_mask = self.mask_head(features,
                                    hs.permute(1, 0, 2))  # (B, N, H, W)
        # print(f"Mask predictions shape: {outputs_mask.shape}")
        
        return {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask
        }