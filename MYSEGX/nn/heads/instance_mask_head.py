"""DETR实例分割头模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

class FPNBlock(nn.Module):
    """特征金字塔网络块
    
    用于融合不同尺度的特征。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 添加通道数转换层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        x = self.conv(x)
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip
        return x

class MHAttentionMap(nn.Module):
    """2D注意力模块，仅返回注意力softmax（不与value相乘）"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

class InstanceMaskHead(nn.Module):
    """DETR实例分割头
    
    将DETR解码器输出的特征解码为实例分割掩码。使用FPN结构融合多尺度特征。
    
    参数:
        in_channels (int): 输入特征通道数
        hidden_dim (int): 隐藏层维度
        num_classes (int): 类别数量
        backbone_type (str): 主干网络类型，用于确定FPN维度
    """
    def __init__(self, in_channels, hidden_dim, num_classes, backbone_type="resnet50"):
        super().__init__()
        
        self.backbone_type = backbone_type.lower()
        
        # FPN层 - 根据backbone类型动态调整通道数
        if backbone_type.lower() in ["resnet50", "resnet101"]:
            fpn_dims = [2048, 1024, 512, 256]  # ResNet50/101的特征维度
        elif backbone_type.lower() in ["resnet18", "resnet34"]:
            fpn_dims = [512, 256, 128, 64]  # ResNet18/34的特征维度
        elif backbone_type.lower() == "mobilenetv2":
            fpn_dims = [1280, 96, 32, 24]  # MobileNetV2的特征维度
        elif backbone_type.lower() in ["vgg16", "vgg19"]:
            fpn_dims = [512, 512, 256, 128]  # VGG16/19的特征维度
        else:
            fpn_dims = [in_channels, in_channels//2, in_channels//4, in_channels//8]
        
        print(f"[DEBUG] InstanceMaskHead初始化 - 主干网络: {backbone_type}")
        print(f"[DEBUG] FPN维度: {fpn_dims}")
        
        # 实例分割使用完整的MaskHeadSmallConv
        # 注意：mask_embeddings的通道数是hidden_dim，bbox_mask的通道数是1
        total_input_channels = hidden_dim + 1  # mask_embeddings + bbox_mask
        inter_dims = [hidden_dim, hidden_dim//2, hidden_dim//4, hidden_dim//8, hidden_dim//16]
        
        print(f"[DEBUG] 实例分割 - 输入通道数: {total_input_channels}, 中间层维度: {inter_dims}")
        
        # 主要处理路径
        self.lay1 = nn.Conv2d(total_input_channels, inter_dims[0], 3, padding=1)
        self.gn1 = nn.GroupNorm(8, inter_dims[0])
        self.lay2 = nn.Conv2d(inter_dims[0], inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(8, inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(8, inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(8, inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)
        
        # FPN适配器 - 修复通道数匹配问题
        # 将FPN特征转换到与主路径相同的维度
        if backbone_type.lower() in ["resnet50", "resnet101"]:
            # ResNet50/101使用较大的通道数
            self.adapter1 = nn.Sequential(
                nn.Conv2d(fpn_dims[0], inter_dims[1], 1),  # 2048 -> 128
                nn.GroupNorm(8, inter_dims[1]),
                nn.ReLU(inplace=True)
            )
            self.adapter2 = nn.Sequential(
                nn.Conv2d(fpn_dims[1], inter_dims[2], 1),  # 1024 -> 64
                nn.GroupNorm(8, inter_dims[2]),
                nn.ReLU(inplace=True)
            )
            self.adapter3 = nn.Sequential(
                nn.Conv2d(fpn_dims[2], inter_dims[3], 1),  # 512 -> 32
                nn.GroupNorm(8, inter_dims[3]),
                nn.ReLU(inplace=True)
            )
        else:
            # 其他主干网络使用较小的通道数配置
            self.adapter1 = nn.Sequential(
                nn.Conv2d(fpn_dims[0], inter_dims[1], 1),
                nn.GroupNorm(8, inter_dims[1]),
                nn.ReLU(inplace=True)
            )
            self.adapter2 = nn.Sequential(
                nn.Conv2d(fpn_dims[1], inter_dims[2], 1),
                nn.GroupNorm(8, inter_dims[2]),
                nn.ReLU(inplace=True)
            )
            self.adapter3 = nn.Sequential(
                nn.Conv2d(fpn_dims[2], inter_dims[3], 1),
                nn.GroupNorm(8, inter_dims[3]),
                nn.ReLU(inplace=True)
            )
        
        print("[DEBUG] 适配器通道配置:")
        if backbone_type.lower() in ["resnet50", "resnet101"]:
            print(f"  adapter1: {fpn_dims[0]} -> {inter_dims[1]}")  # 2048 -> 128
            print(f"  adapter2: {fpn_dims[1]} -> {inter_dims[2]}")  # 1024 -> 64
            print(f"  adapter3: {fpn_dims[2]} -> {inter_dims[3]}")  # 512 -> 32
        else:
            print(f"  adapter1: {fpn_dims[0]} -> {inter_dims[1]}")
            print(f"  adapter2: {fpn_dims[1]} -> {inter_dims[2]}")
            print(f"  adapter3: {fpn_dims[2]} -> {inter_dims[3]}")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, features, mask_embeddings):
        """前向传播
        
        参数:
            features (List[Tensor]): 主干网络输出的多尺度特征列表
            mask_embeddings (Tensor): 解码器输出的掩码嵌入
            
        返回:
            masks (Tensor): 预测的实例分割掩码，上采样到640x640
        """
        print(f"\n[DEBUG] InstanceMaskHead.forward - 主干网络: {self.backbone_type}")
        print(f"[DEBUG] 特征列表长度: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  特征[{i}]: shape={feat.shape}, 范围=[{feat.min():.3f}, {feat.max():.3f}]")
        
        B, N, C = mask_embeddings.shape
        print(f"[DEBUG] 掩码嵌入: shape={mask_embeddings.shape}")
        
        # 使用第一层特征的尺寸作为中间目标尺寸
        first_feat = features[0]  # 第一层特征
        target_size = (first_feat.shape[2], first_feat.shape[3])  # 160x160
        H, W = target_size
        print(f"[DEBUG] 中间目标尺寸: {target_size}, 基于第一层特征 {first_feat.shape}")
        
        # 生成边界框注意力掩码
        bbox_mask = torch.zeros((B * N, H, W), device=features[0].device)
        
        # 特征处理 - 确保维度匹配，使用内存效率更高的方式
        mask_embeddings = mask_embeddings.reshape(B * N, C)  # [B*N, C]
        mask_embeddings = mask_embeddings.unsqueeze(-1).unsqueeze(-1)  # [B*N, C, 1, 1]
        mask_embeddings = F.interpolate(mask_embeddings, size=(H, W), mode='bilinear', align_corners=False)
        print(f"[DEBUG] 处理后的掩码嵌入: shape={mask_embeddings.shape}, 范围=[{mask_embeddings.min():.3f}, {mask_embeddings.max():.3f}]")
        
        # 拼接特征并释放不需要的内存
        x = torch.cat([mask_embeddings, bbox_mask.unsqueeze(1)], 1)  # [B*N, C+1, H, W]
        print(f"[DEBUG] 拼接后的特征: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        del mask_embeddings, bbox_mask  # 释放内存
        
        # 使用FPN特征进行掩码生成
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        print(f"[DEBUG] 第1层后: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        print(f"[DEBUG] 第2层后: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        # 获取FPN特征并应用适配器
        num_fpn_levels = min(3, len(features)-1)  # 最多使用3层FPN特征
        
        for i in range(num_fpn_levels):
            # 从最后一个特征开始处理，确保特征尺寸匹配
            idx = -(i + 1)  # 从最后一个特征开始
            fpn_feat = features[idx]
            print(f"[DEBUG] 处理FPN特征[{i+1}]: 原始shape={fpn_feat.shape}")
            
            # 调整特征大小到目标尺寸
            if fpn_feat.shape[-2:] != (H, W):
                fpn_feat = F.interpolate(fpn_feat, size=(H, W), mode='bilinear', align_corners=False)
                print(f"[DEBUG] 插值后: shape={fpn_feat.shape}, 范围=[{fpn_feat.min():.3f}, {fpn_feat.max():.3f}]")
            
            # 应用适配器
            cur_fpn = getattr(self, f'adapter{i+1}')(fpn_feat)
            
            # 重复特征以匹配实例数量
            if cur_fpn.size(0) != x.size(0):
                # [B, C, H, W] -> [B*N, C, H, W]
                cur_fpn = cur_fpn.unsqueeze(1).expand(-1, N, -1, -1, -1)
                cur_fpn = cur_fpn.reshape(B*N, -1, H, W)
            
            x = x + cur_fpn
            
            # 继续主路径处理
            if i == 0:
                x = self.lay3(x)
                x = self.gn3(x)
                x = F.relu(x)
            elif i == 1:
                x = self.lay4(x)
                x = self.gn4(x)
                x = F.relu(x)
            elif i == 2:
                x = self.lay5(x)
                x = self.gn5(x)
                x = F.relu(x)
        
        # 生成最终掩码
        x = self.out_lay(x)
        x = x.view(B, N, H, W)
        
        # 上采样到固定大小 (640x640)，与语义分割头保持一致
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        print(f"[DEBUG] 最终掩码: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
        
        return x