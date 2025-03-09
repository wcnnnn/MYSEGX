"""DETR分割头模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNBlock(nn.Module):
    """特征金字塔网络块
    
    用于融合不同尺度的特征。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x, skip=None):
        x = self.conv(x)
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip
        return x

class MaskHead(nn.Module):
    """DETR分割头
    
    将DETR解码器输出的特征解码为分割掩码。使用FPN结构融合多尺度特征。
    
    参数:
        in_channels (int): 输入特征通道数
        hidden_dim (int): 隐藏层维度
        num_classes (int): 类别数量
    """
    def __init__(self, in_channels, hidden_dim, num_classes, backbone_type="resnet50"):
        super().__init__()
        
        # FPN层 - 根据backbone类型动态调整通道数
        if in_channels == 2048:  # ResNet50/101
            fpn_channels = [in_channels, in_channels//2, in_channels//4, in_channels//8]
        elif in_channels == 1280:  # MobileNetV2
            fpn_channels = [1280, 96, 32, 24]  # MobileNetV2的特征维度
        elif in_channels == 512 and backbone_type.lower() in ["resnet18", "resnet34"]:  # ResNet18/34
            fpn_channels = [512, 256, 128, 64]  # ResNet18/34的特征维度
        elif in_channels == 512:  # VGG16/19
            fpn_channels = [512, 512, 256, 128]  # VGG16/19的特征维度
        else:  # 其他backbone
            fpn_channels = [in_channels, in_channels//2, in_channels//4, in_channels//8]
            
        self.fpn_blocks = nn.ModuleList([
            FPNBlock(fpn_channels[0], hidden_dim),  # layer4 (1/32)
            FPNBlock(fpn_channels[1], hidden_dim),  # layer3 (1/16)
            FPNBlock(fpn_channels[2], hidden_dim),  # layer2 (1/8)
            FPNBlock(fpn_channels[3], hidden_dim)   # layer1 (1/4)
        ])
        
        # 特征解码器
        self.decoder = nn.Sequential(
            # 初始特征处理
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            
            # 最后上采样到原始分辨率 (x4)
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 掩码预测器
        self.predictor = nn.Conv2d(hidden_dim, 1, 1)
        
    def forward(self, features, mask_embeddings):
        """前向传播
        
        参数:
            features (List[Tensor]): 主干网络输出的多尺度特征列表
                - features[0]: layer1输出 (1/4)
                - features[1]: layer2输出 (1/8)
                - features[2]: layer3输出 (1/16)
                - features[3]: layer4输出 (1/32)
            mask_embeddings (Tensor): 解码器输出的掩码嵌入, shape (B, N, C)
            
        返回:
            masks (Tensor): 预测的分割掩码, shape (B, N, H, W)
        """
        # 打印输入形状
        # print(f"\nMaskHead forward:")
        # print(f"Number of feature maps: {len(features)}")
        for i, feat in enumerate(features):
            # print(f"Feature {i} shape: {feat.shape}")
            pass
        # print(f"Mask embeddings shape: {mask_embeddings.shape}")
        
        # 自顶向下的特征融合
        prev = None
        for i, (feat, fpn) in enumerate(zip(reversed(features), self.fpn_blocks)):
            prev = fpn(feat, prev)
            # print(f"After FPN block {i}, shape: {prev.shape}")
        
        # 解码特征
        features = self.decoder(prev)
        # print(f"Decoded features shape: {features.shape}")
        
        # 为每个查询生成掩码
        B, N, C = mask_embeddings.shape
        H, W = features.shape[-2:]
        
        # 展开特征以进行批处理矩阵乘法
        features = features.view(B, -1, H*W)  # (B, C, HW)
        
        # 计算注意力分数
        attention = torch.bmm(mask_embeddings, features)  # (B, N, HW)
        attention = attention.view(B, N, H, W)  # (B, N, H, W)
        
        # 应用sigmoid激活函数得到最终掩码
        masks = torch.sigmoid(attention)
        # print(f"Output masks shape: {masks.shape}")
        
        return masks