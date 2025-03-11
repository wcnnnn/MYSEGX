"""语义分割掩码头模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .detr_mask_head import FPNBlock

class SemanticMaskHead(nn.Module):
    """语义分割掩码头
    
    使用FPN结构融合多尺度特征，生成语义分割掩码。
    
    参数:
        in_channels (int): 输入特征通道数
        hidden_dim (int): 隐藏层维度
        num_classes (int): 类别数量
    """
    def __init__(self, in_channels, hidden_dim, num_classes, backbone_type="resnet50"):
        super().__init__()
        
        # 根据backbone类型动态调整输入通道数和特征尺度
        if backbone_type.lower() in ["resnet50", "resnet101"]:
            fpn_dims = [2048, 1024, 512, 256]  # ResNet50/101的特征维度
            self.target_size = None  # 使用第一层特征图的大小
        elif backbone_type.lower() in ["resnet18", "resnet34"]:
            fpn_dims = [512, 256, 128, 64]  # ResNet18/34的特征维度
            self.target_size = (200, 200)  # 固定输出大小为200x200
        elif backbone_type.lower() == "mobilenetv2":
            fpn_dims = [1280, 96, 32, 24]  # MobileNetV2的特征维度
            self.target_size = None
        elif backbone_type.lower() in ["vgg16", "vgg19"]:
            fpn_dims = [512, 512, 256, 128]  # VGG16/19的特征维度
            self.target_size = None
        else:
            fpn_dims = [in_channels, in_channels//2, in_channels//4, in_channels//8]
            self.target_size = None
        
        print(f"[DEBUG] SemanticMaskHead初始化 - 主干网络: {backbone_type}")
        print(f"[DEBUG] FPN维度: {fpn_dims}")
        
        # 添加输入特征投影层 - 使用实际的输入通道数
        self.input_proj = nn.Conv2d(fpn_dims[0], hidden_dim, 1)
        print(f"[DEBUG] 输入投影层配置: 输入通道={fpn_dims[0]}, 输出通道={hidden_dim}")
        
        # FPN层 - 使用实际的输入通道数
        self.fpn_blocks = nn.ModuleList([
            FPNBlock(fpn_dims[1], hidden_dim),  # layer3 (1/16)
            FPNBlock(fpn_dims[2], hidden_dim),  # layer2 (1/8)
            FPNBlock(fpn_dims[3], hidden_dim)   # layer1 (1/4)
        ])
        
        # 添加FPN输出的BatchNorm层
        self.fpn_norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim)
        ])
        
        # 特征解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 掩码预测器
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, num_classes, 1),
            nn.Tanh()  # 确保输出在[-1,1]范围内
        )
    
    def forward(self, features):
        """前向传播
        
        参数:
            features (List[Tensor]): 主干网络输出的多尺度特征列表
                - features[0]: layer1输出 (1/4或1/2，取决于backbone)
                - features[1]: layer2输出 (1/8或1/4)
                - features[2]: layer3输出 (1/16或1/8)
                - features[3]: layer4输出 (1/32或1/16)
        
        返回:
            Tensor: 预测的分割掩码 (B, C, H, W)，始终是输入的1/4大小
        """
        print(f"\n[DEBUG] SemanticMaskHead.forward - 特征列表长度: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  特征[{i}]: shape={feat.shape}, 范围=[{feat.min():.3f}, {feat.max():.3f}]")
        
        # 投影输入特征 - 从最深层开始处理
        x = self.input_proj(features[-1])  # 处理最深层特征(layer4)
        
        # 自顶向下的特征融合
        for i, (feat, fpn, norm) in enumerate(zip(reversed(features[:-1]), self.fpn_blocks, self.fpn_norms)):
            x = F.interpolate(x, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            feat = fpn(feat)
            x = x + feat
            x = norm(x)  # 添加BatchNorm
            print(f"[DEBUG] FPN块{i+1}输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        # 解码特征
        x = self.decoder(x)
        print(f"[DEBUG] 解码器输出: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
        
        # 预测掩码
        masks = self.predictor(x)
        print(f"[DEBUG] 预测掩码: shape={masks.shape}, 范围=[{masks.min():.3f}, {masks.max():.3f}]")
        
        # 调整输出大小
        if self.target_size is not None:
            masks = F.interpolate(masks, size=self.target_size, 
                                mode='bilinear', align_corners=False)
        
        print(f"[DEBUG] 最终掩码: shape={masks.shape}, 范围=[{masks.min():.3f}, {masks.max():.3f}]")
        
        return masks
