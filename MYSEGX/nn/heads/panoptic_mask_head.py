"""全景分割掩码头模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from MYSEGX.nn.heads.detr_mask_head import MaskHead

class PanopticMaskHead(MaskHead):
    """全景分割掩码头
    
    在MaskHead的基础上添加了一个额外的分类头，用于区分物体和背景。
    """
    def __init__(self, in_channels, hidden_dim, num_classes, backbone_type="resnet50"):
        super().__init__(in_channels, hidden_dim, num_classes, backbone_type=backbone_type, task_type="panoptic")
        print(f"[DEBUG] PanopticMaskHead初始化 - 主干网络: {backbone_type}")
        
        # 添加额外的分类头
        self.thing_stuff_classifier = nn.Linear(hidden_dim, 2)  # 2类：物体和背景
        print(f"[DEBUG] 创建物体/背景分类器: 输入维度={hidden_dim}, 输出维度=2")
        
    def forward(self, features, mask_embeddings):
        """前向传播
        
        参数:
            features (List[Tensor]): 主干网络输出的多尺度特征列表
            mask_embeddings (Tensor): 解码器输出的掩码嵌入
            
        返回:
            Dict: 包含掩码和物体/背景分类结果的字典
        """
        print(f"\n[DEBUG] PanopticMaskHead.forward - 主干网络: {self.backbone_type}")
        print(f"[DEBUG] 特征列表长度: {len(features)}")
        print(f"[DEBUG] 掩码嵌入: shape={mask_embeddings.shape}")
        
        # 调用父类的前向传播生成掩码
        masks = super().forward(features, mask_embeddings)
        print(f"[DEBUG] 父类MaskHead生成的掩码: shape={masks.shape}, 范围=[{masks.min():.3f}, {masks.max():.3f}]")
        
        # 计算物体/背景分类结果
        B, N, C = mask_embeddings.shape
        thing_stuff_logits = self.thing_stuff_classifier(mask_embeddings.mean(dim=1))
        print(f"[DEBUG] 物体/背景分类结果: shape={thing_stuff_logits.shape}, 范围=[{thing_stuff_logits.min():.3f}, {thing_stuff_logits.max():.3f}]")
        
        # 返回结果字典
        result = {
            'pred_masks': masks,
            'pred_thing_stuff': thing_stuff_logits
        }
        print(f"[DEBUG] 返回结果字典，包含键: {list(result.keys())}")
        
        return result