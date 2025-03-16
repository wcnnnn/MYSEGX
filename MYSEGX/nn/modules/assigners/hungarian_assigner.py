"""匈牙利匹配分配器模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianAssigner(nn.Module):
    """匈牙利匹配分配器
    
    使用匈牙利算法进行最优二分图匹配，用于实例分割任务中的标签分配。
    
    参数:
        match_cost_class (float): 类别代价权重
        match_cost_mask (float): 掩码代价权重
        match_cost_dice (float): Dice代价权重
    """
    def __init__(self, match_cost_class=1, match_cost_mask=1, match_cost_dice=1):
        super().__init__()
        self.match_cost_class = match_cost_class
        self.match_cost_mask = match_cost_mask
        self.match_cost_dice = match_cost_dice
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """前向传播
        
        参数:
            outputs (dict): 模型输出字典
                - pred_logits: 预测的类别logits (B, N, C)
                - pred_masks: 预测的分割掩码 (B, N, H, W)
            targets (list[dict]): 目标字典列表
                - masks: 实例掩码 (M, H, W)
                - labels: 类别标签 (M,)
            
        返回:
            indices (List[Tuple[Tensor, Tensor]]): 每个图像的匹配索引对
        """
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        B = len(targets)
        indices = []
        
        # 对每个图像进行匹配
        for b in range(B):
            # 获取当前图像的预测和目标
            cur_pred_logits = pred_logits[b]  # (N, C)
            cur_pred_masks = pred_masks[b]     # (N, H, W)
            cur_targets = targets[b]
            
            if len(cur_targets['labels']) == 0:
                # 如果没有目标实例，返回空匹配
                indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)
                ))
                continue
            
            # 获取目标掩码和标签
            gt_masks = cur_targets['masks']    # (M, H, W)
            gt_labels = cur_targets['labels']  # (M,)
            
            # 将预测掩码上采样到与真实掩码相同的大小
            if cur_pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                cur_pred_masks = F.interpolate(
                    cur_pred_masks.unsqueeze(0),
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # 计算类别代价矩阵
            cost_class = -cur_pred_logits[:, gt_labels]  # (N, M)
            
            # 计算掩码代价矩阵
            cost_mask = self.compute_mask_cost(cur_pred_masks, gt_masks)  # (N, M)
            
            # 计算Dice代价矩阵
            cost_dice = self.compute_dice_cost(cur_pred_masks, gt_masks)  # (N, M)
            
            # 组合代价矩阵
            C = (self.match_cost_class * cost_class + 
                 self.match_cost_mask * cost_mask +
                 self.match_cost_dice * cost_dice)
            
            # 使用匈牙利算法进行匹配
            C = C.cpu()
            pred_ids, gt_ids = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(pred_ids, dtype=torch.int64),
                torch.as_tensor(gt_ids, dtype=torch.int64)
            ))
        
        return indices
    
    def compute_mask_cost(self, pred_masks, target_masks):
        """计算掩码代价"""
        device = pred_masks.device
        pred_masks = pred_masks.to(device)
        target_masks = target_masks.to(device)
        
        # 展平掩码
        pred_masks = pred_masks.flatten(1)  # (N, H*W)
        target_masks = target_masks.flatten(1)  # (M, H*W)
        
        # 计算二值交叉熵
        pred_masks = pred_masks.sigmoid()
        
        # 确保在同一设备上计算
        pred_masks = pred_masks.to(device)
        target_masks = target_masks.to(device)
        
        # 计算正样本代价 (N, M, H*W)
        pos_cost = -(pred_masks.unsqueeze(1) * target_masks.unsqueeze(0))
        
        # 计算负样本代价 (N, M, H*W)
        neg_cost = -((1 - pred_masks).unsqueeze(1) * (1 - target_masks).unsqueeze(0))
        
        # 在空间维度上取平均
        cost_mask = pos_cost.mean(2) + neg_cost.mean(2)  # (N, M)
        return cost_mask
    
    def compute_dice_cost(self, pred_masks, target_masks):
        """计算Dice代价"""
        device = pred_masks.device
        pred_masks = pred_masks.to(device)
        target_masks = target_masks.to(device)
        
        # 展平掩码
        pred_masks = pred_masks.flatten(1).sigmoid()  # (N, H*W)
        target_masks = target_masks.flatten(1)  # (M, H*W)
        
        # 确保在同一设备上计算
        pred_masks = pred_masks.to(device)
        target_masks = target_masks.to(device)
        
        # 计算分子 (N, M)
        numerator = 2 * torch.matmul(pred_masks, target_masks.t())
        
        # 计算分母 (N, M)
        denominator = pred_masks.sum(-1).unsqueeze(1) + target_masks.sum(-1).unsqueeze(0)
        
        # 计算Dice系数
        cost_dice = 1 - (numerator + 1) / (denominator + 1)
        return cost_dice