"""匈牙利匹配分配器模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianAssigner(nn.Module):
    """匈牙利匹配分配器
    
    使用匈牙利算法进行最优二分图匹配，用于目标检测和分割任务中的标签分配。
    
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
    def forward(self, pred_logits, pred_masks, gt_labels, gt_masks):
        """前向传播
        
        参数:
            pred_logits (Tensor): 预测的类别logits, shape (B, N, C)
            pred_masks (Tensor): 预测的分割掩码, shape (B, N, H, W)
            gt_labels (List[Tensor]): 真实标签列表
            gt_masks (List[Tensor]): 真实掩码列表
            
        返回:
            indices (List[Tuple[Tensor, Tensor]]): 每个图像的匹配索引对
        """
        indices = []
        
        # 对每个图像进行匹配
        for i, (p_logits, p_masks, t_labels, t_masks) in enumerate(
            zip(pred_logits, pred_masks, gt_labels, gt_masks)):
            
            # 打印输入张量的形状
            #print(f"\nBatch {i}:")
            #print(f"pred_logits shape: {p_logits.shape}")
            #print(f"pred_masks shape: {p_masks.shape}")
            #print(f"gt_labels shape: {t_labels.shape}")
            #print(f"gt_masks shape: {t_masks.shape}")
            
            # 将预测掩码上采样到与真实掩码相同的大小
            if p_masks.shape[-2:] != t_masks.shape[-2:]:
                p_masks = F.interpolate(
                    p_masks.unsqueeze(0),  # 添加批次维度
                    size=t_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # 移除批次维度
                #print(f"Upsampled pred_masks shape: {p_masks.shape}")
            
            # 确保掩码维度正确
            if len(p_masks.shape) == 2:  # (N, H*W)
                H, W = t_masks.shape[1:]
                p_masks = p_masks.view(p_masks.shape[0], H, W)  # (N, H, W)
                #print(f"Reshaped pred_masks to: {p_masks.shape}")
            
            # 计算类别代价矩阵
            cost_class = -p_logits[:, t_labels]
            #print(f"cost_class shape: {cost_class.shape}")
            
            # 计算掩码代价矩阵
            cost_mask = self.compute_mask_cost(p_masks, t_masks)
            #print(f"cost_mask shape: {cost_mask.shape}")
            
            # 计算Dice代价矩阵
            cost_dice = self.compute_dice_cost(p_masks, t_masks)
            #print(f"cost_dice shape: {cost_dice.shape}")
            
            # 组合代价矩阵
            C = (self.match_cost_class * cost_class + 
                 self.match_cost_mask * cost_mask +
                 self.match_cost_dice * cost_dice)
            #print(f"Combined cost matrix shape: {C.shape}")
            
            # 使用匈牙利算法进行匹配
            C = C.cpu()
            pred_ids, gt_ids = linear_sum_assignment(C)
            indices.append((torch.as_tensor(pred_ids, dtype=torch.int64),
                          torch.as_tensor(gt_ids, dtype=torch.int64)))
            
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