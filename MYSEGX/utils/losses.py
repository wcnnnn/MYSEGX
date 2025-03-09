"""分割损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice损失函数
    
    计算预测掩码和真实掩码之间的Dice损失。
    
    参数:
        smooth (float): 平滑项，防止分母为0
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """前向传播
        
        参数:
            pred (Tensor): 预测掩码, shape (N, H, W)
            target (Tensor): 真实掩码, shape (N, H, W)
            
        返回:
            loss (Tensor): Dice损失
        """
        # 确保输入是二值掩码
        pred = pred.sigmoid()
        
        # 展平预测和目标
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 计算分子和分母
        intersection = (pred * target).sum()
        sum_pred = pred.sum()
        sum_target = target.sum()
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)
        
        return 1.0 - dice

class FocalLoss(nn.Module):
    """Focal损失函数
    
    计算预测掩码和真实掩码之间的Focal损失。
    
    参数:
        alpha (float): 正样本权重
        gamma (float): 聚焦参数
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """前向传播
        
        参数:
            pred (Tensor): 预测掩码, shape (N, H, W)
            target (Tensor): 真实掩码, shape (N, H, W)
            
        返回:
            loss (Tensor): Focal损失
        """
        # 确保输入是概率
        pred = pred.sigmoid()
        
        # 展平预测和目标
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 计算二值交叉熵
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算focal项
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算alpha项
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # 计算最终损失
        loss = focal_weight * alpha_weight * bce
        
        return loss.mean()

class DETRLoss(nn.Module):
    """DETR损失函数
    
    计算DETR模型的总损失，包括分类损失、掩码损失和匹配损失。
    
    参数:
        num_classes (int): 类别数量
        matcher (nn.Module): 匈牙利匹配器
        weight_dict (dict): 损失权重字典
        eos_coef (float): 背景类别的权重系数
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # 分类损失
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef  # 背景类权重
        self.register_buffer('empty_weight', empty_weight)
        
        # 掩码损失
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
    def loss_labels(self, outputs, targets, indices):
        """计算分类损失
        
        参数:
            outputs (dict): 模型输出字典
                - pred_logits: 预测的类别logits, shape (batch_size, num_queries, num_classes)
            targets (list[dict]): 目标字典列表，每个字典包含:
                - labels: 类别标签, shape (num_instances,)
            indices (list[tuple]): 匹配索引列表，每个元组包含:
                - src_idx: 源序列索引, shape (num_queries,)
                - tgt_idx: 目标序列索引, shape (num_instances,)
                
        返回:
            loss (dict): 分类损失字典
        """
        src_logits = outputs['pred_logits']  # (batch_size, num_queries, num_classes)
        
        # 记录输入形状
        # print(f"\nComputing classification loss:")
        # print(f"src_logits shape: {src_logits.shape}")
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        # print(f"target_classes_o shape: {target_classes_o.shape}")
        # print(f"target_classes_o values: {target_classes_o}")
        
        # 创建目标类别张量，初始化为背景类 (num_classes)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        # print(f"target_classes initial shape: {target_classes.shape}")
        
        # 将匹配的预测分配给对应的真实类别
        target_classes[idx] = target_classes_o
        # print(f"target_classes final shape: {target_classes.shape}")
        # print(f"target_classes unique values: {torch.unique(target_classes)}")
        
        # 确保empty_weight设备正确
        empty_weight = self.empty_weight.to(src_logits.device)
        # print(f"empty_weight shape: {empty_weight.shape}")
        # print(f"empty_weight values: {empty_weight}")
        
        # 计算交叉熵损失
        loss_ce = F.cross_entropy(
            src_logits.reshape(-1, src_logits.shape[-1]),  # (batch_size * num_queries, num_classes)
            target_classes.reshape(-1),                     # (batch_size * num_queries,)
            weight=empty_weight,
            reduction='mean'
        )
        
        return {'loss_ce': loss_ce}
    
    def loss_masks(self, outputs, targets, indices):
        """计算掩码损失
        
        参数:
            outputs (dict): 模型输出字典
                - pred_masks: 预测的分割掩码, shape (batch_size, num_queries, H, W)
            targets (list[dict]): 目标字典列表，每个字典包含:
                - masks: 分割掩码, shape (num_instances, H, W)
            indices (list[tuple]): 匹配索引列表，每个元组包含:
                - src_idx: 源序列索引, shape (num_queries,)
                - tgt_idx: 目标序列索引, shape (num_instances,)
                
        返回:
            loss (dict): 掩码损失字典
        """
        src_idx = self._get_src_permutation_idx(indices)
        batch_idx, src_idx = src_idx
        
        # 获取匹配的预测掩码和目标掩码
        src_masks = outputs['pred_masks']  # (B, N, H, W)
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)])  # (M, H, W)
        
        # print(f"\nComputing mask loss:")
        # print(f"src_masks shape: {src_masks.shape}")
        # print(f"target_masks shape: {target_masks.shape}")
        # print(f"batch_idx: {batch_idx}")
        # print(f"src_idx: {src_idx}")
        
        # 提取匹配的预测掩码
        src_masks = src_masks[batch_idx, src_idx]  # (M, H, W)
        # print(f"Matched src_masks shape: {src_masks.shape}")
        
        # 确保预测掩码和目标掩码具有相同的空间维度
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            # 将目标掩码下采样到预测掩码的尺寸
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),  # (M, 1, H, W)
                size=src_masks.shape[-2:],          # (h, w)
                mode='nearest'
            ).squeeze(1)  # (M, h, w)
            # print(f"Interpolated target_masks shape: {target_masks.shape}")
        
        # 将预测掩码转换为概率
        src_masks = src_masks.sigmoid()
        
        # 计算Dice损失
        num_masks = src_masks.shape[0]
        if num_masks == 0:
            loss_dice = src_masks.sum() * 0
            loss_focal = src_masks.sum() * 0
        else:
            # 展平掩码进行损失计算
            src_masks = src_masks.flatten(1)    # (num_masks, H*W)
            target_masks = target_masks.flatten(1)  # (num_masks, H*W)
            
            # 计算Dice损失
            numerator = 2 * (src_masks * target_masks).sum(1)
            denominator = src_masks.sum(1) + target_masks.sum(1)
            loss_dice = 1 - (numerator + 1e-6) / (denominator + 1e-6)
            loss_dice = loss_dice.mean()
            
            # 计算Focal损失
            loss_focal = F.binary_cross_entropy_with_logits(
                src_masks, target_masks, reduction='none')
            
            # 应用focal权重
            pt = torch.exp(-loss_focal)
            loss_focal = ((1 - pt) ** 2 * loss_focal).mean()
        
        return {
            'loss_dice': loss_dice,
            'loss_focal': loss_focal
        }
    
    def _get_src_permutation_idx(self, indices):
        """获取源序列的排列索引"""
        batch_idx = torch.cat([torch.full_like(src, i) 
                             for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        """前向传播
        
        参数:
            outputs (dict): 模型输出字典
                - pred_logits: 预测的类别logits, shape (batch_size, num_queries, num_classes)
                - pred_masks: 预测的分割掩码, shape (batch_size, num_queries, H, W)
            targets (list[dict]): 目标字典列表，每个字典包含:
                - labels: 类别标签, shape (num_instances,)
                - masks: 分割掩码, shape (num_instances, H, W)
                
        返回:
            losses (dict): 损失字典
        """
        # 记录输入形状
        # print(f"\nDETRLoss forward:")
        # print(f"pred_logits shape: {outputs['pred_logits'].shape}")
        # print(f"pred_masks shape: {outputs['pred_masks'].shape}")
        for i, t in enumerate(targets):
            # print(f"target {i} - labels shape: {t['labels'].shape}, masks shape: {t['masks'].shape}")
            pass
            
        # 匈牙利匹配
        indices = self.matcher(outputs['pred_logits'], outputs['pred_masks'],
                             [t['labels'] for t in targets],
                             [t['masks'] for t in targets])
        
        # 计算所有损失
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
        
        # 应用损失权重
        return {k: v * self.weight_dict[k] 
                for k, v in losses.items() 
                if k in self.weight_dict}

class CrossEntropyLoss(nn.Module):
    """交叉熵损失函数
    
    计算预测掩码和真实掩码之间的交叉熵损失。
    
    参数:
        weight (Tensor): 类别权重
        reduction (str): 损失计算方式，可选'mean'或'sum'
        ignore_index (int): 忽略的标签值，默认为255
    """
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, pred_masks, gt_masks, weights=None):
        """前向传播
        
        参数:
            pred_masks (Tensor): 预测掩码, shape (N, H, W)
            gt_masks (Tensor): 真实掩码, shape (N, H, W)
            weights (Tensor): 样本权重, shape (N,)
            
        返回:
            loss (Tensor): 交叉熵损失
        """
        # 创建掩码来标识需要忽略的位置
        mask = gt_masks != self.ignore_index
        
        # 将预测和真实掩码展平
        pred_flat = pred_masks.flatten(1)
        gt_flat = gt_masks.flatten(1)
        mask_flat = mask.flatten(1)
        
        # 只计算非忽略位置的损失
        loss = F.binary_cross_entropy_with_logits(
            pred_flat[mask_flat],
            gt_flat[mask_flat],
            weight=self.weight,
            reduction='none'
        )
        
        if weights is not None:
            loss = loss * weights.unsqueeze(1)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss