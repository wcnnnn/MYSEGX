"""分割损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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
    
    计算DETR模型的总损失，包括语义分割和实例分割两种模式。
    
    参数:
        num_classes (int): 类别数量
        matcher (nn.Module): 匈牙利匹配器，仅用于实例分割
        weight_dict (dict): 损失权重字典
        task_type (str): 任务类型，'semantic'或'instance'
        eos_coef (float): 背景类别的权重系数，仅用于实例分割
    """
    def __init__(self, num_classes, matcher=None, weight_dict=None, task_type='semantic', eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {'ce': 1.0, 'mask': 1.0, 'dice': 1.0}
        self.task_type = task_type
        
        # 分类损失权重 - 仅用于实例分割
        if task_type == 'instance':
            empty_weight = torch.ones(num_classes)
            empty_weight[0] = eos_coef  # 背景类权重
            self.register_buffer('empty_weight', empty_weight)
        
        # 掩码损失
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.ce_loss = CrossEntropyLoss(ignore_index=255)
        
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        
    def forward(self, outputs, targets):
        """前向传播
        
        参数:
            outputs (dict): 模型输出字典
                - pred_logits: 预测的类别logits (仅实例分割)
                - pred_masks: 预测的分割掩码
                    - 语义分割: (B, C, H, W)
                    - 实例分割: (B, Q, H, W)
            targets (list[dict]): 目标字典列表
                - semantic_mask: 语义分割掩码 (B, H, W)
                - labels: 类别标签 (仅实例分割)
        """
        if self.task_type == 'semantic':
            # 语义分割模式
            logging.debug("DETRLoss - Semantic segmentation mode")
            pred_masks = outputs['pred_masks']  # (B, C, H, W)
            logging.debug(f"Pred masks: shape={pred_masks.shape}, range=[{pred_masks.min().item():.3f}, {pred_masks.max().item():.3f}]")
            
            target_masks = torch.stack([t['semantic_mask'] for t in targets])  # (B, H, W)
            logging.debug(f"Target masks: shape={target_masks.shape}, range=[{target_masks.min().item()}, {target_masks.max().item()}]")
            
            # 将目标掩码下采样到预测掩码的大小
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
            logging.debug(f"Resized targets: shape={target_masks.shape}, range=[{target_masks.min().item()}, {target_masks.max().item()}]")
            
            # 确保目标掩码在有效范围内
            target_masks = torch.clamp(target_masks, 0, self.num_classes - 1)
            
            # 检查设备
            if pred_masks.device != target_masks.device:
                logging.warning(f"Device mismatch - pred: {pred_masks.device}, target: {target_masks.device}")
                target_masks = target_masks.to(pred_masks.device)
            
            # 计算交叉熵损失
            try:
                loss_ce = self.ce_loss(pred_masks, target_masks)
                logging.debug(f"CE loss computed: {loss_ce.item():.3f}")
            except Exception as e:
                logging.error(f"Error computing CE loss: {str(e)}")
                raise
            
            # 计算每个类别的Dice损失
            loss_dice = 0
            try:
                for c in range(1, self.num_classes):  # 跳过背景类
                    pred_c = pred_masks[:, c]  # (B, H, W)
                    target_c = (target_masks == c).float()  # (B, H, W)
                    if target_c.sum() > 0:  # 只在存在当前类别时计算损失
                        class_dice_loss = self.dice_loss(pred_c, target_c)
                        loss_dice += class_dice_loss
                        logging.debug(f"Class {c} Dice loss: {class_dice_loss.item():.3f}")
                loss_dice /= (self.num_classes - 1)  # 平均每个类别的损失
                logging.debug(f"Average Dice loss: {loss_dice.item():.3f}")
            except Exception as e:
                logging.error(f"Error computing Dice loss: {str(e)}")
                raise
            
            # 返回损失字典
            losses = {
                'loss_ce': loss_ce * self.weight_dict['ce'],
                'loss_dice': loss_dice * self.weight_dict['dice']
            }
            logging.debug(f"Final weighted losses - CE: {losses['loss_ce'].item():.3f}, Dice: {losses['loss_dice'].item():.3f}")
            return losses
            
        else:  # instance
            # 获取设备信息
            device = outputs['pred_logits'].device
            
            # 准备目标掩码和标签
            batch_size = len(targets)
            target_masks = []
            target_labels = []
            
            for i in range(batch_size):
                # 获取语义分割掩码
                semantic_mask = targets[i]['semantic_mask']  # (H, W)
                unique_labels = targets[i]['labels']  # 存在的类别标签
                
                # 为每个类别创建二值掩码
                for label in unique_labels:
                    if label == 0:  # 跳过背景类
                        continue
                    binary_mask = (semantic_mask == label).float()  # (H, W)
                    target_masks.append(binary_mask)
                    target_labels.append(label)
            
            if not target_masks:  # 如果没有前景对象
                target_masks = torch.zeros((1, *outputs['pred_masks'].shape[-2:]), device=device)
                target_labels = torch.tensor([0], device=device)
            else:
                target_masks = torch.stack(target_masks)  # (N, H, W)
                target_labels = torch.tensor(target_labels, device=device)  # (N,)
            
            # 将目标重组为列表格式
            targets_formatted = [{
                'masks': target_masks,
                'labels': target_labels
            }]
            
            # 计算匹配
            indices = self.matcher(outputs, targets_formatted)
            
            # 计算分类损失
            src_logits = outputs['pred_logits']  # (B, Q, C)
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets_formatted, indices)])
            
            target_classes = torch.full(src_logits.shape[:2], 0,
                                      dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                    weight=self.empty_weight.to(device))
            
            # 计算掩码损失
            src_masks = outputs['pred_masks']  # (B, Q, H, W)
            
            # 获取匹配的掩码
            src_masks = src_masks[idx]  # (N, H, W)
            target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets_formatted, indices)])
            
            # 将目标掩码下采样到源掩码大小
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=src_masks.shape[-2:],
                mode='nearest'
            ).squeeze(1)
            
            # 计算Dice损失和Focal损失
            loss_dice = self.dice_loss(src_masks, target_masks)
            loss_focal = self.focal_loss(src_masks, target_masks)
            
            # 组合所有损失
            losses = {
                'loss_ce': loss_ce * self.weight_dict['ce'],
                'loss_mask': loss_focal * self.weight_dict['mask'],
                'loss_dice': loss_dice * self.weight_dict['dice']
            }
            return losses

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
            pred_masks (Tensor): 预测掩码, shape (N, C, H, W)
            gt_masks (Tensor): 真实掩码, shape (N, H, W)
            weights (Tensor): 样本权重, shape (N,)
            
        返回:
            loss (Tensor): 交叉熵损失
        """
        logging.debug(f"CrossEntropyLoss input shapes - pred: {pred_masks.shape}, gt: {gt_masks.shape}")
        logging.debug(f"Pred range: [{pred_masks.min().item():.3f}, {pred_masks.max().item():.3f}]")
        logging.debug(f"GT range: [{gt_masks.min().item()}, {gt_masks.max().item()}]")
        
        if self.weight is not None:
            logging.debug(f"Weight tensor shape: {self.weight.shape}")
            
        if weights is not None:
            logging.debug(f"Sample weights shape: {weights.shape}")
        
        # 检查设备
        if pred_masks.device != gt_masks.device:
            logging.warning(f"Device mismatch - pred: {pred_masks.device}, gt: {gt_masks.device}")
            gt_masks = gt_masks.to(pred_masks.device)
        
        # 检查gt_masks的值域
        unique_labels = torch.unique(gt_masks)
        logging.debug(f"Unique labels in gt_masks: {unique_labels.tolist()}")
        if self.ignore_index != 255:
            logging.debug(f"Using custom ignore_index: {self.ignore_index}")
        
        try:
            loss = F.cross_entropy(
                pred_masks,
                gt_masks,
                weight=self.weight,
                reduction=self.reduction,
                ignore_index=self.ignore_index
            )
            logging.debug(f"Loss computed successfully: {loss.item():.3f}")
            return loss
        except Exception as e:
            logging.error(f"Error computing cross entropy loss: {str(e)}")
            raise