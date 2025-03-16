"""分割损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, List

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
        """
        参数:
            pred (Tensor): 预测掩码
                - 语义分割: shape (N, C, H, W)
                - 实例分割: shape (N, H, W)
            target (Tensor): 真实掩码
                - 语义分割: shape (N, H, W)
                - 实例分割: shape (N, H, W)
        """
        # 检查输入维度
        if pred.dim() == 4:  # 语义分割模式
            logging.debug("DiceLoss - 使用语义分割模式")
            # 对logits应用softmax
            pred = F.softmax(pred, dim=1)
            
            # 获取实际存在的类别
            unique_classes = torch.unique(target)
            dice_scores = []
            
            # 对每个存在的类别计算Dice系数
            for i in unique_classes:
                pred_i = pred[:, i, :, :]
                target_i = (target == i).float()
                intersection = (pred_i * target_i).sum()
                union = pred_i.sum() + target_i.sum()
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_scores.append(dice)
                
        else:  # 实例分割模式
            logging.debug("DiceLoss - 使用实例分割模式")
            # pred和target已经是二值掩码
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores = [dice]
        
        # 计算平均Dice损失
        loss = 1.0 - torch.mean(torch.stack(dice_scores))
        logging.debug(f"DiceLoss - final loss: {loss.item():.4f}")
        
        return loss

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
        """
        参数:
            pred (Tensor): 预测掩码
                - 语义分割: shape (N, C, H, W)
                - 实例分割: shape (N, H, W)
            target (Tensor): 真实掩码
                - 语义分割: shape (N, H, W)
                - 实例分割: shape (N, H, W)
        """
        logging.debug(f"FocalLoss - pred shape: {pred.shape}, target shape: {target.shape}")
        
        if pred.dim() == 4:  # 语义分割模式
            logging.debug("FocalLoss - 使用语义分割模式")
            pred = F.softmax(pred, dim=1)
            target_pred = pred.gather(1, target.unsqueeze(1))
        else:  # 实例分割模式
            logging.debug("FocalLoss - 使用实例分割模式")
            # 对于实例分割，pred已经是二值的logits
            pred = torch.sigmoid(pred)
            target_pred = pred
            target = target.float()
        
        logging.debug(f"FocalLoss - pred range: [{pred.min():.3f}, {pred.max():.3f}]")
        
        # 计算focal loss
        eps = 1e-6
        target_pred = torch.clamp(target_pred, eps, 1.0 - eps)
        
        # 计算focal权重
        pt = torch.where(target == 1, target_pred, 1 - target_pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算alpha权重
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # 计算二值交叉熵
        bce = -torch.log(torch.where(target == 1, target_pred, 1 - target_pred))
        
        # 组合所有权重
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
        
        if task_type == 'semantic':
            # 语义分割模式
            self.dice_loss = DiceLoss()
            self.focal_loss = FocalLoss()
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
            logging.info("初始化语义分割损失函数")
        else:
            # 实例分割模式
            empty_weight = torch.ones(num_classes + 1)  # +1 for background
            empty_weight[0] = eos_coef
            # 不要在这里注册buffer，而是作为ce_loss的参数
            self.ce_loss = nn.CrossEntropyLoss(weight=empty_weight)
            self.dice_loss = DiceLoss()
            self.focal_loss = FocalLoss()
            logging.info("初始化实例分割损失函数")
            logging.debug(f"空类别权重: shape={empty_weight.shape}")
        
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        
    def forward(self, outputs, targets):
        """前向传播"""
        device = outputs['pred_masks'].device if isinstance(outputs, dict) else outputs.device
        logging.debug(f"\n=== DETRLoss Forward ===")
        logging.debug(f"当前设备: {device}")
        
        # 确保ce_loss的权重在正确的设备上
        if hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(device)
        
        if self.task_type == 'semantic':
            # 语义分割模式
            logging.debug("处理语义分割损失")
            pred_masks = outputs['pred_masks']  # (B, C, H, W)
            logging.debug(f"预测掩码: shape={pred_masks.shape}, device={pred_masks.device}")
            
            target_masks = torch.stack([t['semantic_mask'] for t in targets])  # (B, H, W)
            logging.debug(f"目标掩码: shape={target_masks.shape}, device={target_masks.device}")
            
            # 确保设备一致性
            if target_masks.device != device:
                logging.debug(f"将目标掩码从 {target_masks.device} 移动到 {device}")
                target_masks = target_masks.to(device)
            
            # 将目标掩码下采样到预测掩码的大小
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
            logging.debug(f"调整大小后的目标掩码: shape={target_masks.shape}")
            
            # 检查设备
            logging.debug(f"损失计算前的设备检查:")
            logging.debug(f"- pred_masks: {pred_masks.device}")
            logging.debug(f"- target_masks: {target_masks.device}")
            logging.debug(f"- ce_loss weight: {getattr(self.ce_loss, 'weight', None)}")
            
            # 计算交叉熵损失
            try:
                loss_ce = self.ce_loss(pred_masks, target_masks)
                logging.debug(f"CE loss计算成功: {loss_ce.item():.4f}")
            except Exception as e:
                logging.error(f"CE loss计算失败: {str(e)}")
                logging.error(f"CE loss参数:")
                logging.error(f"- pred_masks: shape={pred_masks.shape}, device={pred_masks.device}")
                logging.error(f"- target_masks: shape={target_masks.shape}, device={target_masks.device}")
                raise
            
            # 计算Dice损失
            try:
                loss_dice = self.dice_loss(pred_masks, target_masks)
                logging.debug(f"Dice loss计算成功: {loss_dice.item():.4f}")
            except Exception as e:
                logging.error(f"Dice loss计算失败: {str(e)}")
                raise
            
            # 返回损失字典
            losses = {
                'loss_ce': loss_ce * self.weight_dict['ce'],
                'loss_dice': loss_dice * self.weight_dict['dice']
            }
            logging.debug(f"最终加权损失 - CE: {losses['loss_ce'].item():.4f}, Dice: {losses['loss_dice'].item():.4f}")
            return losses
            
        else:  # instance
            logging.debug("处理实例分割损失")
            # 获取设备信息
            device = outputs['pred_logits'].device
            logging.debug(f"当前设备: {device}")
            
            # 计算匹配
            indices = self.matcher(outputs, targets)
            logging.debug(f"匹配索引数量: {len(indices)}")
            
            # 计算分类损失
            src_logits = outputs['pred_logits']  # (B, Q, C+1)
            logging.debug(f"预测logits: shape={src_logits.shape}, device={src_logits.device}")
            
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
            logging.debug(f"目标类别: shape={target_classes_o.shape}, device={target_classes_o.device}")
            
            target_classes = torch.full(src_logits.shape[:2], 0,
                                      dtype=torch.int64, device=device)
            target_classes[idx] = target_classes_o
            
            # 检查设备一致性
            logging.debug(f"\n分类损失计算前的设备检查:")
            logging.debug(f"- src_logits: {src_logits.device}")
            logging.debug(f"- target_classes: {target_classes.device}")
            logging.debug(f"- empty_weight: {self.ce_loss.weight.device}")
            
            try:
                loss_ce = self.ce_loss(src_logits.transpose(1, 2), target_classes)
                logging.debug(f"分类损失计算成功: {loss_ce.item():.4f}")
            except Exception as e:
                logging.error(f"分类损失计算失败: {str(e)}")
                logging.error("详细参数信息:")
                logging.error(f"- src_logits: shape={src_logits.shape}, device={src_logits.device}")
                logging.error(f"- target_classes: shape={target_classes.shape}, device={target_classes.device}")
                logging.error(f"- empty_weight: shape={self.ce_loss.weight.shape}, device={self.ce_loss.weight.device}")
                raise
            
            # 计算掩码损失
            src_masks = outputs['pred_masks']  # (B, Q, H, W)
            logging.debug(f"预测掩码: shape={src_masks.shape}, device={src_masks.device}")
            
            # 获取匹配的掩码
            src_masks = src_masks[idx]  # (N, H, W)
            target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)])
            logging.debug(f"目标掩码: shape={target_masks.shape}, device={target_masks.device}")
            
            # 确保设备一致性
            if target_masks.device != device:
                logging.debug(f"将目标掩码从 {target_masks.device} 移动到 {device}")
                target_masks = target_masks.to(device)
            
            # 将目标掩码下采样到源掩码大小
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=src_masks.shape[-2:],
                mode='nearest'
            ).squeeze(1)
            
            # 计算Dice损失和Focal损失
            try:
                # 确保src_masks经过sigmoid激活
                src_masks_prob = src_masks.sigmoid()
                loss_dice = self.dice_loss(src_masks_prob, target_masks)
                loss_focal = self.focal_loss(src_masks, target_masks)
                logging.debug(f"掩码损失计算成功 - Dice: {loss_dice.item():.4f}, Focal: {loss_focal.item():.4f}")
            except Exception as e:
                logging.error(f"掩码损失计算失败: {str(e)}")
                logging.error(f"src_masks shape: {src_masks.shape}")
                logging.error(f"target_masks shape: {target_masks.shape}")
                raise
            
            # 组合所有损失
            losses = {
                'loss_ce': loss_ce * self.weight_dict['ce'],
                'loss_mask': loss_focal * self.weight_dict['mask'],
                'loss_dice': loss_dice * self.weight_dict['dice']
            }
            logging.debug("\n最终加权损失:")
            for k, v in losses.items():
                logging.debug(f"- {k}: {v.item():.4f}")
            
            return losses

class CrossEntropyLoss(nn.Module):
    """交叉熵损失函数
    
    计算预测掩码和真实掩码之间的交叉熵损失。
    
    参数:
        num_classes (int): 类别数量
        reduction (str): 损失计算方式，可选'mean'或'sum'
        ignore_index (int): 忽略的标签值，默认为255
    """
    def __init__(self, num_classes, reduction='mean', ignore_index=255):
        super().__init__()
        
        # 计算类别权重
        class_weights = torch.ones(num_classes)
        # 背景类(通常是第0类)的权重设置小一些，因为背景像素通常占比较大
        class_weights[0] = 0.1
        # 其他类别权重设为1
        class_weights[1:] = 1.0
        
        self.weight = class_weights
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        logging.info(f"初始化CrossEntropyLoss - 类别数: {num_classes}")
        logging.info(f"类别权重: {class_weights.tolist()}")
        
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
        
        # 确保权重在正确的设备上
        if self.weight is not None:
            self.weight = self.weight.to(pred_masks.device)
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

class OHEMCrossEntropyLoss(nn.Module):
    """带有在线难例挖掘的交叉熵损失函数
    
    通过选择损失值最高的像素点来计算损失，忽略简单样本。
    
    参数:
        num_classes (int): 类别数量
        thresh (float): 损失阈值，高于此值的像素被视为难例
        min_kept (int): 最少保留的像素数量
        ignore_index (int): 忽略的标签值，默认为255
    """
    def __init__(self, num_classes, thresh=0.7, min_kept=100000, ignore_index=255):
        super().__init__()
        
        # 计算类别权重（与CrossEntropyLoss保持一致）
        class_weights = torch.ones(num_classes)
        class_weights[0] = 0.1  # 背景类权重
        class_weights[1:] = 1.0  # 其他类别权重
        
        self.weight = class_weights
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        
        logging.info(f"初始化OHEMCrossEntropyLoss:")
        logging.info(f"- 类别数: {num_classes}")
        logging.info(f"- 损失阈值: {thresh}")
        logging.info(f"- 最少保留像素数: {min_kept}")
        logging.info(f"- 类别权重: {class_weights.tolist()}")
        
    def forward(self, pred_masks, gt_masks, weights=None):
        """前向传播
        
        参数:
            pred_masks (Tensor): 预测掩码, shape (N, C, H, W)
            gt_masks (Tensor): 真实掩码, shape (N, H, W)
            weights (Tensor): 样本权重, shape (N,)
            
        返回:
            loss (Tensor): OHEM交叉熵损失
        """
        logging.debug(f"OHEMCrossEntropyLoss input shapes - pred: {pred_masks.shape}, gt: {gt_masks.shape}")
        logging.debug(f"Pred range: [{pred_masks.min().item():.3f}, {pred_masks.max().item():.3f}]")
        logging.debug(f"GT range: [{gt_masks.min().item()}, {gt_masks.max().item()}]")
        
        # 确保权重在正确的设备上
        if self.weight is not None:
            self.weight = self.weight.to(pred_masks.device)
            logging.debug(f"Weight tensor shape: {self.weight.shape}")
        
        # 检查设备一致性
        if pred_masks.device != gt_masks.device:
            logging.warning(f"Device mismatch - pred: {pred_masks.device}, gt: {gt_masks.device}")
            gt_masks = gt_masks.to(pred_masks.device)
        
        # 计算每个像素的损失
        pixel_losses = F.cross_entropy(
            pred_masks,
            gt_masks,
            weight=self.weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        logging.debug(f"Pixel losses shape: {pixel_losses.shape}")
        logging.debug(f"Pixel losses range: [{pixel_losses.min().item():.3f}, {pixel_losses.max().item():.3f}]")
        
        # 将损失展平并移除忽略的像素
        pixel_losses = pixel_losses.contiguous().view(-1)
        valid_mask = gt_masks.contiguous().view(-1) != self.ignore_index
        pixel_losses = pixel_losses[valid_mask]
        
        # 如果没有有效像素，返回0损失
        if pixel_losses.numel() == 0:
            logging.warning("No valid pixels found for loss calculation")
            return torch.tensor(0.0, device=pred_masks.device)
        
        # 对损失值进行排序
        pixel_losses, _ = torch.sort(pixel_losses, descending=True)
        
        # 确定要保留的像素数量
        if pixel_losses.numel() < self.min_kept:
            # 如果有效像素数量少于min_kept，保留所有像素
            kept_losses = pixel_losses
            logging.debug(f"保留所有 {pixel_losses.numel()} 个有效像素")
        else:
            # 根据阈值和最小保留数量确定要保留的像素
            thresh_mask = pixel_losses > self.thresh
            if thresh_mask.sum() >= self.min_kept:
                # 如果高于阈值的像素数量足够，只保留这些像素
                kept_losses = pixel_losses[thresh_mask]
                logging.debug(f"根据阈值保留 {kept_losses.numel()} 个像素")
            else:
                # 否则保留前min_kept个最难的像素
                kept_losses = pixel_losses[:self.min_kept]
                logging.debug(f"保留前 {self.min_kept} 个最难像素")
        
        logging.debug(f"最终保留像素数量: {kept_losses.numel()}")
        logging.debug(f"保留像素的损失范围: [{kept_losses.min().item():.3f}, {kept_losses.max().item():.3f}]")
        
        # 计算平均损失
        loss = kept_losses.mean()
        logging.debug(f"最终OHEM损失: {loss.item():.3f}")
        
        return loss

class MaskRCNNLoss(nn.Module):
    """Mask R-CNN 损失函数
    
    计算 Mask R-CNN 模型的总损失，包括：
    1. RPN 分类损失
    2. RPN 边界框回归损失
    3. 分类损失
    4. 边界框回归损失
    5. 掩码损失
    """
    def __init__(
        self,
        num_classes: int,
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1
    ):
        """初始化 Mask R-CNN 损失函数
        
        参数:
            num_classes: 类别数量（包括背景）
            weight_dict: 损失权重字典
            eos_coef: 背景类别的权重系数
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {
            'rpn_cls': 1.0,
            'rpn_reg': 1.0,
            'cls': 1.0,
            'reg': 1.0,
            'mask': 1.0
        }
        
        # 初始化各个损失函数
        empty_weight = torch.ones(num_classes + 1)  # +1 for background
        empty_weight[0] = eos_coef
        self.ce_loss = nn.CrossEntropyLoss(weight=empty_weight)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """计算损失
        
        参数:
            outputs: 模型输出字典，包含：
                - pred_logits: 分类预测
                - pred_boxes: 边界框预测
                - pred_masks: 掩码预测
                - rpn_logits: RPN分类预测
                - rpn_boxes: RPN边界框预测
            targets: 目标字典列表
            
        返回:
            losses: 损失字典
        """
        device = outputs['pred_logits'].device
        logging.debug(f"\n=== MaskRCNNLoss Forward ===")
        logging.debug(f"当前设备: {device}")
        
        # 确保损失函数权重在正确的设备上
        if hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(device)
        
        # 计算 RPN 损失
        rpn_cls_loss = self._compute_rpn_cls_loss(outputs['rpn_logits'], targets)
        rpn_reg_loss = self._compute_rpn_reg_loss(outputs['rpn_boxes'], targets)
        
        # 计算分类和边界框回归损失
        cls_loss = self._compute_cls_loss(outputs['pred_logits'], targets)
        reg_loss = self._compute_reg_loss(outputs['pred_boxes'], targets)
        
        # 计算掩码损失
        mask_loss = self._compute_mask_loss(outputs['pred_masks'], targets)
        
        # 组合所有损失
        losses = {
            'loss_rpn_cls': rpn_cls_loss * self.weight_dict['rpn_cls'],
            'loss_rpn_reg': rpn_reg_loss * self.weight_dict['rpn_reg'],
            'loss_classifier': cls_loss * self.weight_dict['cls'],
            'loss_box_reg': reg_loss * self.weight_dict['reg'],
            'loss_mask': mask_loss * self.weight_dict['mask']
        }
        
        logging.debug("\n最终加权损失:")
        for k, v in losses.items():
            logging.debug(f"- {k}: {v.item():.4f}")
        
        return losses
    
    def _compute_rpn_cls_loss(self, rpn_logits: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算 RPN 分类损失"""
        # TODO: 实现 RPN 分类损失计算
        return torch.tensor(0.0, device=rpn_logits[0].device)
    
    def _compute_rpn_reg_loss(self, rpn_boxes: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算 RPN 边界框回归损失"""
        # TODO: 实现 RPN 边界框回归损失计算
        return torch.tensor(0.0, device=rpn_boxes[0].device)
    
    def _compute_cls_loss(self, pred_logits: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算分类损失"""
        # TODO: 实现分类损失计算
        return torch.tensor(0.0, device=pred_logits.device)
    
    def _compute_reg_loss(self, pred_boxes: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算边界框回归损失"""
        # TODO: 实现边界框回归损失计算
        return torch.tensor(0.0, device=pred_boxes.device)
    
    def _compute_mask_loss(self, pred_masks: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算掩码损失"""
        # TODO: 实现掩码损失计算
        return torch.tensor(0.0, device=pred_masks.device)

class YOLACTLoss(nn.Module):
    """YOLACT损失函数
    
    包含以下组件:
    1. 分类损失 - Focal Loss
    2. 边界框回归损失 - Smooth L1 Loss
    3. 掩码系数损失 - Smooth L1 Loss
    4. 原型掩码损失 - BCE Loss
    5. 语义分割损失 - Cross Entropy Loss
    
    参数:
        num_classes (int): 类别数量
        weight_dict (dict): 各损失分量的权重字典
    """
    def __init__(self, num_classes=80, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes
        
        # 设置默认权重
        self.weight_dict = {
            'cls': 1.0,
            'box': 1.5,
            'mask': 6.125,
            'proto': 1.0,
            'semantic': 1.0
        }
        if weight_dict is not None:
            self.weight_dict.update(weight_dict)
            
        # Focal Loss参数
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # 初始化损失函数
        self.focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = DiceLoss()
    
        print(f"初始化YOLACTLoss: num_classes={num_classes}")
        print(f"损失权重: {self.weight_dict}")
        
    def forward(self, outputs, targets):
        """计算YOLACT的总损失
        
        参数:
            outputs (dict): 模型输出字典
                - pred_cls: 预测的类别logits [B, N, num_classes]
                - pred_boxes: 预测的边界框 [B, N, 4]
                - pred_masks: 预测的掩码系数 [B, N, num_protos]
                - proto_out: prototype masks [B, num_protos, H, W]
            targets (list[dict]): 目标字典列表
                - boxes: 真实边界框 [M, 4]
                - labels: 真实类别 [M]
                - masks: 真实掩码 [M, H, W]
                
        返回:
            dict: 包含各个损失分量的字典
        """
        # 提取预测结果
        pred_cls = outputs['pred_cls']  # [B, N, num_classes]
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]
        pred_masks = outputs['pred_masks']  # [B, N, num_protos]
        proto_out = outputs['proto_out']  # [B, num_protos, H, W]
        
        device = pred_cls.device
        batch_size = pred_cls.shape[0]
        
        # 初始化损失
        losses = {
            'loss_cls': torch.zeros(1, device=device),
            'loss_box': torch.zeros(1, device=device),
            'loss_mask': torch.zeros(1, device=device),
            'loss_proto': torch.zeros(1, device=device),
            'loss_semantic': torch.zeros(1, device=device)
        }
        
        # 对每个batch处理
        for b in range(batch_size):
            target = targets[b]
            
            # 获取正样本掩码
            pos_mask = target['labels'] > 0
            num_pos = pos_mask.sum()
            
            if num_pos > 0:
                # 分类损失
                cls_loss = self.focal_loss(pred_cls[b], target['labels'])
                losses['loss_cls'] += cls_loss * self.weight_dict['cls']
                
                # 边界框回归损失
                box_loss = self.smooth_l1(pred_boxes[b][pos_mask], target['boxes'][pos_mask])
                losses['loss_box'] += box_loss * self.weight_dict['box']
                
                # 生成实例掩码
                pred_mask = torch.matmul(pred_masks[b], proto_out[b].view(proto_out.size(1), -1))
                pred_mask = pred_mask.view(-1, proto_out.size(2), proto_out.size(3))
                
                # 掩码损失
                mask_loss = self.dice_loss(pred_mask[pos_mask], target['masks'][pos_mask])
                losses['loss_mask'] += mask_loss * self.weight_dict['mask']
                
                # 原型掩码损失
                proto_loss = self.bce_loss(proto_out[b], target['masks'].float())
                losses['loss_proto'] += proto_loss * self.weight_dict['proto']
                
                # 语义分割损失
                semantic_loss = F.cross_entropy(
                    pred_cls[b].permute(1, 0),  # [num_classes, N]
                    target['labels'],
                    ignore_index=255
                )
                losses['loss_semantic'] += semantic_loss * self.weight_dict['semantic']
        
        # 计算平均损失
        for k in losses.keys():
            losses[k] = losses[k] / batch_size
            
        # 计算总损失
        losses['loss'] = sum(losses.values())
        
        return losses