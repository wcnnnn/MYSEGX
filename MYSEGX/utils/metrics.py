"""分割指标计算模块"""

import torch
import torch.nn.functional as F

def calculate_accuracy(pred, target):
    """计算分类准确率
    
    参数:
        pred: 预测结果，可以是以下格式之一:
            - Tensor shape (N, C): logits
            - Tensor shape (N,): 类别索引
            - dict: 包含'pred_logits'键的字典
        target: 目标类别索引, shape (N,)
        
    返回:
        accuracy (float): 准确率
    """
    if isinstance(pred, dict):
        pred = pred['pred_logits']
    
    if pred.dim() == 2:
        pred = pred.argmax(dim=1)
    
    return (pred == target).float().mean().item()

def calculate_iou(pred, target, smooth=1e-6):
    """计算IoU (Intersection over Union)
    
    参数:
        pred: 预测掩码，可以是以下格式之一:
            - Tensor shape (N, H, W): 二值掩码
            - Tensor shape (N, C, H, W): 多类别掩码
            - dict: 包含'pred_masks'键的字典
        target: 目标掩码, shape (N, H, W)
        smooth: 平滑项，防止除零
        
    返回:
        iou (float): IoU分数
    """
    if isinstance(pred, dict):
        pred = pred['pred_masks']
    
    # 处理多类别掩码
    if pred.dim() == 4 and pred.shape[1] > 1:  # (N, C, H, W)
        pred = pred.argmax(dim=1)  # (N, H, W)
    elif pred.dim() == 4:  # DETR格式 (B, N, H, W)
        pred = pred.sigmoid() > 0.5
    
    # 计算交集和并集
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    return (intersection + smooth) / (union + smooth)

def calculate_dice(pred, target, smooth=1e-6):
    """计算Dice系数
    
    参数:
        pred: 预测掩码，可以是以下格式之一:
            - Tensor shape (N, H, W): 二值掩码
            - Tensor shape (N, C, H, W): 多类别掩码
            - dict: 包含'pred_masks'键的字典
        target: 目标掩码, shape (N, H, W)
        smooth: 平滑项，防止除零
        
    返回:
        dice (float): Dice系数
    """
    if isinstance(pred, dict):
        pred = pred['pred_masks']
    
    # 处理多类别掩码
    if pred.dim() == 4 and pred.shape[1] > 1:  # (N, C, H, W)
        pred = pred.argmax(dim=1)  # (N, H, W)
    elif pred.dim() == 4:  # DETR格式 (B, N, H, W)
        pred = pred.sigmoid() > 0.5
    
    # 计算分子和分母
    intersection = (pred & target).float().sum()
    sum_pred = pred.float().sum()
    sum_target = target.float().sum()
    
    return (2.0 * intersection + smooth) / (sum_pred + sum_target + smooth)

def calculate_pixel_accuracy(pred, target):
    """计算像素级准确率
    
    参数:
        pred: 预测掩码，可以是以下格式之一:
            - Tensor shape (N, H, W): 二值掩码
            - Tensor shape (N, C, H, W): 多类别掩码
            - dict: 包含'pred_masks'键的字典
        target: 目标掩码, shape (N, H, W)
        
    返回:
        accuracy (float): 像素准确率
    """
    if isinstance(pred, dict):
        pred = pred['pred_masks']
    
    # 处理多类别掩码
    if pred.dim() == 4 and pred.shape[1] > 1:  # (N, C, H, W)
        pred = pred.argmax(dim=1)  # (N, H, W)
    elif pred.dim() == 4:  # DETR格式 (B, N, H, W)
        pred = pred.sigmoid() > 0.5
    
    # 计算正确预测的像素数量
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    
    return correct / total

class MetricCalculator:
    """指标计算器
    
    用于在训练和验证过程中累积和计算各种评估指标。
    支持DETR和UNet两种模型类型。
    """
    def __init__(self, model_type='detr'):
        self.model_type = model_type
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        self.total_accuracy = 0.0
        self.total_iou = 0.0
        self.total_dice = 0.0
        self.total_pixel_accuracy = 0.0
        self.num_batches = 0
        
    def update(self, outputs, targets):
        """更新指标
        
        参数:
            outputs: 模型输出
                DETR: 包含以下键的字典:
                    - pred_logits: 预测的类别logits (B, N, C)
                    - pred_masks: 预测的分割掩码 (B, N, H, W)
                UNet: 预测的分割掩码 (B, C, H, W)
            targets: 目标
                DETR: 字典列表，每个字典包含:
                    - labels: 类别标签
                    - masks: 分割掩码
                UNet: 目标掩码张量 (B, H, W)
        """
        if self.model_type == 'detr':
            # DETR模型的指标计算
            pred_logits = outputs['pred_logits']  # (B, N, C)
            pred_masks = outputs['pred_masks']    # (B, N, H, W)
            
            batch_size = pred_logits.shape[0]
            
            # 对每个样本计算指标
            for i in range(batch_size):
                # 分类准确率
                pred_classes = pred_logits[i].argmax(dim=1)  # (N,)
                target_classes = torch.full_like(pred_classes, pred_logits.shape[-1]-1)  # 默认为背景类
                
                # 获取当前批次的目标
                target = targets[i]
                target_classes[:len(target['labels'])] = target['labels']
                self.total_accuracy += (pred_classes == target_classes).float().mean().item()
                
                # IoU和Dice系数
                pred_mask = pred_masks[i]  # (N, H, W)
                target_mask = torch.zeros_like(pred_mask[0])  # (H, W)
                
                # Resize target masks to match prediction size and convert to boolean
                target_masks = F.interpolate(target['masks'].unsqueeze(0), 
                                           size=pred_mask.shape[-2:], 
                                           mode='bilinear', 
                                           align_corners=False)[0]
                target_masks = (target_masks > 0.5)  # Convert to boolean
                target_mask = target_masks.any(dim=0)  # 合并所有实例掩码
                
                self.total_iou += calculate_iou(pred_mask.sigmoid() > 0.5, target_mask)
                self.total_dice += calculate_dice(pred_mask.sigmoid() > 0.5, target_mask)
                self.total_pixel_accuracy += calculate_pixel_accuracy(pred_mask.sigmoid() > 0.5, target_mask)
        else:
            # UNet模型的指标计算
            batch_size = outputs.shape[0]
            
            # 对每个样本计算指标
            for i in range(batch_size):
                pred = outputs[i]  # (C, H, W)
                target = targets[i]  # (H, W)
                
                # 计算准确率
                pred_classes = pred.argmax(dim=0)  # (H, W)
                self.total_accuracy += (pred_classes == target).float().mean().item()
                
                # 计算IoU和Dice系数
                self.total_iou += calculate_iou(pred_classes, target)
                self.total_dice += calculate_dice(pred_classes, target)
                self.total_pixel_accuracy += calculate_pixel_accuracy(pred_classes, target)
        
        self.num_batches += 1
        
    def compute(self):
        """计算平均指标
        
        返回:
            metrics (dict): 包含所有计算得到的指标
        """
        return {
            'accuracy': self.total_accuracy / self.num_batches,
            'iou': self.total_iou / self.num_batches,
            'dice': self.total_dice / self.num_batches,
            'pixel_accuracy': self.total_pixel_accuracy / self.num_batches
        }