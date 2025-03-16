"""分割指标计算模块"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from time import time
from pathlib import Path
from typing import Dict, List, Union, Optional

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

def calculate_iou(pred, target, num_classes=21):
    """计算IoU（Intersection over Union）
    
    参数:
        pred: 预测张量 (N, H, W) 或 (H, W)
        target: 目标张量 (N, H, W) 或 (H, W)
        num_classes: 类别数量
        
    返回:
        IoU张量 (num_classes,) 或 (matched_ious, mean_iou)
    """
    print(f"[DEBUG] calculate_iou - Input shapes:")
    print(f"- pred: {pred.shape}, device={pred.device}, dtype={pred.dtype}")
    print(f"- target: {target.shape}, device={target.device}, dtype={target.dtype}")
    
    # 确保输入是长整型
    pred = pred.long()
    target = target.long()
    
    # 处理实例分割情况（维度不匹配）
    if pred.dim() == 3 and target.dim() == 3 and pred.shape[0] != target.shape[0]:
        print(f"[DEBUG] 检测到实例分割情况 - pred: {pred.shape[0]} instances, target: {target.shape[0]} instances")
        # 展平每个实例的掩码
        pred_flat = pred.view(pred.shape[0], -1)  # (N, H*W)
        target_flat = target.view(target.shape[0], -1)  # (M, H*W)
        
        # 计算所有实例对之间的IoU
        ious = torch.zeros((pred.shape[0], target.shape[0]), device=pred.device)
        for i in range(pred.shape[0]):
            for j in range(target.shape[0]):
                intersection = (pred_flat[i] & target_flat[j]).sum().float()
                union = (pred_flat[i] | target_flat[j]).sum().float()
                if union > 0:
                    ious[i, j] = intersection / union
        
        # 使用匈牙利算法找到最优匹配
        if ious.shape[0] > 0 and ious.shape[1] > 0:
            from scipy.optimize import linear_sum_assignment
            ious_cpu = ious.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-ious_cpu)  # 最大化
            matched_ious = ious[row_ind, col_ind]
            mean_iou = matched_ious.mean()
            return matched_ious, mean_iou
        return torch.zeros(max(pred.shape[0], target.shape[0]), device=pred.device), torch.tensor(0.0, device=pred.device)
    
    # 语义分割情况
    # 初始化IoU张量
    iou = torch.zeros(num_classes, device=pred.device)
    
    # 展平预测和目标张量
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    print(f"[DEBUG] Flattened tensors:")
    print(f"- pred_flat: shape={pred_flat.shape}, range=[{pred_flat.min()}, {pred_flat.max()}]")
    print(f"- target_flat: shape={target_flat.shape}, range=[{target_flat.min()}, {target_flat.max()}]")
    
    # 创建有效像素掩码（排除255）
    valid_mask = target_flat != 255
    
    # 获取目标中实际存在的类别（排除255）
    target_classes = torch.unique(target_flat[valid_mask])
    print(f"[DEBUG] 目标中实际存在的类别: {target_classes}")
    
    # 计算每个类别的IoU
    valid_classes = 0
    for cls in range(num_classes):
        # 如果类别不在目标中，跳过计算
        if cls not in target_classes and cls != 0:  # 始终计算背景类
            iou[cls] = 0.0
            continue
            
        # 创建当前类别的掩码，只在有效区域内计算
        pred_mask = (pred_flat == cls) & valid_mask
        target_mask = (target_flat == cls) & valid_mask
        
        # 计算交集和并集
        intersection = torch.logical_and(pred_mask, target_mask).sum().float()
        union = torch.logical_or(pred_mask, target_mask).sum().float()
        
        print(f"[DEBUG] Class {cls}:")
        print(f"- intersection: {intersection.item()}")
        print(f"- union: {union.item()}")
        
        # 计算IoU
        if union > 0:
            iou[cls] = intersection / union
            valid_classes += 1
        else:
            iou[cls] = 0.0
    
    # 计算平均IoU（只考虑目标中存在的类别）
    if valid_classes > 0:
        # 只计算目标中存在的类别的平均IoU
        mean_iou = sum([iou[cls] for cls in target_classes]) / len(target_classes)
    else:
        mean_iou = torch.tensor(0.0, device=pred.device)
    
    print(f"[DEBUG] IoU per class: {iou}")
    print(f"[DEBUG] Mean IoU (only for classes in target): {mean_iou.item()}")
    
    return iou, mean_iou

def calculate_dice(pred, target, num_classes=21, smooth=1e-6):
    """计算Dice系数
    
    参数:
        pred: 预测掩码 (N, H, W) 或 (H, W)
        target: 目标掩码 (M, H, W) 或 (H, W)
        num_classes: 类别数量
        smooth: 平滑项
        
    返回:
        dice: 平均Dice系数 (0-1范围)
    """
    # 确保输入是长整型
    pred = pred.long()
    target = target.long()
    
    # 处理实例分割情况（预测和目标实例数量不同）
    if pred.dim() == 3 and target.dim() == 3 and pred.shape[0] != target.shape[0]:
        logging.debug(f"检测到实例分割情况 - pred: {pred.shape[0]} instances, target: {target.shape[0]} instances")
        
        # 展平每个实例的掩码
        pred_flat = pred.view(pred.shape[0], -1)  # (N, H*W)
        target_flat = target.view(target.shape[0], -1)  # (M, H*W)
        
        # 计算所有实例对之间的Dice系数
        dice_matrix = torch.zeros((pred.shape[0], target.shape[0]), device=pred.device)
        
        for i in range(pred.shape[0]):
            for j in range(target.shape[0]):
                intersection = (pred_flat[i] & target_flat[j]).sum().float()
                total = pred_flat[i].sum().float() + target_flat[j].sum().float()
                dice_matrix[i, j] = (2.0 * intersection + smooth) / (total + smooth)
        
        # 使用匈牙利算法找到最优匹配
        if dice_matrix.shape[0] > 0 and dice_matrix.shape[1] > 0:
            from scipy.optimize import linear_sum_assignment
            dice_matrix_cpu = dice_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-dice_matrix_cpu)  # 最大化
            matched_dice = dice_matrix[row_ind, col_ind].mean()
            return matched_dice
        return torch.tensor(0.0, device=pred.device)
    
    # 语义分割情况
    total_dice = 0.0
    num_valid_classes = 0
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # 计算交集
        intersection = (pred_mask & target_mask).sum().float()
        # 计算并集
        total = pred_mask.sum().float() + target_mask.sum().float()
        
        # 只在类别存在时计算Dice
        if total > 0:
            dice = (2.0 * intersection + smooth) / (total + smooth)
            total_dice += dice
            num_valid_classes += 1
    
    # 返回平均Dice (0-1范围)
    if num_valid_classes == 0:
        return torch.tensor(0.0, device=pred.device)
    return total_dice / num_valid_classes

def calculate_pixel_accuracy(pred, target):
    """计算像素准确率
    
    参数:
        pred: 预测掩码 (H, W) 或 (N, H, W)
        target: 目标掩码 (H, W) 或 (N, H, W)
    """
    if pred.dim() == 3 and target.dim() == 3:
        # 实例分割情况
        correct = 0
        total = 0
        for p, t in zip(pred, target):
            correct += (p == t).float().sum()
            total += t.numel()
        return (correct / total).item()
    else:
        # 语义分割情况
        return (pred == target).float().mean().item()

class ConfusionMatrix:
    """混淆矩阵计算类"""
    
    def __init__(self, nc: int):
        """初始化混淆矩阵
        
        参数:
            nc (int): 类别数量
        """
        self.matrix = np.zeros((nc, nc))
        self.nc = nc
        self.conf = np.zeros((nc, nc))
        self.eps = 1e-6
    
    def update(self, preds, targets):
        """更新混淆矩阵
        
        参数:
            preds: 预测结果
            targets: 目标值
        """
        logging.debug(f"ConfusionMatrix.update - Input types: preds={type(preds)}, targets={type(targets)}")
        if isinstance(preds, torch.Tensor):
            logging.debug(f"Preds tensor: shape={preds.shape}, dtype={preds.dtype}, device={preds.device}")
            # 将布尔类型转换为整型
            if preds.dtype == torch.bool:
                preds = preds.long()
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            logging.debug(f"Targets tensor: shape={targets.shape}, dtype={targets.dtype}, device={targets.device}")
            # 将浮点类型转换为整型
            if targets.dtype == torch.float32:
                targets = targets.round().long()
            targets = targets.detach().cpu().numpy()
        
        # 确保输入是一维数组
        preds = preds.flatten()
        targets = targets.flatten()
        logging.debug(f"After flatten - preds: shape={preds.shape}, range=[{preds.min()}, {preds.max()}]")
        logging.debug(f"After flatten - targets: shape={targets.shape}, range=[{targets.min()}, {targets.max()}]")
        
        # 创建混淆矩阵
        try:
            # 确保值在有效范围内
            preds = np.clip(preds, 0, self.nc - 1).astype(np.int32)
            targets = np.clip(targets, 0, self.nc - 1).astype(np.int32)
            
            # 更新混淆矩阵
            for p, t in zip(preds, targets):
                self.matrix[p, t] += 1
            logging.debug("Matrix updated successfully")
        except Exception as e:
            logging.error(f"Error updating confusion matrix: {str(e)}")
            raise
    
    def compute(self):
        """计算各项指标
        
        返回:
            dict: 包含precision, recall, iou, f1和混淆矩阵
        """
        # 计算每个类别的TP, FP, FN
        TP = np.diag(self.matrix)
        FP = self.matrix.sum(0) - np.diag(self.matrix)
        FN = self.matrix.sum(1) - np.diag(self.matrix)
        
        # 计算指标
        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)
        iou = TP / (TP + FP + FN + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        return {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1,
            'matrix': self.matrix
        }

class SegmentationMetrics:
    """分割评估指标计算类"""
    
    def __init__(self, num_classes, save_dir=Path('.'), names=(), task_type='semantic'):
        """初始化
        
        参数:
            num_classes (int): 类别数量
            save_dir (Path): 结果保存目录
            names (tuple): 类别名称
            task_type (str): 任务类型，'semantic'或'instance'
        """
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.names = names
        self.task_type = task_type
        self.confusion_matrix = ConfusionMatrix(nc=num_classes)  # 创建混淆矩阵
        
        # 初始化指标字典
        self.metrics = {
            'pixel_accuracy': [],
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # 添加实例分割特有指标
        if task_type == 'instance':
            self.metrics.update({
                'AP50': [],
                'AP75': [],
                'mAP': [],
                'AR@1': [],
                'AR@10': [],
                'AR@100': []
            })
        
        logging.info(f"初始化{task_type}分割评估指标计算器:")
        logging.info(f"- 类别数: {num_classes}")
        logging.info(f"- 类别名称: {names}")
        logging.info(f"- 保存目录: {save_dir}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.reset()
    
    def update(self, pred, target):
        """更新指标
        
        参数:
            pred: 预测结果，可以是以下形式之一：
                - Tensor (N, H, W): 语义分割预测
                - Tensor (B, Q, H, W): 实例分割预测，其中Q是查询数
                - Dict: 包含pred_masks的字典
            target: 目标值 (N, H, W)
        """
        try:
            logging.debug(f"Metrics update - pred shape={pred.shape}, target shape={target.shape}")
            logging.debug(f"Initial devices: pred={pred.device}, target={target.device}")
            
            # 处理预测结果
            if isinstance(pred, dict):
                pred = pred['pred_masks']
            
            # 处理实例分割预测
            if pred.dim() == 4 and pred.shape[1] > 1:  # (B, Q, H, W) 格式
                logging.debug("检测到实例分割预测格式，进行后处理...")
                batch_size, num_queries, H, W = pred.shape
                
                # 重新组织所有候选实例
                all_candidates = pred.view(-1, H, W)  # (B*Q, H, W)
                logging.debug(f"重组后的候选实例形状: {all_candidates.shape}")
                
                # 计算所有候选的面积和置信度
                areas = all_candidates.sum(dim=(1,2))  # (B*Q,)
                confidences = all_candidates.max(dim=1)[0].max(dim=1)[0]  # (B*Q,)
                
                # 计算综合得分 (面积 * 置信度)
                scores = areas * confidences
                
                # 按得分排序
                sorted_indices = torch.argsort(scores, descending=True)
                selected_masks = []
                
                # 贪婪选择不重叠的掩码
                min_area = 5  # 最小面积阈值
                max_overlap = 0.95  # 重叠阈值
                max_instances = max(5, min(target.shape[0] + 8, 20))  # 最大实例数
                
                for idx in sorted_indices:
                    mask = all_candidates[idx] > 0.15  # 进一步降低二值化阈值
                    area = mask.sum()
                    
                    if area > min_area:
                        if len(selected_masks) == 0:
                            # 第一个掩码直接添加
                            selected_masks.append(mask.long())
                        else:
                            # 检查与已选掩码的重叠
                            valid_mask = True
                            for selected_mask in selected_masks:
                                intersection = (mask & selected_mask.bool()).sum().float()
                                min_area_current = min(area, selected_mask.sum().float())
                                overlap_ratio = intersection / min_area_current
                                
                                if overlap_ratio > max_overlap:
                                    valid_mask = False
                                    break
                            
                            if valid_mask:
                                selected_masks.append(mask.long())
                        
                        # 检查是否达到最大实例数
                        if len(selected_masks) >= max_instances:
                            break
                
                logging.debug(f"总共选择了 {len(selected_masks)} 个实例")
                
                if selected_masks:
                    pred = torch.stack(selected_masks)
                else:
                    # 如果没有有效掩码，添加一个空掩码
                    pred = torch.zeros((1, H, W), dtype=torch.long, device=pred.device)
                
                logging.debug(f"后处理完成 - 新的预测形状: {pred.shape}")
                
                # 如果是实例分割任务，计算实例分割特有指标
                if self.task_type == 'instance':
                    # 确保pred和target在同一设备上
                    if pred.device != target.device:
                        logging.debug(f"[DEBUG] 将目标从 {target.device} 移动到 {pred.device}")
                        target = target.to(pred.device)
                        
                    # 计算AP50
                    ap50 = self.calculate_ap(pred, target, 0.5)
                    self.metrics['AP50'].append(ap50)
                    
                    # 计算AP75
                    ap75 = self.calculate_ap(pred, target, 0.75)
                    self.metrics['AP75'].append(ap75)
                    
                    # 计算mAP
                    ap_sum = 0
                    for threshold in torch.linspace(0.5, 0.95, 10):
                        ap = self.calculate_ap(pred, target, threshold)
                        ap_sum += ap
                    self.metrics['mAP'].append(ap_sum / 10)
                    
                    # 计算AR
                    self.metrics['AR@1'].append(self.calculate_ar(pred, target, 1))
                    self.metrics['AR@10'].append(self.calculate_ar(pred, target, 10))
                    self.metrics['AR@100'].append(self.calculate_ar(pred, target, 100))
            
            # 确保target在正确的设备上
            if pred.device != target.device:
                target = target.to(pred.device)
            
            # 确保数据类型正确
            if pred.dtype == torch.bool:
                pred = pred.long()
            if target.dtype == torch.float32:
                target = target.round().long()
            
            logging.debug(f"After device sync and type conversion:")
            logging.debug(f"- pred: shape={pred.shape}, dtype={pred.dtype}, device={pred.device}")
            logging.debug(f"- target: shape={target.shape}, dtype={target.dtype}, device={target.device}")
            
            # 确保分离梯度
            pred = pred.detach()
            target = target.detach()
            
            # 确保目标值在有效范围内
            if target.max() >= self.num_classes:
                logging.warning(f"Found target values {target.max().item()} outside expected range [0, {self.num_classes-1}]")
            
            logging.debug(f"Target range: [{target.min().item()}, {target.max().item()}]")
            
            # 更新混淆矩阵
            self.confusion_matrix.update(pred, target)
            
            # 计算基础指标
            pixel_acc = calculate_pixel_accuracy(pred, target)
            iou, mean_iou = calculate_iou(pred, target, num_classes=self.num_classes)
            dice = calculate_dice(pred, target, num_classes=self.num_classes)
            
            logging.debug(f"Calculated metrics - PA: {pixel_acc}, IoU: {mean_iou}, Dice: {dice}")
            
            # 更新指标
            self.metrics['pixel_accuracy'].append(float(pixel_acc))
            self.metrics['iou'].append(float(mean_iou))
            self.metrics['dice'].append(float(dice))
            
        except Exception as e:
            logging.error(f"Error in metrics update: {str(e)}")
            raise
            
    def calculate_ap(self, pred_masks, target_masks, iou_threshold):
        """计算指定IoU阈值下的AP"""
        # 确保所有张量在同一设备上
        device = pred_masks.device
        if target_masks.device != device:
            logging.debug(f"[DEBUG] 将目标掩码从 {target_masks.device} 移动到 {device}")
            target_masks = target_masks.to(device)
            
        # 创建一个简单的置信度分数 - 对于已选择的掩码，我们假设它们都有高置信度
        pred_scores = torch.ones(len(pred_masks), device=device)
        
        logging.debug(f"\n[DEBUG] 计算AP@{iou_threshold:.2f}")
        logging.debug(f"[DEBUG] 预测掩码: 数量={len(pred_masks)}, 形状={pred_masks.shape}, 设备={pred_masks.device}")
        logging.debug(f"[DEBUG] 目标掩码: 数量={len(target_masks)}, 形状={target_masks.shape}, 设备={target_masks.device}")
        
        # 计算IoU矩阵
        ious = torch.zeros((len(pred_masks), len(target_masks)), device=device)
        for i, pred in enumerate(pred_masks):
            for j, target in enumerate(target_masks):
                # 确保掩码是布尔类型
                pred_bool = pred.bool() if pred.dtype != torch.bool else pred
                target_bool = target.bool() if target.dtype != torch.bool else target
                
                intersection = (pred_bool & target_bool).float().sum()
                union = (pred_bool | target_bool).float().sum()
                ious[i, j] = intersection / (union + 1e-6)
        
        logging.debug(f"[DEBUG] IoU矩阵: 形状={ious.shape}, 范围=[{ious.min():.3f}, {ious.max():.3f}]")
        if len(pred_masks) > 0 and len(target_masks) > 0:
            logging.debug(f"[DEBUG] IoU矩阵平均值: {ious.mean():.3f}")
        
        # 计算AP
        tp = torch.zeros(len(pred_masks), device=device)
        fp = torch.zeros(len(pred_masks), device=device)
        matched_targets = set()
        
        for i in range(len(pred_masks)):
            # 找到最大IoU的目标
            if len(target_masks) > 0:
                max_iou, max_idx = torch.max(ious[i], dim=0)
                if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                    tp[i] = 1
                    matched_targets.add(max_idx.item())
                    logging.debug(f"[DEBUG] 预测{i}匹配到目标{max_idx.item()}, IoU={max_iou:.3f} (+)")
                else:
                    fp[i] = 1
                    if max_iou < iou_threshold:
                        logging.debug(f"[DEBUG] 预测{i}的最大IoU={max_iou:.3f} < 阈值{iou_threshold:.2f} (-)")
                    else:
                        logging.debug(f"[DEBUG] 预测{i}的最佳匹配目标{max_idx.item()}已被匹配 (-)")
            else:
                fp[i] = 1
                logging.debug(f"[DEBUG] 预测{i}没有可匹配的目标 (-)")
        
        logging.debug(f"[DEBUG] 真阳性(TP)数量: {tp.sum().item()}")
        logging.debug(f"[DEBUG] 假阳性(FP)数量: {fp.sum().item()}")
        
        # 计算precision和recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        if len(target_masks) > 0:
            recalls = tp_cumsum / len(target_masks)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            logging.debug(f"[DEBUG] Precision范围: [{precisions.min():.3f}, {precisions.max():.3f}]")
            logging.debug(f"[DEBUG] Recall范围: [{recalls.min():.3f}, {recalls.max():.3f}]")
            
            # 计算AP (使用简化的方法)
            ap = 0
            for t in torch.linspace(0, 1, 11, device=device):
                recall_mask = recalls >= t
                if recall_mask.any():
                    ap += torch.max(precisions * recall_mask.float())
            ap = ap / 11
        else:
            ap = torch.tensor(0.0, device=device)
            logging.debug("[DEBUG] 没有目标掩码，AP=0")
            
        logging.debug(f"[DEBUG] 最终AP@{iou_threshold:.2f}: {ap.item():.4f}")
        return ap.item()
    
    def calculate_ar(self, pred_masks, target_masks, max_dets):
        """计算平均召回率"""
        # 确保所有张量在同一设备上
        device = pred_masks.device
        if target_masks.device != device:
            logging.debug(f"[DEBUG] 将目标掩码从 {target_masks.device} 移动到 {device}")
            target_masks = target_masks.to(device)
            
        pred_masks = pred_masks[:max_dets]
        total_recall = 0.0  # 使用Python浮点数而不是张量
        
        logging.debug(f"\n[DEBUG] 计算AR@{max_dets}")
        logging.debug(f"[DEBUG] 预测掩码: 数量={len(pred_masks)}, 形状={pred_masks.shape}, 设备={pred_masks.device}")
        logging.debug(f"[DEBUG] 目标掩码: 数量={len(target_masks)}, 形状={target_masks.shape}, 设备={target_masks.device}")
        
        if len(target_masks) > 0:
            for i, target in enumerate(target_masks):
                # 计算与目标的最大IoU
                max_iou = 0.0  # 使用Python浮点数
                best_pred_idx = -1
                
                for j, pred in enumerate(pred_masks):
                    # 确保掩码是布尔类型
                    pred_bool = pred.bool() if pred.dtype != torch.bool else pred
                    target_bool = target.bool() if target.dtype != torch.bool else target
                    
                    intersection = (pred_bool & target_bool).float().sum().item()  # 转换为Python浮点数
                    union = (pred_bool | target_bool).float().sum().item()  # 转换为Python浮点数
                    iou = intersection / (union + 1e-6)
                    if iou > max_iou:
                        max_iou = iou
                        best_pred_idx = j
                
                # 使用Python布尔值和浮点数
                is_detected = max_iou >= 0.5
                total_recall += 1.0 if is_detected else 0.0
                
                if is_detected:
                    logging.debug(f"[DEBUG] 目标{i}被预测{best_pred_idx}检测到, IoU={max_iou:.3f} (+)")
                else:
                    logging.debug(f"[DEBUG] 目标{i}未被检测到, 最大IoU={max_iou:.3f} (-)")
            
            ar = total_recall / len(target_masks)
            logging.debug(f"[DEBUG] 检测到的目标数量: {total_recall}/{len(target_masks)}")
        else:
            ar = 0.0  # 使用Python浮点数
            logging.debug("[DEBUG] 没有目标掩码，AR=0")
            
        logging.debug(f"[DEBUG] 最终AR@{max_dets}: {ar:.4f}")
        return ar
    
    def compute(self):
        """计算所有指标的平均值"""
        results = {}
        
        # 计算基础指标的平均值
        for k, v in self.metrics.items():
            if k not in ['precision', 'recall', 'f1'] and v:  # 排除precision和recall
                # 确保所有元素都是CPU上的标量值
                cpu_values = []
                for val in v:
                    if isinstance(val, torch.Tensor):
                        cpu_values.append(val.detach().cpu().item())
                    else:
                        cpu_values.append(val)
                results[k] = float(np.mean(cpu_values))  # 确保结果是Python float
            else:
                results[k] = 0.0
        
        # 计算混淆矩阵相关指标
        cm_metrics = self.confusion_matrix.compute()
        results.update({
            'mean_precision': float(np.mean(cm_metrics['precision'])),
            'mean_iou': float(np.mean(cm_metrics['iou']))
        })
        
        return results
        
    def reset(self):
        """重置所有指标"""
        self.confusion_matrix = ConfusionMatrix(self.num_classes)
        for k in self.metrics:
            self.metrics[k] = []

class MetricCalculator:
    def __init__(self, model_type='detr', num_classes=21, task_type='semantic'):
        self.model_type = model_type
        self.task_type = task_type
        self.num_classes = num_classes
        self.confusion_matrix = ConfusionMatrix(num_classes)
        logging.info(f"初始化MetricCalculator - 模型类型: {model_type}, 任务类型: {task_type}, 类别数: {num_classes}")
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        self.total_iou = 0.0
        self.total_pixel_accuracy = 0.0
        self.total_dice = 0.0
        self.total_precision = 0.0
        self.num_samples = 0
        
        # 实例分割指标
        if self.task_type == 'instance':
            self.total_ap50 = 0.0  # AP@IoU=0.5
            self.total_ap75 = 0.0  # AP@IoU=0.75
            self.total_map = 0.0   # mAP@[0.5:0.95]
            self.total_ar1 = 0.0   # AR@1
            self.total_ar10 = 0.0  # AR@10
            self.total_ar100 = 0.0 # AR@100
            
        # 全景分割指标
        elif self.task_type == 'panoptic':
            self.total_pq = 0.0    # Panoptic Quality
            self.total_sq = 0.0    # Segmentation Quality
            self.total_rq = 0.0    # Recognition Quality
            self.total_stuff_iou = 0.0  # stuff类别的mIoU
            self.total_thing_map = 0.0  # thing类别的mAP
        
    def calculate_ap(self, pred_masks, pred_scores, target_masks, iou_threshold):
        """计算指定IoU阈值下的AP"""
        logging.debug(f"\n计算AP - IoU阈值: {iou_threshold}")
        logging.debug(f"预测掩码: shape={pred_masks.shape}, 范围=[{pred_masks.min():.3f}, {pred_masks.max():.3f}]")
        logging.debug(f"预测分数: shape={pred_scores.shape}, 范围=[{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
        logging.debug(f"目标掩码: shape={target_masks.shape}, 范围=[{target_masks.min():.3f}, {target_masks.max():.3f}]")
        
        # 对预测掩码按置信度排序
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_masks = pred_masks[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        device = pred_scores.device
        # 计算IoU矩阵
        ious = torch.zeros((len(pred_masks), len(target_masks)), device=device)
        for i, pred in enumerate(pred_masks):
            for j, target in enumerate(target_masks):
                # 确保掩码是布尔类型
                pred_bool = pred.bool() if pred.dtype != torch.bool else pred
                target_bool = target.bool() if target.dtype != torch.bool else target
                
                intersection = (pred_bool & target_bool).float().sum()
                union = (pred_bool | target_bool).float().sum()
                ious[i, j] = intersection / (union + 1e-6)
        
        logging.debug(f"IoU矩阵: shape={ious.shape}, 范围=[{ious.min():.3f}, {ious.max():.3f}]")
        
        # 计算AP
        tp = torch.zeros(len(pred_masks))
        fp = torch.zeros(len(pred_masks))
        matched_targets = set()
        
        for i, pred_idx in enumerate(range(len(pred_masks))):
            max_iou, max_idx = torch.max(ious[pred_idx], dim=0)
            if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                tp[i] = 1
                matched_targets.add(max_idx.item())
            else:
                fp[i] = 1
        
        # 计算precision和recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / len(target_masks)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        logging.debug(f"TP累计和: {tp_cumsum}")
        logging.debug(f"FP累计和: {fp_cumsum}")
        logging.debug(f"Recall范围: [{recalls.min():.3f}, {recalls.max():.3f}]")
        logging.debug(f"Precision范围: [{precisions.min():.3f}, {precisions.max():.3f}]")
        
        # 计算AP
        ap = torch.trapz(precisions, recalls)
        logging.debug(f"计算得到AP: {ap.item():.4f}")
        return ap.item()
    
    def calculate_ar(self, pred_masks, target_masks, max_dets):
        """计算平均召回率"""
        logging.debug(f"\n计算AR@{max_dets}")
        logging.debug(f"预测掩码: shape={pred_masks.shape}")
        logging.debug(f"目标掩码: shape={target_masks.shape}")
        
        pred_masks = pred_masks[:max_dets]
        total_recall = 0
        
        logging.debug(f"\n[DEBUG] 计算AR@{max_dets}")
        logging.debug(f"[DEBUG] 预测掩码: 数量={len(pred_masks)}, 形状={pred_masks.shape}")
        logging.debug(f"[DEBUG] 目标掩码: 数量={len(target_masks)}, 形状={target_masks.shape}")
        
        if len(target_masks) > 0:
            for i, target in enumerate(target_masks):
                # 计算与目标的最大IoU
                max_iou = 0
                for j, pred in enumerate(pred_masks):
                    # 确保掩码是布尔类型
                    pred_bool = pred.bool() if pred.dtype != torch.bool else pred
                    target_bool = target.bool() if target.dtype != torch.bool else target
                    
                    intersection = (pred_bool & target_bool).float().sum()
                    union = (pred_bool | target_bool).float().sum()
                    iou = intersection / (union + 1e-6)
                    max_iou = max(max_iou, iou)
                total_recall += (max_iou >= 0.5).float()
                logging.debug(f"目标 {i} - 最大IoU: {max_iou:.3f}")
            
            ar = total_recall / len(target_masks)
            logging.debug(f"[DEBUG] 检测到的目标数量: {total_recall.item()}/{len(target_masks)}")
        else:
            ar = torch.tensor(0.0, device=pred_masks.device)
            logging.debug("[DEBUG] 没有目标掩码，AR=0")
            
        logging.debug(f"[DEBUG] 最终AR@{max_dets}: {ar.item():.4f}")
        return ar.item()
    
    def calculate_panoptic_metrics(self, pred_stuff, pred_things, target_stuff, target_things):
        """计算全景分割指标"""
        logging.debug("\n计算全景分割指标")
        logging.debug(f"预测stuff掩码: shape={pred_stuff.shape}")
        logging.debug(f"预测things掩码: shape={pred_things.shape}")
        logging.debug(f"目标stuff掩码: shape={target_stuff.shape}")
        logging.debug(f"目标things掩码: shape={target_things.shape}")
        
        # 计算stuff类别的mIoU
        stuff_iou, _ = calculate_iou(pred_stuff, target_stuff, self.num_classes)
        logging.debug(f"Stuff mIoU: {stuff_iou:.4f}")
        
        # 计算thing类别的指标
        thing_map = self.calculate_ap(pred_things, torch.ones(len(pred_things)), target_things, 0.5)
        logging.debug(f"Thing mAP: {thing_map:.4f}")
        
        # 计算PQ指标
        matched_pairs = []
        unmatched_pred = []
        unmatched_target = []
        
        # 匹配预测和目标实例
        for i, pred in enumerate(pred_things):
            max_iou = 0
            max_idx = -1
            for j, target in enumerate(target_things):
                intersection = (pred & target).float().sum()
                union = (pred | target).float().sum()
                iou = intersection / (union + 1e-6)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= 0.5:
                matched_pairs.append((i, max_idx, max_iou))
                logging.debug(f"匹配对 {i}->{max_idx}, IoU: {max_iou:.4f}")
            else:
                unmatched_pred.append(i)
                logging.debug(f"未匹配的预测 {i}, 最大IoU: {max_iou:.4f}")
        
        # 计算PQ组件
        if len(matched_pairs) > 0:
            sq = sum(iou for _, _, iou in matched_pairs) / len(matched_pairs)
            rq = len(matched_pairs) / (len(matched_pairs) + len(unmatched_pred) + len(unmatched_target))
            pq = sq * rq
            logging.debug(f"SQ: {sq:.4f}, RQ: {rq:.4f}, PQ: {pq:.4f}")
        else:
            sq = pq = rq = 0
            logging.debug("没有匹配的实例对，指标均为0")
            
        return stuff_iou, thing_map, pq, sq, rq
        
    def update(self, outputs, targets):
        """更新指标"""
        logging.debug(f"\n更新{self.task_type}分割指标")
        
        if self.task_type == 'semantic':
            if isinstance(outputs, dict):
                pred_masks = outputs['pred_masks']
            else:
                pred_masks = outputs
            
            logging.debug(f"语义分割预测掩码: shape={pred_masks.shape}")
            
            if pred_masks.dim() == 4:  # (B, C, H, W)
                pred_masks = pred_masks.argmax(dim=1)  # (B, H, W)
            
            batch_size = pred_masks.shape[0]
            self.num_samples += batch_size
            logging.debug(f"处理批次大小: {batch_size}, 累计样本数: {self.num_samples}")
            
            # 计算每个样本的指标
            for i in range(batch_size):
                pred = pred_masks[i]  # (H, W)
                target = targets[i]  # (H, W)
                
                logging.debug(f"\n样本 {i}:")
                pixel_acc = calculate_pixel_accuracy(pred, target)
                iou, mean_iou = calculate_iou(pred, target, num_classes=self.num_classes)
                dice = calculate_dice(pred, target, self.num_classes)
                
                logging.debug(f"像素准确率: {pixel_acc:.4f}")
                logging.debug(f"IoU: {mean_iou:.4f}")
                logging.debug(f"Dice: {dice:.4f}")
                
                self.total_pixel_accuracy += pixel_acc
                self.total_iou += mean_iou
                self.total_dice += dice
                
        elif self.task_type == 'instance':
            pred_masks = outputs['pred_masks']  # (B, N, H, W)
            pred_logits = outputs['pred_logits']  # (B, N, C)
            
            logging.debug(f"实例分割预测掩码: shape={pred_masks.shape}")
            logging.debug(f"实例分割预测logits: shape={pred_logits.shape}")
            
            batch_size = pred_masks.shape[0]
            self.num_samples += batch_size
            
            for i in range(batch_size):
                logging.debug(f"\n处理批次 {i}:")
                
                # 获取当前样本的预测和目标
                cur_pred_masks = pred_masks[i]  # (N, H, W)
                cur_pred_logits = pred_logits[i]  # (N, C)
                cur_target = targets[i]
                
                if len(cur_target['labels']) == 0:
                    logging.debug("跳过空目标样本")
                    continue
                
                # 将预测掩码转换为二值掩码
                pred_probs = cur_pred_masks.sigmoid()
                pred_binary = (pred_probs > 0.5).float()
                
                # 获取目标掩码
                target_masks = cur_target['masks']  # (M, H, W)
                
                logging.debug(f"预测掩码数量: {len(pred_binary)}")
                logging.debug(f"目标掩码数量: {len(target_masks)}")
                
                # 调整大小以匹配
                if pred_binary.shape[-2:] != target_masks.shape[-2:]:
                    pred_binary = F.interpolate(
                        pred_binary.unsqueeze(0),
                        size=target_masks.shape[-2:],
                        mode='nearest'
                    ).squeeze(0)
                
                # 计算基础指标
                pixel_acc = calculate_pixel_accuracy(pred_binary, target_masks)
                iou, mean_iou = calculate_iou(pred_binary, target_masks, num_classes=2)
                dice = calculate_dice(pred_binary, target_masks, num_classes=2)
                
                logging.debug(f"基础指标 - PA: {pixel_acc:.4f}, IoU: {mean_iou:.4f}, Dice: {dice:.4f}")
                
                self.total_pixel_accuracy += pixel_acc
                self.total_iou += mean_iou
                self.total_dice += dice
                
                # 计算AP指标
                pred_scores = pred_logits.softmax(-1)[:, :-1].max(-1)[0]  # 获取最高类别概率
                logging.debug(f"预测分数范围: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
                
                self.total_ap50 += self.calculate_ap(pred_binary, pred_scores, target_masks, 0.5)
                self.total_ap75 += self.calculate_ap(pred_binary, pred_scores, target_masks, 0.75)
                
                # 计算mAP
                ap_sum = 0
                for threshold in torch.linspace(0.5, 0.95, 10):
                    ap = self.calculate_ap(pred_binary, pred_scores, target_masks, threshold)
                    ap_sum += ap
                    logging.debug(f"AP@{threshold:.2f}: {ap:.4f}")
                self.total_map += ap_sum / 10
                
                # 计算AR
                self.total_ar1 += self.calculate_ar(pred_binary, target_masks, 1)
                self.total_ar10 += self.calculate_ar(pred_binary, target_masks, 10)
                self.total_ar100 += self.calculate_ar(pred_binary, target_masks, 100)
                
        else:  # panoptic
            pred_stuff = outputs['pred_stuff']  # (B, H, W)
            pred_things = outputs['pred_things']  # (B, N, H, W)
            
            logging.debug(f"全景分割预测 - Stuff: shape={pred_stuff.shape}")
            logging.debug(f"全景分割预测 - Things: shape={pred_things.shape}")
            
            batch_size = pred_stuff.shape[0]
            self.num_samples += batch_size
            
            for i in range(batch_size):
                logging.debug(f"\n处理批次 {i}:")
                
                # 获取当前样本的预测和目标
                cur_pred_stuff = pred_stuff[i]
                cur_pred_things = pred_things[i]
                cur_target = targets[i]
                
                # 计算全景分割指标
                stuff_iou, thing_map, pq, sq, rq = self.calculate_panoptic_metrics(
                    cur_pred_stuff,
                    cur_pred_things,
                    cur_target['stuff_mask'],
                    cur_target['thing_masks']
                )
                
                self.total_stuff_iou += stuff_iou
                self.total_thing_map += thing_map
                self.total_pq += pq
                self.total_sq += sq
                self.total_rq += rq
                
                logging.debug(f"批次 {i} 指标:")
                logging.debug(f"Stuff mIoU: {stuff_iou:.4f}")
                logging.debug(f"Thing mAP: {thing_map:.4f}")
                logging.debug(f"PQ: {pq:.4f}, SQ: {sq:.4f}, RQ: {rq:.4f}")
    
    def compute(self):
        """计算平均指标"""
        if self.num_samples == 0:
            logging.warning("没有样本进行评估")
            return {
                'mIoU': 0.0,
                'pixel_accuracy': 0.0,
                'dice': 0.0,
                'precision': 0.0
            }
        
        logging.debug(f"\n计算{self.task_type}分割的平均指标")
        logging.debug(f"样本总数: {self.num_samples}")
        
        # 从混淆矩阵计算精确率
        cm_metrics = self.confusion_matrix.compute()
        
        # 基础指标
        metrics = {
            'mIoU': self.total_iou / self.num_samples,
            'pixel_accuracy': self.total_pixel_accuracy / self.num_samples,
            'dice': self.total_dice / self.num_samples,
            'precision': float(np.mean(cm_metrics['precision']))
        }
        
        logging.debug("\n基础指标:")
        for k, v in metrics.items():
            logging.debug(f"{k}: {v:.4f}")
        
        # 添加特定任务的指标
        if self.task_type == 'instance':
            instance_metrics = {
                'AP50': self.total_ap50 / self.num_samples,
                'AP75': self.total_ap75 / self.num_samples,
                'mAP': self.total_map / self.num_samples,
                'AR@1': self.total_ar1 / self.num_samples,
                'AR@10': self.total_ar10 / self.num_samples,
                'AR@100': self.total_ar100 / self.num_samples
            }
            metrics.update(instance_metrics)
            
            logging.debug("\n实例分割特有指标:")
            for k, v in instance_metrics.items():
                logging.debug(f"{k}: {v:.4f}")
                
        elif self.task_type == 'panoptic':
            panoptic_metrics = {
                'PQ': self.total_pq / self.num_samples,
                'SQ': self.total_sq / self.num_samples,
                'RQ': self.total_rq / self.num_samples,
                'stuff_mIoU': self.total_stuff_iou / self.num_samples,
                'thing_mAP': self.total_thing_map / self.num_samples
            }
            metrics.update(panoptic_metrics)
            
            logging.debug("\n全景分割特有指标:")
            for k, v in panoptic_metrics.items():
                logging.debug(f"{k}: {v:.4f}")
        
        return metrics


