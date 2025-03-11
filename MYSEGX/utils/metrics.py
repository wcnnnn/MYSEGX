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
    """计算IoU (Intersection over Union)
    
    参数:
        pred: 预测结果 (N, H, W)
        target: 目标值 (N, H, W)
        num_classes: 类别数量
        
    返回:
        iou: 平均IoU值
    """
    # 确保输入是长整型
    pred = pred.long()
    target = target.long()
    
    # 计算每个类别的IoU
    total_iou = 0.0
    num_valid_classes = 0
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # 计算交集和并集
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        # 只在类别存在时计算IoU
        if union > 0:
            iou = (intersection / union) * 100.0
            total_iou += iou
            num_valid_classes += 1
    
    # 返回平均IoU
    if num_valid_classes == 0:
        return torch.tensor(0.0, device=pred.device)
    return total_iou / num_valid_classes

def calculate_dice(pred, target, num_classes=21, smooth=1e-6):
    """计算Dice系数
    
    参数:
        pred: 预测掩码 (N, H, W)
        target: 目标掩码 (N, H, W)
        num_classes: 类别数量
        smooth: 平滑项
        
    返回:
        dice: 平均Dice系数 (0-100范围)
    """
    # 确保输入是长整型
    pred = pred.long()
    target = target.long()
    
    # 计算每个类别的Dice系数
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
            dice = ((2.0 * intersection + smooth) / (total + smooth)) * 100.0
            total_dice += dice
            num_valid_classes += 1
    
    # 返回平均Dice
    if num_valid_classes == 0:
        return torch.tensor(0.0, device=pred.device)
    return total_dice / num_valid_classes

def calculate_pixel_accuracy(pred, target):
    """计算像素级准确率
    
    参数:
        pred: 预测掩码，可以是以下格式之一:
            - Tensor shape (N, H, W): 二值掩码
            - Tensor shape (N, C, H, W): 多类别掩码
            - dict: 包含'pred_masks'键的字典
        target: 目标掩码, shape (N, H, W)
        
    返回:
        accuracy (float): 像素准确率 (0-100范围)
    """
    if isinstance(pred, dict):
        pred = pred['pred_masks']
    
    # 处理多类别掩码
    if pred.dim() == 4 and pred.shape[1] > 1:  # (N, C, H, W)
        pred = pred.argmax(dim=1)  # (N, H, W)
    elif pred.dim() == 4 and pred.shape[1] == 1:  # 单通道 (N, 1, H, W)
        pred = pred.squeeze(1)  # 移除通道维度 (N, H, W)
    
    # 确保在比较前两个张量形状相同
    if pred.shape != target.shape:
        # 如果是DETR格式，可能需要调整大小
        if pred.dim() > target.dim():
            # 如果预测有额外的维度，尝试压缩它
            while pred.dim() > target.dim():
                pred = pred.squeeze(1)
        elif target.dim() > pred.dim():
            # 如果目标有额外的维度，尝试压缩它
            while target.dim() > pred.dim():
                target = target.squeeze(1)
        
        # 如果空间维度不同，将预测调整为目标的大小
        if pred.shape != target.shape:
            # 获取目标的空间维度
            target_spatial_dims = target.shape[-2:]
            
            # 调整预测的空间维度
            if pred.dim() == 3:  # (N, H, W)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1).float(),  # 添加通道维度 (N, 1, H, W)
                    size=target_spatial_dims,
                    mode='nearest'
                ).squeeze(1)  # 移除通道维度 (N, H, W)
            elif pred.dim() == 4:  # (N, C, H, W)
                pred = torch.nn.functional.interpolate(
                    pred.float(),
                    size=target_spatial_dims,
                    mode='nearest'
                )
                pred = pred.argmax(dim=1)  # (N, H, W)
    
    # 转换为整数类型进行比较
    pred = pred.long()
    target = target.long()
    
    # 计算准确率并转换为百分比 (0-100范围)
    correct = (pred == target).sum().float()
    total = torch.numel(target)
    accuracy = (correct / total) * 100.0
    
    # 确保结果在0-100范围内
    accuracy = torch.clamp(accuracy, 0.0, 100.0).item()
    
    return accuracy

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
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            logging.debug(f"Targets tensor: shape={targets.shape}, dtype={targets.dtype}, device={targets.device}")
            targets = targets.detach().cpu().numpy()
        
        # 确保输入是一维数组
        preds = preds.flatten()
        targets = targets.flatten()
        logging.debug(f"After flatten - preds: shape={preds.shape}, range=[{preds.min()}, {preds.max()}]")
        logging.debug(f"After flatten - targets: shape={targets.shape}, range=[{targets.min()}, {targets.max()}]")
        
        # 创建混淆矩阵
        # 对于二值分割，确保值为0或1
        if self.nc == 2:
            preds = np.round(preds).astype(np.int32)
            targets = np.round(targets).astype(np.int32)
        
        # 对于多类别分割，确保值在有效范围内
        mask = (0 <= targets) & (targets < self.nc)
        if not np.all(mask):
            logging.warning(f"Found {(~mask).sum()} invalid target indices outside range [0, {self.nc})")
            # 过滤掉无效的目标索引
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                logging.warning("No valid targets found, skipping update")
                return  # 没有有效的目标，直接返回
            preds = preds[valid_indices]
            targets = targets[valid_indices]
            logging.debug(f"After filtering - shapes: preds={preds.shape}, targets={targets.shape}")
        
        # 更新混淆矩阵
        try:
            for p, t in zip(preds, targets):
                # 确保索引是整数且在有效范围内
                p_idx = min(max(int(round(p)), 0), self.nc - 1)
                t_idx = min(max(int(round(t)), 0), self.nc - 1)
                if p_idx != p or t_idx != t:
                    logging.warning(f"Value clamped - pred: {p}->{p_idx}, target: {t}->{t_idx}")
                self.matrix[p_idx, t_idx] += 1
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

    def plot(self, names=None, save_dir='', show=True):
        """绘制混淆矩阵
        
        参数:
            names: 类别名称
            save_dir: 保存路径
            show: 是否显示
        """
        import seaborn as sn
        
        array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + self.eps)  # normalize
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.heatmap(array, 
                  annot=self.nc < 30,
                  annot_kws={"size": 8},
                  cmap='Blues', 
                  fmt='.2f', 
                  square=True,
                  xticklabels=names,
                  yticklabels=names)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix')
        
        if save_dir:
            plt.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        if show:
            plt.show()
        plt.close()

class SegmentationMetrics:
    """分割评价指标计算类"""
    
    def __init__(self, num_classes, save_dir=Path('.'), names=()):
        """初始化
        
        参数:
            num_classes: 类别数量
            save_dir: 结果保存路径
            names: 类别名称
        """
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.names = names
        self.confusion_matrix = ConfusionMatrix(num_classes)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'pixel_accuracy': [],
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def update(self, pred, target):
        """更新指标
        
        参数:
            pred: 预测结果 (N, H, W) 或 (N, C, H, W)
            target: 目标值 (N, H, W)
        """
        try:
            logging.debug(f"Metrics update - pred shape={pred.shape}, target shape={target.shape}")
            logging.debug(f"Initial devices: pred={pred.device}, target={target.device}")
            
            # 处理预测掩码
            if pred.dim() == 4:  # (N, C, H, W)
                pred = pred.argmax(dim=1)  # 转换为 (N, H, W)
            
            # 确保target是3D (N, H, W)
            if target.dim() != 3:
                target = target.view(pred.shape)
            
            # 检查设备并统一
            if pred.device != target.device:
                target = target.to(pred.device)
            
            logging.debug(f"After device sync: pred={pred.device}, target={target.device}")
            
            # 确保分离梯度
            pred = pred.detach()
            target = target.detach()
            
            # 确保目标值在有效范围内
            if target.max() >= self.num_classes:
                logging.warning(f"Found target values outside valid range [0, {self.num_classes-1}]. Clamping values.")
                target = torch.clamp(target, 0, self.num_classes - 1)
            
            logging.debug(f"Target range after clamping: [{target.min().item()}, {target.max().item()}]")
            
            # 更新混淆矩阵
            self.confusion_matrix.update(pred, target)
            
            # 计算基础指标
            pixel_acc = calculate_pixel_accuracy(pred, target)
            iou = calculate_iou(pred, target, num_classes=self.num_classes)
            dice = calculate_dice(pred, target, num_classes=self.num_classes)
            
            logging.debug(f"Calculated metrics - PA: {pixel_acc}, IoU: {iou}, Dice: {dice}")
            
            # 更新指标
            self.metrics['pixel_accuracy'].append(float(pixel_acc))
            self.metrics['iou'].append(float(iou))
            self.metrics['dice'].append(float(dice))
            
        except Exception as e:
            logging.error(f"Error in metrics update: {str(e)}")
            raise
    
    def compute(self):
        """计算所有指标的平均值"""
        results = {}
        
        # 计算基础指标的平均值
        for k, v in self.metrics.items():
            if v:  # 只在列表非空时计算平均值
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
            'mean_recall': float(np.mean(cm_metrics['recall'])),
            'mean_iou': float(np.mean(cm_metrics['iou'])),
            'mean_f1': float(np.mean(cm_metrics['f1']))
        })
        
        return results
    
    def plot(self):
        """绘制评价指标相关的图表"""
        # 绘制混淆矩阵
        self.confusion_matrix.plot(
            names=self.names,
            save_dir=str(self.save_dir),
            show=False
        )
        
        # 这里可以添加其他可视化，如PR曲线等
        
    def reset(self):
        """重置所有指标"""
        self.confusion_matrix = ConfusionMatrix(self.num_classes)
        for k in self.metrics:
            self.metrics[k] = []

class MetricCalculator:
    def __init__(self, model_type='detr', num_classes=21):
        self.model_type = model_type
        self.num_classes = num_classes
        self.confusion_matrix = ConfusionMatrix(num_classes)
        self.reset()
        self.debug_times = {'data_prep': 0, 'mask_proc': 0, 'metric_calc': 0}
        
    def reset(self):
        """重置所有指标"""
        self.total_accuracy = 0.0
        self.total_iou = 0.0
        self.total_dice = 0.0
        self.total_pixel_accuracy = 0.0
        self.num_samples = 0
        self.debug_times = {'data_prep': 0, 'mask_proc': 0, 'metric_calc': 0}
        
    def update(self, outputs, targets):
        t_start = time()
        
        if self.model_type == 'detr':
            logger = logging.getLogger(__name__)
            logger.info(f"Processing DETR batch with shapes - outputs: {outputs['pred_masks'].shape}, targets: {len(targets)}")
            pred_logits = outputs['pred_logits']
            pred_masks = outputs['pred_masks']
            
            batch_size = pred_logits.shape[0]
            self.num_samples += batch_size
            
            t_data = time()
            self.debug_times['data_prep'] += t_data - t_start
            
            for i in range(batch_size):
                logger.debug(f"Processing sample {i}/{batch_size}")
                # 分类准确率
                pred_classes = pred_logits[i].argmax(dim=1)
                target_classes = torch.full_like(pred_classes, pred_logits.shape[-1]-1)
                target = targets[i]
                target_classes[:len(target['labels'])] = target['labels']
                self.total_accuracy += (pred_classes == target_classes).float().mean().item() * 100
                
                # 掩码处理和评估
                t_mask_start = time()
                pred_mask = pred_masks[i]
                target_masks = F.interpolate(target['masks'].unsqueeze(0),
                                           size=pred_mask.shape[-2:],
                                           mode='bilinear',
                                           align_corners=False)[0]
                
                logger.debug(f"Mask shapes - pred: {pred_mask.shape}, target: {target_masks.shape}")
                
                # 优化掩码处理
                pred_probs = pred_mask.sigmoid()
                pred_mask = (pred_probs > 0.5).float()
                
                t_mask_end = time()
                self.debug_times['mask_proc'] += t_mask_end - t_mask_start
                
                # 更新混淆矩阵
                t_metric_start = time()
                pred_mask_cpu = pred_mask.cpu()
                target_masks_cpu = target_masks.cpu()
                
                for c in range(pred_mask.shape[0]):
                    pred_binary = (pred_mask[c] > 0.5).float().cpu().numpy()
                    target_binary = (target_masks[c] > 0.5).float().cpu().numpy()
                    self.confusion_matrix.update(pred_binary.flatten(), target_binary.flatten())
                
                # 计算基础指标
                iou = calculate_iou(pred_mask, target_masks, num_classes=self.num_classes)
                dice = calculate_dice(pred_mask, target_masks, num_classes=self.num_classes)
                pixel_acc = calculate_pixel_accuracy(pred_mask, target_masks)
                
                self.total_iou += iou * 100
                self.total_dice += dice * 100
                self.total_pixel_accuracy += pixel_acc * 100
                
                t_metric_end = time()
                self.debug_times['metric_calc'] += t_metric_end - t_metric_start
                
                if i % 10 == 0:  # 每10个样本记录一次时间统计
                    logger.info(f"Time stats for {i}th sample - Data prep: {self.debug_times['data_prep']:.3f}s, "
                              f"Mask processing: {self.debug_times['mask_proc']:.3f}s, "
                              f"Metric calculation: {self.debug_times['metric_calc']:.3f}s")
        else:
            logger = logging.getLogger(__name__)
            logger.info(f"Processing UNet/SAUNet batch with shapes - outputs: {outputs.shape}, targets: {targets.shape}")
            if outputs.dim() != 4 or targets.dim() != 3:
                raise ValueError(f"预期输出形状为(B,C,H,W)，目标形状为(B,H,W)，但得到输出形状{outputs.shape}和目标形状{targets.shape}")
            
            batch_size = outputs.shape[0]
            num_classes = outputs.shape[1]
            self.num_samples += batch_size
            
            t_data = time()
            self.debug_times['data_prep'] += t_data - t_start
            
            for i in range(batch_size):
                logger.debug(f"Processing sample {i}/{batch_size}")
                t_mask_start = time()
                
                pred = outputs[i]
                target = targets[i]
                
                pred_probs = F.softmax(pred, dim=0)
                pred_classes = pred_probs.argmax(dim=0)
                
                t_mask_end = time()
                self.debug_times['mask_proc'] += t_mask_end - t_mask_start
                
                t_metric_start = time()
                
                pixel_acc = (pred_classes == target).float().mean().item() * 100
                self.total_pixel_accuracy += pixel_acc
                
                pred_cpu = pred_classes.cpu().numpy()
                target_cpu = target.cpu().numpy()
                self.confusion_matrix.update(pred_cpu.flatten(), target_cpu.flatten())
                
                class_ious = []
                class_dices = []
                for c in range(num_classes):
                    pred_mask = (pred_classes == c).float()
                    target_mask = (target == c).float()
                    
                    if target_mask.sum() > 0:
                        class_iou = calculate_iou(pred_mask, target_mask, num_classes=self.num_classes)
                        class_dice = calculate_dice(pred_mask, target_mask, num_classes=self.num_classes)
                        
                        class_ious.append(min(max(class_iou, 0.0), 1.0))
                        class_dices.append(min(max(class_dice, 0.0), 1.0))
                
                if class_ious:
                    self.total_iou += sum(class_ious) / len(class_ious) * 100
                    self.total_dice += sum(class_dices) / len(class_dices) * 100
                    
                t_metric_end = time()
                self.debug_times['metric_calc'] += t_metric_end - t_metric_start
                
                if i % 10 == 0:  # 每10个样本记录一次时间统计
                    logger.info(f"Time stats for {i}th sample - Data prep: {self.debug_times['data_prep']:.3f}s, "
                              f"Mask processing: {self.debug_times['mask_proc']:.3f}s, "
                              f"Metric calculation: {self.debug_times['metric_calc']:.3f}s")

def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=(), on_plot=None):
    """绘制PR曲线
    
    参数:
        px: P值
        py: R值
        ap: AP值
        save_dir: 保存路径
        names: 类别名称
        on_plot: 回调函数
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # 显示每个类别的图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
    else:
        ax.plot(px, py, linewidth=1, color='grey')

    ax.plot(px, py.mean(1), linewidth=3, color='blue', 
            label=f'all classes {ap[:, 0].mean():.3f} mAP@0.5')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)
