"""可视化绘图模块"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Union
import torch

def plot_segmentation(image: Union[np.ndarray, torch.Tensor], 
                     mask: Union[np.ndarray, torch.Tensor], 
                     alpha: float = 0.5, 
                     save_path: str = None):
    """绘制分割结果
    
    参数:
        image: 原始图像，支持以下格式：
               - NumPy数组 (H, W, C) 或 (C, H, W)
               - PyTorch张量 (C, H, W)
        mask: 分割掩码 (H, W)
        alpha: 透明度
        save_path: 保存路径，如果为None则显示图像
    """
    # 转换图像格式
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 如果图像是(C,H,W)格式，转换为(H,W,C)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 确保图像值在0-255范围内
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    plt.figure(figsize=(12, 4))
    
    # 显示原图
    plt.subplot(131)
    plt.imshow(image)  # RGB格式，不需要转换
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示mask
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    # 显示叠加结果
    plt.subplot(133)
    overlay = image.copy()
    mask_rgb = np.stack([mask]*3, axis=-1)
    # 修复掩码叠加
    mask_indices = np.where(mask_rgb > 0)
    overlay[mask_indices[0], mask_indices[1]] = [0, 255, 0]
    
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    plt.imshow(blended)  # RGB格式，不需要转换
    plt.title('Overlay Result')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_curves(losses: Dict[str, List[float]], 
                        metrics: Dict[str, List[float]], 
                        save_path: str = None):
    """绘制训练曲线
    
    参数:
        losses: 损失值字典，格式为 {'loss_name': [values...]}
        metrics: 评估指标字典，格式为 {'metric_name': [values...]}
        save_path: 保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(121)
    for loss_name, loss_values in losses.items():
        # 确保转换为CPU tensor并转为numpy数组
        if isinstance(loss_values, torch.Tensor):
            loss_values = loss_values.cpu().numpy()
        elif isinstance(loss_values, list):
            loss_values = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in loss_values]
        plt.plot(loss_values, label=loss_name)
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    # 绘制指标曲线
    plt.subplot(122)
    for metric_name, metric_values in metrics.items():
        # 确保转换为CPU tensor并转为numpy数组
        if isinstance(metric_values, torch.Tensor):
            metric_values = metric_values.cpu().numpy()
        elif isinstance(metric_values, list):
            metric_values = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in metric_values]
        plt.plot(metric_values, label=metric_name)
    plt.title('Training Metrics')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_comparison(images: List[np.ndarray], 
                   titles: List[str], 
                   save_path: str = None,
                   figsize: tuple = (15, 5)):
    """绘制图像对比结果
    
    参数:
        images: 图像列表
        titles: 标题列表
        save_path: 保存路径
        figsize: 图像大小
    """
    assert len(images) == len(titles), "图像数量和标题数量必须相同"
    
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        # 确保转换为CPU tensor并转为numpy数组
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()