"""可视化绘图模块"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Union
import torch

def generate_colors(num_classes: int) -> Dict[int, tuple]:
    """生成类别颜色映射
    
    使用预定义的颜色方案，确保每个类别都有独特且易区分的颜色
    
    参数:
        num_classes: 类别数量
        
    返回:
        Dict[int, tuple]: 类别颜色映射字典，格式为 {class_id: (R,G,B)}
    """
    # 预定义一些鲜明的基础颜色
    base_colors = [
        (0, 0, 0),       # 背景-黑色
        (128, 0, 0),     # 深红
        (0, 128, 0),     # 深绿
        (128, 128, 0),   # 橄榄
        (0, 0, 128),     # 深蓝
        (128, 0, 128),   # 紫色
        (0, 128, 128),   # 青色
        (128, 128, 128), # 灰色
        (64, 0, 0),      # 暗红
        (192, 0, 0),     # 鲜红
        (64, 128, 0),    # 草绿
        (192, 128, 0),   # 金色
        (64, 0, 128),    # 深紫
        (192, 0, 128),   # 粉红
        (64, 128, 128),  # 浅青
        (192, 128, 128), # 浅粉
        (0, 64, 0),      # 深森林绿
        (128, 64, 0),    # 棕色
        (0, 192, 0),     # 亮绿
        (128, 192, 0),   # 黄绿
        (0, 64, 128),    # 深天蓝
    ]
    
    colors = {}
    
    # 首先使用预定义颜色
    for i in range(min(num_classes, len(base_colors))):
        colors[i] = base_colors[i]
    
    # 如果类别数超过预定义颜色数量，则生成额外的颜色
    if num_classes > len(base_colors):
        for i in range(len(base_colors), num_classes):
            # 使用HSV色彩空间生成额外的颜色
            hue = (i * 0.618033988749895) % 1  # 黄金比例
            saturation = 0.8 + (i % 3) * 0.1   # 在0.8-1.0之间变化
            value = 0.8 + (i % 2) * 0.2        # 在0.8-1.0之间变化
            
            # 将HSV转换为RGB
            h = float(hue * 6)
            s = float(saturation)
            v = float(value)
            
            i_part = int(h)
            f = h - i_part
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            if i_part == 0:
                r, g, b = v, t, p
            elif i_part == 1:
                r, g, b = q, v, p
            elif i_part == 2:
                r, g, b = p, v, t
            elif i_part == 3:
                r, g, b = p, q, v
            elif i_part == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    
    return colors

def plot_segmentation(
    image: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    task_type: str = 'semantic',
    class_colors: Dict[int, tuple] = None,
    class_names: List[str] = None,
    alpha: float = 0.5,
    save_path: str = None
) -> np.ndarray:
    """绘制分割结果"""
    # 转换输入为numpy数组
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    # 确保图像格式正确 (H, W, C)
    if image.shape[0] == 3:  # 如果是(C, H, W)格式
        image = np.transpose(image, (1, 2, 0))
    
    # 归一化图像到[0, 1]范围
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # 创建颜色映射
    num_classes = max(len(class_names) if class_names else 21,  # 使用类别名称列表长度或默认21
                     max(np.max(target), np.max(pred)) + 1)  # 或使用实际出现的最大类别ID + 1
    if class_colors is None:
        class_colors = generate_colors(num_classes)
    
    # 创建彩色掩码
    target_mask = np.zeros((*target.shape, 3))
    pred_mask = np.zeros((*pred.shape, 3))
    
    # 为每个类别上色
    unique_targets = np.unique(target)
    unique_preds = np.unique(pred)
    
    print(f"[DEBUG] Target unique values: {unique_targets}")
    print(f"[DEBUG] Pred unique values: {unique_preds}")
    print(f"[DEBUG] Number of classes: {num_classes}")
    
    # 创建所有类别的图例
    target_patches = []
    pred_patches = []
    
    # 为所有可能的类别创建图例（不仅仅是出现的类别）
    for label in range(num_classes):
        if label == 0:  # 跳过背景
            continue
        color = class_colors.get(int(label), (128, 128, 128))  # 默认灰色
        
        # 为真实标签中的类别上色
        mask = target == label
        target_mask[mask] = [c/255.0 for c in color]  # 归一化颜色值
        
        # 为预测结果中的类别上色
        mask = pred == label
        pred_mask[mask] = [c/255.0 for c in color]  # 归一化颜色值
        
        # 添加到图例
        if class_names and int(label) < len(class_names):
            patch = plt.Rectangle((0, 0), 1, 1, fc=[c/255.0 for c in color])
            class_name = class_names[int(label)]
            target_patches.append((patch, f"{class_name} ({label})"))
            pred_patches.append((patch, f"{class_name} ({label})"))
    
    # 创建主图和图例的布局
    fig = plt.figure(figsize=(20, 10))  # 增加图像大小
    
    # 创建网格布局
    gs = plt.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.5])
    
    # 显示原始图像
    ax1 = plt.subplot(gs[0])
    ax1.imshow(image)
    ax1.set_title('Original Image', pad=10)
    ax1.axis('off')
    
    # 显示真实标签
    ax2 = plt.subplot(gs[1])
    ax2.imshow(target_mask)
    ax2.set_title('Ground Truth', pad=10)
    ax2.axis('off')
    
    # 显示预测结果
    ax3 = plt.subplot(gs[2])
    ax3.imshow(pred_mask)
    ax3.set_title('Prediction', pad=10)
    ax3.axis('off')
    
    # 显示叠加结果
    ax4 = plt.subplot(gs[3])
    image_uint8 = (image * 255).astype(np.uint8)
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    blended = cv2.addWeighted(image_uint8, 1-alpha, pred_mask_uint8, alpha, 0)
    ax4.imshow(blended / 255.0)
    ax4.set_title('Overlay Result', pad=10)
    ax4.axis('off')
    
    # 创建单独的图例区域
    ax5 = plt.subplot(gs[4])
    ax5.axis('off')
    
    # 获取图像中实际出现的类别
    active_classes = sorted(set(unique_targets.tolist() + unique_preds.tolist()))
    active_classes = [c for c in active_classes if c != 0]  # 移除背景类
    
    # 创建图例
    legend_patches = []
    legend_labels = []
    
    # 添加图例标题
    ax5.text(0, 1.1, 'Classes Legend', fontsize=12, fontweight='bold')
    
    # 为每个活跃的类别创建图例项
    for label in active_classes:
        if label < num_classes and class_names and int(label) < len(class_names):
            color = class_colors.get(int(label), (128, 128, 128))
            patch = plt.Rectangle((0, 0), 1, 1, fc=[c/255.0 for c in color])
            class_name = class_names[int(label)]
            legend_patches.append(patch)
            legend_labels.append(f"{class_name} ({label})")
    
    # 添加图例，设置合适的字体大小和列数
    if legend_patches:
        ax5.legend(legend_patches, legend_labels,
                  loc='center left',
                  bbox_to_anchor=(0, 0.5),
                  fontsize=10,
                  ncol=1)
    
    plt.tight_layout()
    
    # 为TensorBoard准备图像
    fig.canvas.draw()
    
    # 获取图像数据
    buf = fig.canvas.buffer_rgba()
    # 转换为numpy数组
    X = np.asarray(buf)
    # 转换RGBA为RGB
    X = X[:, :, :3]
    plt.close()
    
    # 转换为CHW格式用于TensorBoard
    X = np.transpose(X, (2, 0, 1))
    return X.astype(np.uint8)

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