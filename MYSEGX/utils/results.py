"""结果保存模块"""
import os
import json
import numpy as np
import cv2
import torch
from datetime import datetime
from typing import Dict, Any, Union, List

class ResultSaver:
    def __init__(self, base_dir: str = "results"):
        """初始化结果保存器
        
        参数:
            base_dir: 基础保存目录
        """
        self.base_dir = base_dir
        self._create_experiment_dir()
        
    def _create_experiment_dir(self):
        """创建实验目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.base_dir, f"experiment_{timestamp}")
        self.img_dir = os.path.join(self.exp_dir, "images")
        self.mask_dir = os.path.join(self.exp_dir, "masks")
        self.plot_dir = os.path.join(self.exp_dir, "plots")
        
        # 创建所需的子目录
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def save_config(self, config: Dict[str, Any]):
        """保存配置信息
        
        参数:
            config: 配置字典
        """
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
    def save_metrics(self, metrics: Dict[str, float]):
        """保存评估指标
        
        参数:
            metrics: 评估指标字典
        """
        metrics_path = os.path.join(self.exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
    def _convert_to_serializable(self, obj):
        """将对象转换为可序列化的类型
        
        参数:
            obj: 输入对象
            
        返回:
            可序列化的对象
        """
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
            
    def save_training_history(self, history: dict):
        """保存训练历史
        
        参数:
            history: 训练历史字典，包含损失和指标
        """
        # 转换所有数据为可序列化类型
        serializable_history = self._convert_to_serializable(history)
        
        # 保存到文件
        history_path = os.path.join(self.exp_dir, 'history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=4, ensure_ascii=False)
            
    def save_prediction(self, 
                       image: Union[np.ndarray, torch.Tensor], 
                       mask: Union[np.ndarray, torch.Tensor], 
                       filename: str,
                       alpha: float = 0.5):
        """保存预测结果
        
        参数:
            image: 原始图像，支持以下格式：
                   - NumPy数组 (H, W, C) 或 (C, H, W)
                   - PyTorch张量 (C, H, W)
            mask: 预测的掩码 (H, W)
            filename: 文件名
            alpha: 叠加透明度
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
        else:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # 转换为BGR格式（OpenCV格式）
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 保存原始图像
        img_path = os.path.join(self.img_dir, filename)
        cv2.imwrite(img_path, image)
        
        # 保存掩码
        mask_filename = os.path.splitext(filename)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)  # 将掩码转换为0-255范围
        
        # 生成叠加图像
        overlay = image.copy()
        mask_rgb = np.stack([mask]*3, axis=-1)
        # 修复掩码叠加
        mask_indices = np.where(mask_rgb > 0)
        overlay[mask_indices[0], mask_indices[1]] = [0, 255, 0]  # BGR格式
        
        # 保存叠加结果
        blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        overlay_filename = os.path.splitext(filename)[0] + "_overlay.png"
        overlay_path = os.path.join(self.img_dir, overlay_filename)
        cv2.imwrite(overlay_path, blended)
        
    def save_plot(self, plot_name: str):
        """获取绘图保存路径
        
        参数:
            plot_name: 绘图名称
            
        返回:
            plot_path: 绘图保存路径
        """
        return os.path.join(self.plot_dir, f"{plot_name}.png")