"""结果保存模块"""
import os
import json
import numpy as np
import cv2
import torch
import csv
import pandas as pd
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
        self.images_dir = os.path.join(self.exp_dir, "images")
        self.masks_dir = os.path.join(self.exp_dir, "masks")
        self.csv_dir = os.path.join(self.exp_dir, "csv")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        self.seg_results_dir = os.path.join(self.exp_dir, "segmentation_results")
        
        # 创建所需的子目录
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.seg_results_dir, exist_ok=True)
        
    def save_config(self, config: Dict[str, Any]):
        """保存配置信息
        
        参数:
            config: 配置字典
        """
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
    def save_metrics_to_csv(self, history: Dict[str, List], epoch: int = None):
        """将训练和验证的损失和指标保存为CSV文件
        
        参数:
            history: 训练历史字典，包含损失和指标
            epoch: 当前训练轮次，如果为None则保存所有历史
        """
        # 确保CSV目录存在
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # 如果只保存当前epoch的数据
        if epoch is not None:
            # 创建当前epoch的数据字典
            current_data = {
                'epoch': epoch,
                'train_loss': history['train_loss'][epoch],
                'val_loss': history['val_loss'][epoch]
            }
            
            # 添加训练指标
            if epoch < len(history['train_metrics']):
                for metric_name, value in history['train_metrics'][epoch].items():
                    current_data[f'train_{metric_name}'] = value
            
            # 添加验证指标
            if epoch < len(history['val_metrics']):
                for metric_name, value in history['val_metrics'][epoch].items():
                    current_data[f'val_{metric_name}'] = value
            
            # 将当前数据添加到CSV文件
            csv_path = os.path.join(self.csv_dir, 'metrics.csv')
            
            # 检查文件是否存在，如果不存在则创建并写入表头
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=current_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(current_data)
        else:
            # 保存所有历史数据
            # 创建一个包含所有epoch数据的列表
            all_data = []
            
            for e in range(len(history['train_loss'])):
                epoch_data = {
                    'epoch': e,
                    'train_loss': history['train_loss'][e],
                }
                
                # 添加验证损失（如果存在）
                if e < len(history['val_loss']):
                    epoch_data['val_loss'] = history['val_loss'][e]
                
                # 添加训练指标
                if e < len(history['train_metrics']):
                    for metric_name, value in history['train_metrics'][e].items():
                        epoch_data[f'train_{metric_name}'] = value
                
                # 添加验证指标
                if e < len(history['val_metrics']):
                    for metric_name, value in history['val_metrics'][e].items():
                        epoch_data[f'val_{metric_name}'] = value
                
                all_data.append(epoch_data)
            
            # 使用pandas保存为CSV
            df = pd.DataFrame(all_data)
            csv_path = os.path.join(self.csv_dir, 'metrics.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path
            
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
        try:
            # 转换图像格式
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            
            # 如果图像是(C,H,W)格式，转换为(H,W,C)
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # 确保图像值在0-255范围内
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            # 转换为BGR格式（OpenCV格式）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 保存原始图像
            img_path = os.path.join(self.images_dir, filename)
            cv2.imwrite(img_path, image)
            
            # 保存掩码
            mask_filename = os.path.splitext(filename)[0] + "_mask.png"
            mask_path = os.path.join(self.masks_dir, mask_filename)
            
            # 确保掩码是2D的
            if len(mask.shape) > 2:
                mask = mask.squeeze()  # 移除多余的维度
            
            # 将掩码转换为0-255范围
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
                
            cv2.imwrite(mask_path, mask)
            
            # 生成叠加图像
            overlay = image.copy()
            
            # 确保掩码是3通道的，用于叠加
            if len(mask.shape) == 2:
                mask_rgb = np.stack([mask]*3, axis=-1) if len(image.shape) == 3 else mask
            else:
                mask_rgb = mask
                
            # 修复掩码叠加
            if len(image.shape) == 3 and len(mask_rgb.shape) == 3:
                mask_indices = np.where(mask_rgb[:,:,0] > 0)
                if len(mask_indices[0]) > 0:
                    overlay[mask_indices[0], mask_indices[1]] = [0, 255, 0]  # BGR格式
            
            # 保存叠加结果
            blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
            overlay_filename = os.path.splitext(filename)[0] + "_overlay.png"
            overlay_path = os.path.join(self.images_dir, overlay_filename)
            cv2.imwrite(overlay_path, blended)
            
            return img_path, mask_path, overlay_path
        except Exception as e:
            print(f"保存预测结果失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 尝试使用PIL保存
            try:
                from PIL import Image
                
                # 保存原始图像
                img_path = os.path.join(self.images_dir, filename)
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                pil_img = Image.fromarray(image)
                pil_img.save(img_path)
                
                # 保存掩码
                mask_filename = os.path.splitext(filename)[0] + "_mask.png"
                mask_path = os.path.join(self.masks_dir, mask_filename)
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                pil_mask = Image.fromarray(mask)
                pil_mask.save(mask_path)
                
                # 简单返回路径，不生成叠加图像
                overlay_path = None
                print(f"使用PIL成功保存图像和掩码")
                return img_path, mask_path, overlay_path
            except Exception as e:
                print(f"使用PIL保存也失败: {str(e)}")
                return None, None, None
        
    def save_segmentation_result(self, 
                                img_grid: Union[np.ndarray, torch.Tensor], 
                                epoch: int, 
                                sample_idx: int):
        """保存分割结果图像
        
        参数:
            img_grid: 分割结果图像网格
            epoch: 当前训练轮次
            sample_idx: 样本索引
        """
        # 确保目录存在
        epoch_dir = os.path.join(self.seg_results_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 转换图像格式
        if isinstance(img_grid, torch.Tensor):
            # 如果是PyTorch张量，转换为NumPy数组
            if img_grid.dim() == 4 and img_grid.shape[0] == 1:  # [1, C, H, W]
                img_grid = img_grid.squeeze(0)
            if img_grid.dim() == 3:  # [C, H, W]
                img_grid = img_grid.permute(1, 2, 0).cpu().numpy()
            else:
                img_grid = img_grid.cpu().numpy()
        
        # 确保图像值在0-255范围内
        if img_grid.max() <= 1.0:
            img_grid = (img_grid * 255).astype(np.uint8)
        
        # 检查图像形状，如果是(C, H, W)格式，转换为(H, W, C)
        if len(img_grid.shape) == 3 and img_grid.shape[0] == 3:
            img_grid = np.transpose(img_grid, (1, 2, 0))
        
        # 如果是RGB格式，转换为BGR（OpenCV格式）
        if len(img_grid.shape) == 3 and img_grid.shape[2] == 3:
            img_grid = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
        
        # 检查图像尺寸是否合理
        if img_grid.shape[0] > 10000 or img_grid.shape[1] > 10000:
            print(f"警告：图像尺寸过大 {img_grid.shape}，将被调整")
            # 调整图像大小到合理范围
            scale_factor = min(1.0, 1000 / max(img_grid.shape[0], img_grid.shape[1]))
            new_size = (int(img_grid.shape[1] * scale_factor), int(img_grid.shape[0] * scale_factor))
            img_grid = cv2.resize(img_grid, new_size)
        
        # 保存图像
        save_path = os.path.join(epoch_dir, f"sample_{sample_idx}.png")
        
        try:
            # 尝试使用PIL保存图像
            from PIL import Image
            if len(img_grid.shape) == 3 and img_grid.shape[2] == 3:
                # RGB图像
                pil_img = Image.fromarray(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
            else:
                # 灰度图像
                pil_img = Image.fromarray(img_grid.squeeze())
            pil_img.save(save_path)
            print(f"使用PIL成功保存图像: {save_path}")
        except Exception as e:
            print(f"使用PIL保存图像失败: {str(e)}")
            
            try:
                # 尝试使用matplotlib保存图像
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                if len(img_grid.shape) == 3 and img_grid.shape[2] == 3:
                    plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(img_grid.squeeze(), cmap='gray')
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"使用matplotlib成功保存图像: {save_path}")
            except Exception as e:
                print(f"使用matplotlib保存图像也失败: {str(e)}")
                
                try:
                    # 最后尝试使用OpenCV保存
                    # 确保图像是正确的格式和尺寸
                    if len(img_grid.shape) == 3 and img_grid.shape[2] == 3:
                        # 确保图像是uint8类型
                        img_to_save = img_grid.astype(np.uint8)
                    else:
                        # 对于灰度图像，确保它是2D的
                        img_to_save = img_grid.squeeze().astype(np.uint8)
                    
                    # 限制图像尺寸
                    max_dim = 2000
                    h, w = img_to_save.shape[:2]
                    if h > max_dim or w > max_dim:
                        scale = min(max_dim / h, max_dim / w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        img_to_save = cv2.resize(img_to_save, (new_w, new_h))
                    
                    success = cv2.imwrite(save_path, img_to_save)
                    if success:
                        print(f"使用OpenCV成功保存图像: {save_path}")
                    else:
                        print(f"使用OpenCV保存图像失败")
                except Exception as e:
                    print(f"所有图像保存方法都失败: {str(e)}")
        
        return save_path
        
    def get_tensorboard_dir(self):
        """获取TensorBoard日志目录
        
        返回:
            tensorboard_dir: TensorBoard日志目录
        """
        return self.tensorboard_dir
        
    def save_checkpoint(self, checkpoint: Dict[str, Any], filename: str):
        """保存模型检查点
        
        参数:
            checkpoint: 包含模型状态等信息的字典
            filename: 保存的文件名
        """
        checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)