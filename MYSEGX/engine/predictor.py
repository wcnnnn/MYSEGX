"""MYSEGX 预测器模块"""

from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class Predictor:
    """模型预测器类"""
    
    def __init__(self, model):
        """初始化预测器
        
        参数:
            model: 待预测的模型
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # 定义图像预处理转换
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess(self, image_path):
        """图像预处理
        
        参数:
            image_path: 图像路径
        """
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        # 应用预处理转换
        image_tensor = self.transform(image)
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor, image.size
    
    def predict(self, image_path):
        """执行预测
        
        参数:
            image_path: 图像路径
            
        返回:
            outputs: 预测结果字典，包含类别和掩码
        """
        # 预处理
        image_tensor, original_size = self.preprocess(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        # 后处理
        results = self.postprocess(outputs, original_size)
        return results
            
    def postprocess(self, outputs, original_size):
        """预测结果后处理
        
        参数:
            outputs: 模型输出
            original_size: 原始图像尺寸
            
        返回:
            results: 处理后的预测结果字典
        """
        # 获取预测的类别和掩码
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        
        # 对类别进行softmax
        probas = pred_logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7  # 置信度阈值
        
        # 处理掩码
        masks = pred_masks[0, keep]
        probas = probas[keep]
        
        # 调整掩码尺寸到原始图像大小
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(0), size=original_size[::-1], mode='bilinear')[0]
        masks = masks > 0.5  # 二值化掩码
        
        # 转换为numpy数组
        masks = masks.cpu().numpy()
        probas = probas.cpu().numpy()
        
        return {
            'masks': masks,
            'probas': probas,
            'labels': probas.argmax(-1)
        }