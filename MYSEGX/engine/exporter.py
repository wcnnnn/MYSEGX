"""MYSEGX 导出器模块"""

import torch
import torch.onnx
from pathlib import Path

class Exporter:
    """模型导出器类"""
    
    def __init__(self, model):
        """初始化导出器
        
        参数:
            model: 待导出的模型
        """
        self.model = model
        self.device = next(model.parameters()).device
    
    def to_onnx(self, save_path, input_shape=(1, 3, 640, 640)):
        """导出为ONNX格式
        
        参数:
            save_path: 保存路径
            input_shape: 输入张量形状，默认为(1, 3, 640, 640)
        """
        save_path = Path(save_path)
        
        # 确保保存路径存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备示例输入
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 导出模型
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f'模型已成功导出到: {save_path}')
    
    def to_torchscript(self, save_path, input_shape=(1, 3, 640, 640)):
        """导出为TorchScript格式
        
        参数:
            save_path: 保存路径
            input_shape: 输入张量形状，默认为(1, 3, 640, 640)
        """
        save_path = Path(save_path)
        
        # 确保保存路径存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备示例输入
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 转换为TorchScript
        traced_script_module = torch.jit.trace(self.model, dummy_input)
        
        # 保存模型
        traced_script_module.save(str(save_path))
        
        print(f'模型已成功导出到: {save_path}')