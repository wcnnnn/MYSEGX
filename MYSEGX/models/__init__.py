"""MYSEGX 模型构建模块"""

from pathlib import Path
import torch

from .detr import build_detr
from .unet import build_unet
from .cnn import build_cnn

def build_model(model_type, config=None, weights_path=None):
    """构建模型
    
    参数:
        model_type (str): 模型类型，支持 'detr'、'unet' 和 'cnn'
        config (dict): 模型配置字典，包含模型参数
        weights_path (str): 权重文件路径，如果提供则加载预训练权重
        
    返回:
        model: 构建好的模型实例
        
    示例:
        >>> model = build_model('detr', config={'backbone_type': 'resnet34'})
        >>> model = build_model('unet', weights_path='weights/unet.pth')
        >>> model = build_model('cnn', config={'model': {'n_channels': 3, 'n_classes': 2}})
    """
    # 根据模型类型构建相应的模型
    model_builders = {
        'detr': build_detr,
        'unet': build_unet,
        'cnn': build_cnn
    }
    
    if model_type not in model_builders:
        raise ValueError(f"不支持的模型类型: {model_type}，可用类型: {list(model_builders.keys())}")
    
    # 构建模型，传递配置参数
    if config and model_type == 'detr' and 'backbone_type' in config['model']:
        model = model_builders[model_type](backbone_type=config['model']['backbone_type'])
    else:
        model = model_builders[model_type](config)
    
    # 如果提供了权重路径，加载预训练权重
    if weights_path is not None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
            
        # 加载权重
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model

__all__ = ['build_model']