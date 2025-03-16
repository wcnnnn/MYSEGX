"""MYSEGX 模型构建模块"""

from pathlib import Path
import torch

from .detr import build_detr
from .unet import build_unet
from .saunet import build_saunet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .yolact import YOLACT

def build_deeplabv3(config):
    """构建DeepLabV3模型
    
    参数:
        config (dict): 模型配置字典
        
    返回:
        model: DeepLabV3模型实例
    """
    model_config = config['model']
    return DeepLabV3(
        num_classes=model_config.get('num_classes', 21),
        backbone=model_config.get('backbone', 'resnet101'),
        output_stride=model_config.get('output_stride', 16),
        pretrained_backbone=model_config.get('pretrained_backbone', True)
    )

def build_deeplabv3plus(config):
    """构建DeepLabV3+模型
    
    参数:
        config (dict): 模型配置字典
        
    返回:
        model: DeepLabV3+模型实例
    """
    model_config = config['model']
    return DeepLabV3Plus(
        num_classes=model_config.get('num_classes', 21),
        backbone=model_config.get('backbone', 'resnet101'),
        output_stride=model_config.get('output_stride', 16),
        pretrained_backbone=model_config.get('pretrained_backbone', True)
    )

def build_yolact(config):
    """构建YOLACT模型
    
    参数:
        config (dict): 模型配置字典
        
    返回:
        model: YOLACT模型实例
    """
    model_config = config['model']
    return YOLACT(
        num_classes=model_config.get('num_classes', 80),
        backbone_type=model_config.get('backbone_type', 'resnet50'),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_protos=model_config.get('num_protos', 32),
        use_gn=model_config.get('use_gn', False),
        top_k=model_config.get('top_k', 200),
        score_threshold=model_config.get('score_threshold', 0.05),
        nms_threshold=model_config.get('nms_threshold', 0.5)
    )

def build_model(model_type, config=None, weights_path=None):
    """构建模型
    
    参数:
        model_type (str): 模型类型，支持 'detr'、'unet'、'cnn'、'saunet'、'mask_rcnn'、'deeplabv3'、'deeplabv3plus'、'yolact'
        config (dict): 模型配置字典，包含模型参数
        weights_path (str): 权重文件路径，如果提供则加载预训练权重
        
    返回:
        model: 构建好的模型实例
        
    示例:
        >>> model = build_model('detr', config={'model': {'backbone_type': 'resnet34', 'task_type': 'semantic', 'num_classes': 21}})
        >>> model = build_model('unet', weights_path='weights/unet.pth')
        >>> model = build_model('cnn', config={'model': {'n_channels': 3, 'n_classes': 2}})
        >>> model = build_model('mask_rcnn', config={'model': {'num_classes': 21, 'backbone_type': 'resnet50'}})
        >>> model = build_model('deeplabv3', config={'model': {'num_classes': 21, 'backbone': 'resnet101'}})
        >>> model = build_model('deeplabv3plus', config={'model': {'num_classes': 21, 'backbone': 'resnet50'}})
        >>> model = build_model('yolact', config={'model': {'num_classes': 80, 'backbone_type': 'resnet50'}})
    """
    # 根据模型类型构建相应的模型
    model_builders = {
        'detr': build_detr,
        'unet': build_unet,
        'saunet': build_saunet,
        'deeplabv3': build_deeplabv3,
        'deeplabv3plus': build_deeplabv3plus,
        'yolact': build_yolact
    }
    
    if model_type not in model_builders:
        raise ValueError(f"不支持的模型类型: {model_type}，可用类型: {list(model_builders.keys())}")
    
    # 构建模型，传递配置参数
    if model_type == 'detr':
        # 获取所有DETR相关参数
        model_config = config['model']
        return build_detr(
            backbone_type=model_config.get('backbone_type', 'resnet50'),
            task_type=config.get('task', {}).get('type', 'semantic'),
            num_classes=model_config.get('num_classes', 21),
            num_queries=model_config.get('num_queries', 20),
            hidden_dim=model_config.get('hidden_dim', 256),
            nhead=model_config.get('nhead', 8),
            num_encoder_layers=model_config.get('num_encoder_layers', 6),
            num_decoder_layers=model_config.get('num_decoder_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 2048),
            dropout=model_config.get('dropout', 0.1),
            output_size=model_config.get('output_size', None)
        )
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

__all__ = ['build_model', 'DeepLabV3', 'DeepLabV3Plus', 'YOLACT']