"""DETR模型构建模块"""

from .detr import DETR

def build_detr(backbone_type='resnet50', task_type='semantic', num_classes=21):
    """构建DETR模型
    
    参数:
        backbone_type (str): 主干网络类型，可选：resnet18, resnet34, resnet50, resnet101, vgg16, vgg19, mobilenetv2, mobilenetv3small, mobilenetv3large
        task_type (str): 任务类型，可选：semantic, instance, panoptic
        num_classes (int): 类别数量，包括背景类
        
    返回:
        model: DETR模型实例
    """
    model = DETR(
        num_classes=num_classes,
        hidden_dim=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_queries=100,
        backbone_type=backbone_type,
        task_type=task_type
    )
    return model

__all__ = ['DETR', 'build_detr']