"""DETR模型构建模块"""

from .detr import DETR

def build_detr(backbone_type='resnet50', task_type='semantic', num_classes=21, 
               num_queries=20, hidden_dim=256, nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
               output_size=None):
    """构建DETR模型
    
    参数:
        backbone_type (str): 主干网络类型
        task_type (str): 任务类型
        num_classes (int): 类别数量
        num_queries (int): 目标查询数量
        hidden_dim (int): 隐藏层维度
        nhead (int): 注意力头数
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        dim_feedforward (int): 前馈网络维度
        dropout (float): dropout比率
        output_size (tuple): 输出尺寸，如果为None则保持输入尺寸
    """
    model = DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=num_queries,
        backbone_type=backbone_type,
        task_type=task_type,
        output_size=output_size
    )
    return model

__all__ = ['DETR', 'build_detr']