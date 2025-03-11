"""MYSEGX
~~~~~

现代化图像分割框架，支持DETR、UNet和CNN等多种模型。

使用示例:
    >>> from MYSEGX import build_model, train
    >>> 
    >>> # 训练模型
    >>> history = train('configs/models/detr/detr_r18.yaml')
    >>> 
    >>> # 或者手动构建模型
    >>> model = build_model('detr', config_path='configs/models/detr/detr_r18.yaml')
"""

__version__ = '0.1.0'

from MYSEGX.models import build_model
from MYSEGX.api import train

__all__ = [
    'build_model',
     'train',
]
