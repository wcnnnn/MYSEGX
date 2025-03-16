"""MYSEGX语义分割数据集包"""

from .voc import VOCSegmentation
from .voc import dataloader as voc_dataloader
from .cityscapes import CityscapesSegmentation
from .cityscapes import dataloader as cityscapes_dataloader
from .ade20k import ADE20KSegmentation
from .ade20k import dataloader as ADE20Kdataloder
__all__ = [
    'VOCSegmentation', 
    'CityscapesSegmentation',
    'voc_dataloader',
    'cityscapes_dataloader',
    'ADE20KSegmentation',
    'ADE20Kdataloder'
]
