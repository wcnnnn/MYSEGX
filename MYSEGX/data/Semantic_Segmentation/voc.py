"""VOC语义分割数据集处理模块"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class VOCSegmentation(Dataset):
    """VOC语义分割数据集
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet' 和 'cnn'
    """
    # VOC数据集类别映射
    VOC_CLASSES = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
    
    def __init__(self, root, split='train', transform=None, model_type='detr'):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.model_type = model_type.lower()  # 转换为小写以确保一致性
        
        # 类别数量（包括背景类）
        self.num_classes = 21
        
        # VOC数据集的调色板
        self.palette = [
            0, 0, 0,        # 背景 (0)
            128, 0, 0,      # aeroplane (1)
            0, 128, 0,      # bicycle (2)
            128, 128, 0,    # bird (3)
            0, 0, 128,      # boat (4)
            128, 0, 128,    # bottle (5)
            0, 128, 128,    # bus (6)
            128, 128, 128,  # car (7)
            64, 0, 0,       # cat (8)
            192, 0, 0,      # chair (9)
            64, 128, 0,     # cow (10)
            192, 128, 0,    # diningtable (11)
            64, 0, 128,     # dog (12)
            192, 0, 128,    # horse (13)
            64, 128, 128,   # motorbike (14)
            192, 128, 128,  # person (15)
            0, 64, 0,       # pottedplant (16)
            128, 64, 0,     # sheep (17)
            0, 192, 0,      # sofa (18)
            128, 192, 0,    # train (19)
            0, 64, 128      # tvmonitor (20)
        ]
        
        # 创建颜色到类别的映射
        self.color_to_class = {}
        for i in range(self.num_classes):
            color = tuple(self.palette[i*3:(i+1)*3])
            self.color_to_class[color] = i
        
        # 图像和标注路径
        self.images_dir = self.root / 'JPEGImages'
        self.masks_dir = self.root / 'SegmentationClass'
        
        # 读取数据集划分文件
        split_file = self.root / 'ImageSets' / 'Segmentation' / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本
        
        返回:
            dict: 包含以下键的字典:
                - image: 图像张量 (3, H, W)
                - target: 语义分割掩码张量 (H, W)，值范围为[0, num_classes-1]
                - image_id: 图像ID
        """
        img_id = self.ids[idx]
        
        # 读取图像
        img_path = self.images_dir / f'{img_id}.jpg'
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取分割标注（使用BGR格式读取）
        mask_path = self.masks_dir / f'{img_id}.png'
        mask_bgr = cv2.imread(str(mask_path))
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        
        # 创建语义分割掩码
        semantic_mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        
        # 将RGB颜色映射到类别ID
        for y in range(mask_rgb.shape[0]):
            for x in range(mask_rgb.shape[1]):
                pixel_color = tuple(mask_rgb[y, x])
                semantic_mask[y, x] = self.color_to_class.get(pixel_color, 0)  # 默认为背景类
        
        # 数据预处理
        if self.transform:
            # 转换图像
            transformed = self.transform(image=img)
            img = transformed['image']
            
            # 转换掩码
            h, w = semantic_mask.shape
            semantic_mask = cv2.resize(semantic_mask, (img.shape[-1], img.shape[-2]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # 将掩码转换为张量
        target = torch.as_tensor(semantic_mask, dtype=torch.long)
        
        # 根据模型类型准备目标
        if self.model_type == 'detr':
            target_dict = {
                'semantic_mask': target,  # 语义分割掩码
                'labels': torch.unique(target),  # 存在的类别标签
                'image_id': img_id
            }
            return {
                'image': img,
                'target': target_dict
            }
        elif self.model_type in ['unet', 'saunet', 'cnn']:
            return {
                'image': img,
                'target': target,
                'image_id': img_id
            }
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

def collate_fn(batch):
    """自定义批次收集函数
    
    参数:
        batch: 包含数据集样本的列表
        
    返回:
        dict: 包含以下键的字典:
            - image: 堆叠的图像张量 (B, C, H, W)
            - target: 堆叠的语义分割掩码张量 (B, H, W) 或目标字典列表
    """
    images = []
    targets = []
    
    for b in batch:
        images.append(b['image'])
        targets.append(b['target'])
    
    # 堆叠图像
    images = torch.stack(images)
    
    # 如果目标是字典（DETR模型），不进行堆叠
    if isinstance(targets[0], dict):
        return {
            'image': images,
            'target': targets
        }
    
    # 其他模型（UNet、SAUNet、CNN），堆叠目标掩码
    targets = torch.stack(targets)
    return {
        'image': images,
        'target': targets
    }

def dataloader(root, split='train', batch_size=16, num_workers=4, transform=None, model_type='detr', task_type='semantic', return_dataset=False):
    """创建VOC数据集加载器
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet' 和 'cnn'
        task_type (str): 任务类型，支持 'semantic' 或 'instance'
        return_dataset (bool): 是否返回数据集对象而不是数据加载器
    """
    # 确保使用正确的任务类型
    assert task_type == 'semantic', "此数据加载器仅支持语义分割任务"
    
    # 确保模型类型受支持
    model_type = model_type.lower()  # 转换为小写以确保一致性
    assert model_type in ['detr', 'unet', 'saunet', 'cnn'], f"不支持的模型类型: {model_type}"
    
    dataset = VOCSegmentation(
        root=root,
        split=split,
        transform=transform,
        model_type=model_type
    )
    
    if return_dataset:
        return dataset
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train'),
    )
    
    return dataloader
