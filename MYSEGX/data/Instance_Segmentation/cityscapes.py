"""Cityscapes实例分割数据集处理模块"""

import os
import cv2
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from MYSEGX.data.transforms import build_transforms, Resize
import logging

class CityscapesInstanceSegmentation(Dataset):
    """Cityscapes实例分割数据集
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'mask_rcnn'、'yolact'
        img_size (tuple): 图像尺寸，默认为(480, 480)
    """
    # Cityscapes类别映射（只包含有实例的类别）
    INSTANCE_CLASSES = {
        'person': 11,
        'rider': 12,
        'car': 13,
        'truck': 14,
        'bus': 15,
        'train': 16,
        'motorcycle': 17,
        'bicycle': 18
    }
    
    def __init__(self, root, split='train', transform=None, model_type='detr', img_size=(480, 480)):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.model_type = model_type.lower()
        self.img_size = img_size
        
        # 验证模型类型
        valid_models = ['detr', 'mask_rcnn', 'yolact']
        if self.model_type not in valid_models:
            raise ValueError(f"不支持的模型类型: {model_type}，可用选项: {valid_models}")
        
        # 实例分割类别数量（8个可以有实例的类别）
        self.num_classes = len(self.INSTANCE_CLASSES)
        
        # 图像和标注路径
        self.images_dir = self.root / 'images' / split
        self.gtFine_dir = self.root / 'gtFine' / split
        
        # 获取所有城市
        self.cities = [d.name for d in self.images_dir.iterdir() if d.is_dir()]
        
        # 收集所有图像文件
        self.images = []
        self.instance_masks = []  # instanceIds.png
        self.semantic_masks = []  # labelTrainIds.png
        self.json_files = []     # polygons.json
        
        for city in self.cities:
            city_img_dir = self.images_dir / city
            city_gt_dir = self.gtFine_dir / city
            
            # 获取该城市的所有图像
            for img_file in city_img_dir.glob('*_leftImg8bit.png'):
                # 构建对应的标签文件名
                basename = img_file.stem.replace('_leftImg8bit', '')
                instance_file = city_gt_dir / f"{basename}_gtFine_instanceIds.png"
                semantic_file = city_gt_dir / f"{basename}_gtFine_labelTrainIds.png"
                json_file = city_gt_dir / f"{basename}_gtFine_polygons.json"
                
                if instance_file.exists() and semantic_file.exists() and json_file.exists():
                    self.images.append(img_file)
                    self.instance_masks.append(instance_file)
                    self.semantic_masks.append(semantic_file)
                    self.json_files.append(json_file)
        
        logging.info(f"加载了 {len(self.images)} 个Cityscapes {split}集实例分割样本")
        
    def __len__(self):
        return len(self.images)
    
    def _load_json_info(self, json_path):
        """加载JSON文件中的实例信息"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _process_instance_mask(self, instance_mask, semantic_mask):
        """处理实例掩码，提取每个实例的掩码和类别"""
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids != 0]  # 移除背景
        
        masks = []
        labels = []
        boxes = []
        
        for instance_id in instance_ids:
            # 提取单个实例掩码
            binary_mask = (instance_mask == instance_id).astype(np.uint8)
            
            # 获取该实例的语义类别
            # instanceIds的前16位是类别ID
            category_id = instance_id >> 16
            semantic_label = semantic_mask[binary_mask > 0][0]  # 获取实例区域内的语义标签
            
            # 只处理可以有实例的类别
            if semantic_label in self.INSTANCE_CLASSES.values():
                # 计算边界框
                y_indices, x_indices = np.where(binary_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, x2 = np.min(x_indices), np.max(x_indices)
                    y1, y2 = np.min(y_indices), np.max(y_indices)
                    
                    # 添加掩码、标签和边界框
                    masks.append(torch.from_numpy(binary_mask))
                    labels.append(semantic_label)
                    boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        
        return masks, labels, boxes
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        logger = logging.getLogger('MYSEGX.dataset')
        
        # 读取图像
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取实例掩码和语义掩码
        instance_mask = cv2.imread(str(self.instance_masks[idx]), cv2.IMREAD_UNCHANGED)
        semantic_mask = cv2.imread(str(self.semantic_masks[idx]), cv2.IMREAD_GRAYSCALE)
        
        # 首先调整图像和掩码到统一尺寸
        resize_transform = Resize(self.img_size)
        resized = resize_transform(image=img, mask=instance_mask)
        img = resized['image']
        instance_mask = resized['mask']
        
        # 调整语义掩码大小
        semantic_mask = cv2.resize(semantic_mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=img, mask=instance_mask)
            img = transformed['image']
            instance_mask = transformed['mask']
            semantic_mask = cv2.resize(semantic_mask, (img.shape[1], img.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # 确保图像是 [C, H, W] 格式的张量
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        elif isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        
        # 处理实例掩码和标签
        masks, labels, boxes = self._process_instance_mask(instance_mask, semantic_mask)
        
        # 如果没有实例，添加空张量
        if len(masks) == 0:
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            masks = torch.stack(masks)
            labels = torch.tensor(labels, dtype=torch.int64)
            boxes = torch.stack(boxes)
        
        # 获取图像ID
        img_id = img_path.stem.replace('_leftImg8bit', '')
        
        # 根据模型类型准备返回格式
        if self.model_type == 'yolact':
            target_dict = {
                'masks': masks,
                'labels': labels,
                'boxes': boxes,
                'image_id': img_id,
                'size': torch.tensor([img.shape[1], img.shape[2]]),
                'orig_size': torch.tensor(self.img_size)
            }
        else:  # detr或mask_rcnn
            target_dict = {
                'masks': masks,
                'labels': labels,
                'boxes': boxes,
                'image_id': img_id,
                'size': torch.tensor([img.shape[1], img.shape[2]])
            }
        
        return {
            'image': img,
            'target': target_dict,
            'image_id': img_id
        }

def collate_fn(batch):
    """自定义批次收集函数"""
    images = []
    targets = []
    image_ids = []
    
    for b in batch:
        # 确保图像是 [C, H, W] 格式
        img = b['image']
        if img.dim() == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        # 确保图像是float类型
        img = img.float()
        images.append(img)
        targets.append(b['target'])
        image_ids.append(b.get('image_id', ''))
    
    # 堆叠图像
    images = torch.stack(images)  # [B, C, H, W]
    
    # 确保所有张量都在同一个设备上
    processed_targets = []
    for target in targets:
        processed_target = {}
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                processed_target[k] = v.contiguous()
            else:
                processed_target[k] = v
        processed_targets.append(processed_target)
    
    return {
        'image': images.contiguous(),
        'target': processed_targets,
        'image_id': image_ids
    }

def dataloader(root, split='train', batch_size=16, num_workers=4, transform=None, 
               transform_config=None, model_type='detr', task_type='instance', 
               return_dataset=False, img_size=(640, 640)):
    """创建Cityscapes实例分割数据集加载器"""
    logger = logging.getLogger('MYSEGX.dataloader')
    
    # 记录初始参数
    logger.info(f"创建Cityscapes实例分割数据加载器:")
    logger.info(f"- 数据集根目录: {root}")
    logger.info(f"- 数据集划分: {split}")
    logger.info(f"- 批次大小: {batch_size}")
    logger.info(f"- 工作进程数: {num_workers}")
    logger.info(f"- 模型类型: {model_type}")
    logger.info(f"- 图像尺寸: {img_size}")
    
    # 确保使用正确的任务类型
    assert task_type == 'instance', "此数据加载器仅支持实例分割任务"
    
    # 确保模型类型受支持
    model_type = model_type.lower()
    assert model_type in ['detr', 'mask_rcnn'], f"不支持的模型类型: {model_type}"
    
    # 构建数据增强
    if transform_config is not None:
        transform = build_transforms(transform_config, train=(split=='train'))
    
    # 创建数据集
    dataset = CityscapesInstanceSegmentation(
        root=root,
        split=split,
        transform=transform,
        model_type=model_type,
        img_size=img_size
    )
    
    if return_dataset:
        return dataset
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )
    
    return dataloader
