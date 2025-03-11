"""VOC实例分割数据集处理模块"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask

class VOCInstanceSegmentation(Dataset):
    """VOC实例分割数据集
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet' 和 'cnn'
        task_type (str): 任务类型，支持 'semantic' 或 'instance'
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
    
    def __init__(self, root, split='train', transform=None, model_type='detr', task_type='instance'):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.model_type = model_type.lower()  # 转换为小写以确保一致性
        self.task_type = task_type
        
        # 类别数量（包括背景类）
        self.num_classes = 21
        
        # 图像和标注路径
        self.images_dir = os.path.join(root, 'JPEGImages')
        self.masks_dir = os.path.join(root, 'SegmentationObject')
        
        # 读取数据集划分文件
        split_file = os.path.join(root, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.ids)
    
    def convert_to_coco_format(self, mask):
        """将二值掩码转换为COCO RLE格式"""
        mask = np.asfortranarray(mask)
        rle = coco_mask.encode(mask)
        return rle
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本
        
        返回:
            dict: 包含以下键的字典:
                - image: 图像张量 (3, H, W)
                - target: 目标字典，根据模型类型有不同格式
                - image_id: 图像ID
        """
        img_id = self.ids[idx]
        
        # 读取图像
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取实例掩码
        mask_path = os.path.join(self.masks_dir, f'{img_id}.png')
        instance_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 数据预处理
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
            
            # 调整掩码大小
            h, w = instance_mask.shape
            instance_mask = cv2.resize(instance_mask, (img.shape[-1], img.shape[-2]),
                                     interpolation=cv2.INTER_NEAREST)
        
        # 处理实例掩码和标签
        unique_ids = np.unique(instance_mask)
        unique_ids = unique_ids[unique_ids != 0]  # 移除背景
        
        masks = []
        labels = []
        boxes = []
        
        for instance_id in unique_ids:
            # 提取单个实例掩码
            binary_mask = (instance_mask == instance_id).astype(np.uint8)
            
            # 计算边界框
            y_indices, x_indices = np.where(binary_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x1, x2 = np.min(x_indices), np.max(x_indices)
                y1, y2 = np.min(y_indices), np.max(y_indices)
                
                # 添加掩码、标签和边界框
                masks.append(torch.from_numpy(binary_mask))
                label = instance_id % self.num_classes  # 使用掩码值对类别数取模获取类别ID
                labels.append(label)
                boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        
        # 如果没有实例，添加空张量
        if len(masks) == 0:
            masks = torch.zeros((0, img.shape[-2], img.shape[-1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            masks = torch.stack(masks)
            labels = torch.tensor(labels, dtype=torch.int64)
            boxes = torch.stack(boxes)
        
        # 根据模型类型准备目标
        if self.model_type == 'detr':
            target_dict = {
                'masks': masks,  # 实例掩码
                'labels': labels,  # 类别标签
                'boxes': boxes,  # 边界框
                'image_id': img_id
            }
            return {
                'image': img,
                'target': target_dict
            }
        elif self.model_type in ['unet', 'saunet', 'cnn']:  # 添加对SAUNet的支持
            # 对于其他模型，返回合并的实例掩码
            combined_mask = torch.zeros((img.shape[-2], img.shape[-1]), dtype=torch.long)
            for i, (mask, label) in enumerate(zip(masks, labels)):
                # 确保标签在正确的范围内 [0, num_classes-1]
                label = torch.clamp(label, 0, self.num_classes - 1)
                combined_mask[mask > 0] = label + 1  # +1 因为0是背景
            
            return {
                'image': img,
                'target': combined_mask,
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
            - target: 目标字典列表或堆叠的实例掩码张量
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

def dataloader(root, split='train', batch_size=16, num_workers=4, transform=None, model_type='detr', task_type='instance'):
    """创建VOC数据集加载器
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet' 和 'cnn'
        task_type (str): 任务类型，支持 'semantic' 或 'instance'
    """
    # 确保使用正确的任务类型
    assert task_type == 'instance', "此数据加载器仅支持实例分割任务"
    
    # 确保模型类型受支持
    model_type = model_type.lower()  # 转换为小写以确保一致性
    assert model_type in ['detr', 'unet', 'saunet', 'cnn'], f"不支持的模型类型: {model_type}"
    
    dataset = VOCInstanceSegmentation(
        root=root,
        split=split,
        transform=transform,
        model_type=model_type,
        task_type=task_type
    )
    
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