"""Cityscapes语义分割数据集处理模块"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from MYSEGX.data.transforms import build_transforms, Resize
import logging

class CityscapesSegmentation(Dataset):
    """Cityscapes语义分割数据集
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet'、'cnn'、'deeplabv3'、'deeplabv3plus'
        img_size (tuple): 图像尺寸，默认为(480, 480)
    """
    # Cityscapes训练ID的类别映射
    CLASSES = {
        'road': 0,
        'sidewalk': 1,
        'building': 2,
        'wall': 3,
        'fence': 4,
        'pole': 5,
        'traffic light': 6,
        'traffic sign': 7,
        'vegetation': 8,
        'terrain': 9,
        'sky': 10,
        'person': 11,
        'rider': 12,
        'car': 13,
        'truck': 14,
        'bus': 15,
        'train': 16,
        'motorcycle': 17,
        'bicycle': 18,
        'void': 255  # 忽略的类别
    }
    
    def __init__(self, root, split='train', transform=None, model_type='detr', img_size=(480, 480)):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.model_type = model_type.lower()
        self.img_size = img_size
        
        # 类别数量（不包括void类）
        self.num_classes = 19
        
        # 修改图像和标注路径
        self.images_dir = self.root / 'images' / split
        self.masks_dir = self.root / 'gtFine' / split
        
        # 获取所有城市
        self.cities = [d.name for d in self.images_dir.iterdir() if d.is_dir()]
        
        # 收集所有图像文件
        self.images = []
        self.masks = []
        
        for city in self.cities:
            city_img_dir = self.images_dir / city
            city_mask_dir = self.masks_dir / city
            
            # 获取该城市的所有图像
            for img_file in city_img_dir.glob('*_leftImg8bit.png'):
                # 构建对应的标签文件名
                basename = img_file.stem.replace('_leftImg8bit', '')
                mask_file = city_mask_dir / f"{basename}_gtFine_labelTrainIds.png"
                
                if mask_file.exists():
                    self.images.append(img_file)
                    self.masks.append(mask_file)
        
        logging.info(f"加载了 {len(self.images)} 个Cityscapes {split}集样本")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本
        
        返回:
            dict: 包含以下键的字典:
                - image: 图像张量 (3, H, W)
                - target: 语义分割掩码张量 (H, W)，值范围为[0, num_classes-1]
                - image_id: 图像ID
        """
        logger = logging.getLogger('MYSEGX.dataset')
        
        # 读取图像
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if idx == 0:
            logger.info(f"读取图像: {img_path}")
            logger.info(f"原始图像形状: {img.shape}")
        
        # 读取分割标注
        mask_path = self.masks[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
        
        if idx == 0:
            logger.info(f"读取掩码: {mask_path}")
            logger.info(f"原始掩码形状: {mask.shape}")
            logger.info(f"掩码唯一值: {np.unique(mask)}")
        
        # 首先调整图像和掩码到统一尺寸
        resize_transform = Resize(self.img_size)
        resized = resize_transform(image=img, mask=mask)
        img = resized['image']
        mask = resized['mask']
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # 确保图像是 [C, H, W] 格式的张量
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        elif isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        
        # 将掩码转换为张量
        if isinstance(mask, np.ndarray):
            target = torch.from_numpy(mask).long()
        else:
            target = mask.long()
        
        # 获取图像ID
        img_id = img_path.stem.replace('_leftImg8bit', '')
        
        # 根据模型类型准备返回格式
        if self.model_type == 'detr':
            target_dict = {
                'semantic_mask': target,
                'labels': torch.unique(target[target != 255]),  # 排除void类
                'image_id': img_id
            }
            return {
                'image': img,
                'target': target_dict,
                'image_id': img_id
            }
        else:
            return {
                'image': img,
                'target': target,
                'image_id': img_id
            }
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
    image_ids = []
    
    for b in batch:
        # 确保图像是 [C, H, W] 格式
        img = b['image']
        if img.dim() == 3 and img.shape[0] != 3:  # 如果通道维不在前面
            img = img.permute(2, 0, 1)
        # 确保图像是float类型
        img = img.float()
        images.append(img)
        targets.append(b['target'])
        image_ids.append(b.get('image_id', ''))
    
    # 堆叠图像并确保维度正确
    images = torch.stack(images)  # [B, C, H, W]
    
    # 如果目标是字典（DETR模型），不进行堆叠
    if isinstance(targets[0], dict):
        # 确保所有张量都在同一个设备上
        processed_targets = []
        for target in targets:
            processed_target = {}
            for k, v in target.items():
                if isinstance(v, torch.Tensor):
                    processed_target[k] = v.contiguous()  # 确保内存连续
                else:
                    processed_target[k] = v
            processed_targets.append(processed_target)
        
        return {
            'image': images.contiguous(),  # 确保内存连续
            'target': processed_targets,
            'image_id': image_ids
        }
    
    # 其他模型（UNet、SAUNet、CNN、DeepLabV3、DeepLabV3Plus），堆叠目标掩码
    targets = torch.stack(targets)
    return {
        'image': images.contiguous(),  # 确保内存连续
        'target': targets.contiguous(),  # 确保内存连续
        'image_id': image_ids
    }

def dataloader(root, split='train', batch_size=16, num_workers=4, transform=None, 
               transform_config=None, model_type='detr', task_type='semantic', 
               return_dataset=False, img_size=(640, 640)):
    """创建Cityscapes数据集加载器"""
    logger = logging.getLogger('MYSEGX.dataloader')
    
    # 构建数据增强
    if transform_config is not None:
        transform = build_transforms(transform_config, train=(split=='train'))
    
    # 创建数据集
    dataset = CityscapesSegmentation(
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
        collate_fn=collate_fn,  # 使用与VOC相同的collate_fn
        drop_last=(split == 'train'),
    )
    
    return dataloader