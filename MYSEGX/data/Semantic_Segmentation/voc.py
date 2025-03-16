"""VOC语义分割数据集处理模块"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from MYSEGX.data.transforms import build_transforms, Resize
import logging

class VOCSegmentation(Dataset):
    """VOC语义分割数据集
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        transform (callable, optional): 数据增强和预处理
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet'、'cnn'、'deeplabv3'、'deeplabv3plus'
        img_size (tuple): 图像尺寸，默认为(480, 480)
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
    
    def __init__(self, root, split='train', transform=None, model_type='detr', img_size=(480, 480)):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.model_type = model_type.lower()  # 转换为小写以确保一致性
        self.img_size = img_size  # 添加图像尺寸参数
        
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
        logger = logging.getLogger('MYSEGX.dataset')
        
        img_id = self.ids[idx]
        
        # 读取图像
        img_path = self.images_dir / f'{img_id}.jpg'
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if idx == 0:  # 只对第一个样本记录详细日志
            logger.info(f"读取图像: {img_path}")
            logger.info(f"原始图像形状: {img.shape}")
            logger.info(f"原始图像数据类型: {img.dtype}")
            logger.info(f"原始图像值范围: [{np.min(img)}, {np.max(img)}]")
        
        # 读取分割标注（使用BGR格式读取）
        mask_path = self.masks_dir / f'{img_id}.png'
        mask_bgr = cv2.imread(str(mask_path))
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        
        if idx == 0:
            logger.info(f"读取掩码: {mask_path}")
            logger.info(f"原始掩码形状: {mask_rgb.shape}")
        
        # 创建语义分割掩码
        semantic_mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        
        # 将RGB颜色映射到类别ID
        for y in range(mask_rgb.shape[0]):
            for x in range(mask_rgb.shape[1]):
                pixel_color = tuple(mask_rgb[y, x])
                semantic_mask[y, x] = self.color_to_class.get(pixel_color, 0)  # 默认为背景类
        
        if idx == 0:
            logger.info(f"语义掩码形状: {semantic_mask.shape}")
            logger.info(f"语义掩码数据类型: {semantic_mask.dtype}")
            logger.info(f"语义掩码值范围: [{np.min(semantic_mask)}, {np.max(semantic_mask)}]")
            logger.info(f"语义掩码唯一值: {np.unique(semantic_mask)}")
        
        # 首先调整图像和掩码到统一尺寸，确保批处理时尺寸一致
        resize_transform = Resize(self.img_size)
        resized = resize_transform(image=img, mask=semantic_mask)
        img = resized['image']
        semantic_mask = resized['mask']
        
        if idx == 0:
            logger.info(f"调整大小后图像形状: {img.shape}")
            logger.info(f"调整大小后图像值范围: [{np.min(img)}, {np.max(img)}]")
            logger.info(f"调整大小后掩码形状: {semantic_mask.shape}")
        
        # 应用其他数据增强
        if self.transform:
            if idx == 0:
                logger.info("应用数据增强转换...")
                if hasattr(self.transform, 'transforms'):
                    logger.info("转换流水线包含:")
                    for i, t in enumerate(self.transform.transforms):
                        logger.info(f"  {i+1}. {t.__class__.__name__}")
            
            transformed = self.transform(image=img, mask=semantic_mask)
            img = transformed['image']
            semantic_mask = transformed['mask']
            
            if idx == 0:
                if isinstance(img, np.ndarray):
                    logger.info(f"转换后图像类型: numpy.ndarray")
                    logger.info(f"转换后图像形状: {img.shape}")
                    logger.info(f"转换后图像值范围: [{np.min(img)}, {np.max(img)}]")
                elif isinstance(img, torch.Tensor):
                    logger.info(f"转换后图像类型: torch.Tensor")
                    logger.info(f"转换后图像形状: {img.shape}")
                    logger.info(f"转换后图像值范围: [{img.min().item():.4f}, {img.max().item():.4f}]")
                    
                    # 检查是否已归一化
                    max_val = img.max().item()
                    if max_val > 10.0:
                        logger.error(f"图像可能未正确归一化! 最大值: {max_val:.4f}")
                    else:
                        logger.info(f"图像已正确归一化，值范围正常")
        else:
            if idx == 0:
                logger.warning("未应用数据增强转换!")
        
        # 确保图像是 [C, H, W] 格式
        if isinstance(img, np.ndarray):
            if idx == 0:
                logger.info("将numpy数组转换为torch张量...")
                logger.info(f"转换前图像形状: {img.shape}")
                logger.info(f"转换前图像值范围: [{np.min(img)}, {np.max(img)}]")
                
                # 检查是否需要归一化
                if np.max(img) > 10.0:
                    logger.warning("图像未归一化，手动应用归一化...")
                    img = img.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img = (img - mean) / std
                    logger.info(f"手动归一化后图像值范围: [{np.min(img)}, {np.max(img)}]")
            
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # 确保是float类型
            
            if idx == 0:
                logger.info(f"转换后图像形状: {img.shape}")
                logger.info(f"转换后图像值范围: [{img.min().item():.4f}, {img.max().item():.4f}]")
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] != 3:  # 如果通道维不在前面
                if idx == 0:
                    logger.info(f"调整张量维度顺序: {img.shape} -> {img.permute(2, 0, 1).shape}")
                img = img.permute(2, 0, 1)
            
            # 检查是否需要归一化
            if idx == 0 and img.max() > 10.0:
                logger.warning(f"张量未归一化，手动应用归一化... 当前最大值: {img.max().item():.4f}")
                img = img / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img - mean) / std
                logger.info(f"手动归一化后张量值范围: [{img.min().item():.4f}, {img.max().item():.4f}]")
            
            img = img.float()  # 确保是float类型
        
        # 将掩码转换为张量
        if isinstance(semantic_mask, np.ndarray):
            target = torch.from_numpy(semantic_mask).long()
        else:
            target = semantic_mask.long()
        
        # 打印调试信息
        if idx == 0:  # 只打印第一个样本的信息
            print(f"[DEBUG] VOC数据集样本 - 图像ID: {img_id}")
            print(f"[DEBUG] 图像形状: {img.shape}, 范围: [{img.min().item()}, {img.max().item()}]")
            print(f"[DEBUG] 掩码形状: {target.shape}, 范围: [{target.min().item()}, {target.max().item()}]")
            print(f"[DEBUG] 掩码中的唯一类别: {torch.unique(target)}")
        
        # 根据模型类型准备目标
        if self.model_type == 'detr':
            target_dict = {
                'semantic_mask': target,  # 语义分割掩码
                'labels': torch.unique(target),  # 存在的类别标签
                'image_id': img_id
            }
            return {
                'image': img,
                'target': target_dict,
                'image_id': img_id
            }
        elif self.model_type in ['unet', 'saunet', 'cnn', 'deeplabv3', 'deeplabv3plus']:
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

def dataloader(root, split='train', batch_size=16, num_workers=4, transform=None, transform_config=None, model_type='detr', task_type='semantic', return_dataset=False, img_size=(640, 640)):
    """创建VOC数据集加载器
    
    参数:
        root (str): 数据集根目录
        split (str): 数据集划分，可选'train'或'val'
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        transform (callable, optional): 数据增强和预处理
        transform_config (dict, optional): 数据增强配置
        model_type (str): 模型类型，支持 'detr'、'unet'、'saunet'、'cnn'、'deeplabv3'、'deeplabv3plus'
        task_type (str): 任务类型，支持 'semantic'
        return_dataset (bool): 是否返回数据集对象而不是数据加载器
        img_size (tuple): 图像尺寸，默认为(480, 480)
    """
    logger = logging.getLogger('MYSEGX.dataloader')
    
    # 记录初始参数
    logger.info(f"创建数据加载器 - 分割类型: {task_type}, 模型类型: {model_type}, 数据集分割: {split}")
    logger.info(f"数据集根目录: {root}")
    logger.info(f"批次大小: {batch_size}, 工作进程数: {num_workers}")
    logger.info(f"图像尺寸: {img_size}")
    
    # 记录transform_config详情
    if transform_config is not None:
        logger.info(f"传入的transform_config: {transform_config}")
        
        # 处理transform_config为列表的情况
        if isinstance(transform_config, list):
            logger.info("transform_config是列表格式")
            
            # 检查列表中是否包含Normalize
            has_normalize = False
            for item in transform_config:
                if isinstance(item, dict) and item.get('name') == 'Normalize':
                    has_normalize = True
                    logger.info(f"找到normalize配置: {item}")
                    break
            
            if not has_normalize:
                logger.warning("transform_config列表中缺少Normalize配置，添加默认配置")
                transform_config.append({
                    'name': 'Normalize',
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225)
                })
                logger.info(f"添加默认normalize配置后的transform_config: {transform_config}")
        # 处理transform_config为字典的情况
        elif isinstance(transform_config, dict):
            if 'normalize' in transform_config:
                logger.info(f"normalize配置: {transform_config['normalize']}")
            else:
                logger.warning("transform_config字典中缺少normalize配置!")
                # 添加默认normalize配置
                transform_config['normalize'] = {
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225)
                }
                logger.info(f"默认normalize配置: {transform_config['normalize']}")
        else:
            logger.warning(f"transform_config类型不支持: {type(transform_config)}")
    else:
        logger.warning("未提供transform_config!")
        transform_config = {
            'normalize': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225)
            }
        }
        logger.info(f"创建默认transform_config: {transform_config}")
    
    # 确保使用正确的任务类型
    assert task_type == 'semantic', "此数据加载器仅支持语义分割任务"
    
    # 确保模型类型受支持
    model_type = model_type.lower()  # 转换为小写以确保一致性
    assert model_type in ['detr', 'unet', 'saunet', 'cnn', 'deeplabv3', 'deeplabv3plus'], f"不支持的模型类型: {model_type}"
    
    # 从配置中获取图像尺寸（如果提供）
    if transform_config and 'size' in transform_config:
        img_size = transform_config['size']
        logger.info(f"从transform_config获取图像尺寸: {img_size}")
    
    # 构建数据增强
    if transform_config is not None:
        logger.info("构建数据增强流水线...")
        transform = build_transforms(transform_config, train=(split=='train'))
        
        # 记录构建的transform
        if hasattr(transform, 'transforms'):
            logger.info("数据增强流水线包含以下转换:")
            for i, t in enumerate(transform.transforms):
                logger.info(f"  {i+1}. {t.__class__.__name__}")
                # 特别检查是否包含Normalize
                if t.__class__.__name__ == 'Normalize':
                    logger.info(f"    - 均值: {t.mean}")
                    logger.info(f"    - 标准差: {t.std}")
    else:
        logger.warning("未构建数据增强流水线!")
    
    logger.info(f"创建VOCSegmentation数据集 - 分割: {split}, 模型类型: {model_type}")
    dataset = VOCSegmentation(
        root=root,
        split=split,
        transform=transform,
        model_type=model_type,
        img_size=img_size
    )
    
    # 检查数据集样本
    logger.info(f"数据集大小: {len(dataset)}")
    if len(dataset) > 0:
        # 获取第一个样本进行检查
        sample = dataset[0]
        logger.info("检查第一个样本:")
        logger.info(f"  图像形状: {sample['image'].shape}")
        logger.info(f"  图像数据类型: {sample['image'].dtype}")
        logger.info(f"  图像值范围: [{sample['image'].min().item():.4f}, {sample['image'].max().item():.4f}]")
        
        # 特别检查图像是否已归一化
        max_val = sample['image'].max().item()
        if max_val > 10.0:
            logger.error(f"图像未正确归一化! 最大值: {max_val:.4f}")
        else:
            logger.info(f"图像已正确归一化，值范围正常")
        
        if isinstance(sample['target'], torch.Tensor):
            logger.info(f"  目标形状: {sample['target'].shape}")
            logger.info(f"  目标数据类型: {sample['target'].dtype}")
            logger.info(f"  目标值范围: [{sample['target'].min().item()}, {sample['target'].max().item()}]")
        elif isinstance(sample['target'], dict):
            logger.info(f"  目标类型: 字典")
            for k, v in sample['target'].items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"    {k} 形状: {v.shape}")
                    logger.info(f"    {k} 数据类型: {v.dtype}")
                    if v.numel() > 0:  # 确保张量不为空
                        logger.info(f"    {k} 值范围: [{v.min().item()}, {v.max().item()}]")
    
    if return_dataset:
        logger.info("返回数据集对象")
        return dataset
    
    logger.info(f"创建数据加载器 - 批次大小: {batch_size}, 工作进程数: {num_workers}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train'),
    )
    
    # 检查第一个批次
    try:
        logger.info("尝试获取第一个批次进行检查...")
        batch = next(iter(dataloader))
        logger.info(f"批次结构: {list(batch.keys())}")
        logger.info(f"批次图像形状: {batch['image'].shape}")
        logger.info(f"批次图像数据类型: {batch['image'].dtype}")
        logger.info(f"批次图像值范围: [{batch['image'].min().item():.4f}, {batch['image'].max().item():.4f}]")
        
        # 特别检查批次图像是否已归一化
        batch_max_val = batch['image'].max().item()
        if batch_max_val > 10.0:
            logger.error(f"批次图像未正确归一化! 最大值: {batch_max_val:.4f}")
        else:
            logger.info(f"批次图像已正确归一化，值范围正常")
            
        if isinstance(batch['target'], torch.Tensor):
            logger.info(f"批次目标形状: {batch['target'].shape}")
            logger.info(f"批次目标数据类型: {batch['target'].dtype}")
            logger.info(f"批次目标值范围: [{batch['target'].min().item()}, {batch['target'].max().item()}]")
        elif isinstance(batch['target'], list):
            logger.info(f"批次目标类型: 列表，长度: {len(batch['target'])}")
            if len(batch['target']) > 0:
                first_target = batch['target'][0]
                if isinstance(first_target, dict):
                    logger.info(f"  第一个目标结构: {list(first_target.keys())}")
                    for k, v in first_target.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"    {k} 形状: {v.shape}")
                            logger.info(f"    {k} 数据类型: {v.dtype}")
                            if v.numel() > 0:  # 确保张量不为空
                                logger.info(f"    {k} 值范围: [{v.min().item()}, {v.max().item()}]")
    except Exception as e:
        logger.warning(f"检查第一个批次时出错: {str(e)}")
    
    logger.info(f"数据加载器创建完成，包含 {len(dataloader)} 个批次")
    return dataloader
