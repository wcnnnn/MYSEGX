"""数据增强和预处理模块"""

import random
import math
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F
import logging

class Compose:
    """组合多个转换操作
    
    参数:
        transforms (list): 转换操作列表
    """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, **kwargs):
        image = kwargs.get('image')
        mask = kwargs.get('mask', None)
        
        for t in self.transforms:
            result = t(image=image, mask=mask)
            if isinstance(result, dict):
                image = result['image']
                # 只有当原始输入包含mask时才更新mask
                if mask is not None and 'mask' in result:
                    mask = result['mask']
            else:
                # 如果转换操作直接返回图像
                image = result
        
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class Resize:
    """调整图像大小
    
    参数:
        size (tuple): 目标尺寸 (height, width)
    """
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def __call__(self, image, mask=None):
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class RandomResize:
    """随机调整大小
    
    参数:
        min_size (int): 最小尺寸
        max_size (int): 最大尺寸
    """
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        
    def __call__(self, image, mask=None):
        h, w = image.shape[:2]
        size = random.randint(self.min_size, self.max_size)
        scale = size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class RandomRotation:
    """随机旋转
    
    参数:
        degrees (float): 最大旋转角度
        expand (bool): 是否扩展图像以容纳整个旋转后的图像
    """
    def __init__(self, degrees=10, expand=False):
        self.degrees = degrees
        self.expand = expand
        
    def __call__(self, image, mask=None):
        angle = random.uniform(-self.degrees, self.degrees)
        height, width = image.shape[:2]
        center = (width/2, height/2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if self.expand:
            # 计算旋转后的图像大小
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            
            # 调整变换矩阵
            M[0, 2] += (new_width / 2) - center[0]
            M[1, 2] += (new_height / 2) - center[1]
            width, height = new_width, new_height
        
        image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (width, height), flags=cv2.INTER_NEAREST)
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class RandomScale:
    """随机缩放
    
    参数:
        min_scale (float): 最小缩放比例
        max_scale (float): 最大缩放比例
    """
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(self, image, mask=None):
        scale = random.uniform(self.min_scale, self.max_scale)
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class RandomCrop:
    """随机裁剪
    
    参数:
        size (tuple): 裁剪尺寸 (height, width)
    """
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else tuple(size)
        
    def __call__(self, image, mask=None):
        h, w = image.shape[:2]
        new_h, new_w = self.size
        
        if h > new_h and w > new_w:
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            
            image = image[top:top + new_h, left:left + new_w]
            if mask is not None:
                mask = mask[top:top + new_h, left:left + new_w]
                return {'image': image, 'mask': mask}
            return {'image': image}
        
        # 如果原图小于目标尺寸，先调整大小
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return {'image': image, 'mask': mask}
        return {'image': image}

class RandomHorizontalFlip:
    """随机水平翻转
    
    参数:
        p (float): 翻转概率
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, mask=None):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class ColorJitter:
    """颜色增强
    
    参数:
        brightness (float): 亮度调整范围
        contrast (float): 对比度调整范围
        saturation (float): 饱和度调整范围
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        
    def __call__(self, image, mask=None):
        if self.brightness > 0:
            factor = random.uniform(max(0, 1-self.brightness), 1+self.brightness)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
        if self.contrast > 0:
            factor = random.uniform(max(0, 1-self.contrast), 1+self.contrast)
            image = np.clip(127.5 + factor * (image - 127.5), 0, 255).astype(np.uint8)
            
        if self.saturation > 0:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            factor = random.uniform(max(0, 1-self.saturation), 1+self.saturation)
            image[:, :, 1] = np.clip(image[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

class Normalize:
    """标准化图像
    
    参数:
        mean (tuple): RGB均值
        std (tuple): RGB标准差
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
    def __call__(self, image, mask=None):
        logger = logging.getLogger('MYSEGX.normalize')
        
        # 记录输入图像信息
        if isinstance(image, np.ndarray):
            logger.debug(f"Normalize输入图像形状: {image.shape}")
            logger.debug(f"Normalize输入图像类型: {image.dtype}")
            logger.debug(f"Normalize输入图像值范围: [{np.min(image)}, {np.max(image)}]")
            
            # 检查输入图像是否已经归一化
            if np.max(image) > 10.0:
                logger.debug(f"输入图像未归一化，最大值: {np.max(image)}")
            else:
                logger.warning(f"输入图像可能已经归一化，最大值: {np.max(image)}")
        
        # 确保使用float32类型
        original_type = type(image)
        original_dtype = getattr(image, 'dtype', None)
        logger.debug(f"原始图像类型: {original_type}, dtype: {original_dtype}")
        
        # 归一化到[0,1]
        if np.max(image) > 10.0:  # 如果图像在[0,255]范围
            logger.debug("将图像从[0,255]归一化到[0,1]范围")
            image = image.astype(np.float32) / 255.0
            logger.debug(f"归一化到[0,1]后值范围: [{np.min(image)}, {np.max(image)}]")
        else:
            logger.debug("图像已经在[0,1]范围内，跳过除以255步骤")
            image = image.astype(np.float32)
        
        # 标准化处理
        logger.debug(f"应用均值: {self.mean}, 标准差: {self.std}")
        image = (image - self.mean) / self.std
        logger.debug(f"标准化后值范围: [{np.min(image)}, {np.max(image)}]")
        
        # 转换为torch tensor并确保是float32类型
        logger.debug("转换为torch.Tensor并调整通道顺序")
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        logger.debug(f"转换后形状: {image.shape}")
        logger.debug(f"转换后值范围: [{image.min().item():.4f}, {image.max().item():.4f}]")
        
        if mask is not None:
            mask = torch.from_numpy(mask).long()  # 对于掩码使用long类型
            logger.debug(f"掩码形状: {mask.shape}, 类型: {mask.dtype}")
        
        return {'image': image, 'mask': mask} if mask is not None else {'image': image}

def get_training_transform(size=(512, 512)):
    """获取训练集数据增强"""
    return Compose([
        Resize(size),
        RandomHorizontalFlip(),
        ColorJitter(),
        Normalize()
    ])

def get_validation_transform(size=(512, 512)):
    """获取验证集数据增强"""
    return Compose([
        Resize(size),
        Normalize()
    ])

def build_transforms(config, train=True):
    """根据配置构建数据增强流水线
    
    参数:
        config (dict或list): 数据增强配置，可以是字典或列表
        train (bool): 是否为训练模式
    """
    logger = logging.getLogger('MYSEGX.transforms')
    
    transforms = []
    
    # 处理列表格式的配置
    if isinstance(config, list):
        logger.info(f"使用列表格式的transform_config: {config}")
        
        # 创建转换操作映射
        transform_map = {
            'Resize': Resize,
            'RandomResize': RandomResize,
            'RandomRotation': RandomRotation,
            'RandomScale': RandomScale,
            'RandomCrop': RandomCrop,
            'RandomHorizontalFlip': RandomHorizontalFlip,
            'ColorJitter': ColorJitter,
            'Normalize': Normalize,
            'ToTensor': lambda: None  # 占位符，实际上不会使用
        }
        
        for item in config:
            if not isinstance(item, dict) or 'name' not in item:
                logger.warning(f"跳过无效的转换配置: {item}")
                continue
                
            transform_name = item['name']
            if transform_name not in transform_map:
                logger.warning(f"未知的转换操作: {transform_name}")
                continue
                
            # 对于ToTensor，我们跳过，因为Normalize已经包含了转换为tensor的功能
            if transform_name == 'ToTensor':
                logger.info("跳过ToTensor，因为Normalize已经包含了转换为tensor的功能")
                continue
                
            # 创建转换操作实例
            try:
                # 移除name字段，只保留参数
                params = {k: v for k, v in item.items() if k != 'name'}
                transform_cls = transform_map[transform_name]
                transform = transform_cls(**params)
                transforms.append(transform)
                logger.info(f"添加转换操作: {transform_name} 参数: {params}")
            except Exception as e:
                logger.error(f"创建转换操作 {transform_name} 失败: {str(e)}")
        
        # 确保最后一个操作是Normalize
        if not any(isinstance(t, Normalize) for t in transforms):
            logger.warning("转换流水线中缺少Normalize，添加默认Normalize")
            transforms.append(Normalize())
    
    # 处理字典格式的配置
    else:
        if train:
            # 随机调整大小
            if 'random_resize' in config:
                transforms.append(RandomResize(**config['random_resize']))
                
            # 随机旋转
            if 'random_rotation' in config:
                transforms.append(RandomRotation(**config['random_rotation']))
                
            # 随机缩放
            if 'random_scale' in config:
                transforms.append(RandomScale(**config['random_scale']))
                
            # 随机裁剪
            if 'random_crop' in config:
                # 确保size是元组格式
                crop_config = config['random_crop'].copy()
                if isinstance(crop_config['size'], list):
                    crop_config['size'] = tuple(crop_config['size'])
                transforms.append(RandomCrop(**crop_config))
                
            # 随机翻转
            if 'horizontal_flip' in config:
                transforms.append(RandomHorizontalFlip(**config['horizontal_flip']))
                
            # 颜色增强
            if 'color_jitter' in config:
                transforms.append(ColorJitter(**config['color_jitter']))
        
        # 调整大小
        if 'resize' in config:
            # 确保size是元组格式
            resize_config = config['resize'].copy()
            if isinstance(resize_config['size'], list):
                resize_config['size'] = tuple(resize_config['size'])
            transforms.append(Resize(**resize_config))
            
        # 标准化
        if 'normalize' in config:
            transforms.append(Normalize(**config['normalize']))
    
    # 记录最终的转换流水线
    logger.info("最终转换流水线:")
    for i, t in enumerate(transforms):
        logger.info(f"  {i+1}. {t.__class__.__name__}")
    
    return Compose(transforms)