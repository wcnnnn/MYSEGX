"""数据增强和预处理模块"""

import random
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F

class Compose:
    """组合多个转换操作
    
    参数:
        transforms (list): 转换操作列表
    """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, mask=None):
        for t in self.transforms:
            if mask is None:
                image = t(image)
            else:
                image, mask = t(image, mask)
        
        if mask is None:
            return {'image': image}
        return {'image': image, 'mask': mask}

class Resize:
    """调整图像和掩码大小
    
    参数:
        size (tuple): 目标尺寸 (height, width)
        interpolation (int): 插值方法
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
        
    def __call__(self, image, mask=None):
        image = cv2.resize(image, self.size[::-1], interpolation=self.interpolation)
        if mask is not None:
            mask = cv2.resize(mask, self.size[::-1], interpolation=cv2.INTER_NEAREST)
            return image, mask
        return image

class RandomCrop:
    """随机裁剪
    
    参数:
        size (tuple): 裁剪尺寸 (height, width)
    """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, mask):
        h, w = image.shape[:2]
        new_h, new_w = self.size
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        image = image[top:top + new_h, left:left + new_w]
        mask = mask[top:top + new_h, left:left + new_w]
        return image, mask

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
                return image, mask
        return image

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
        
        if mask is not None:
            return image, mask
        return image

class Normalize:
    """标准化
    
    参数:
        mean (tuple): 均值
        std (tuple): 标准差
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, mask=None):
        # 确保图像是float32类型
        image = image.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        
        # 广播均值和标准差到正确的形状
        if image.ndim == 3:  # (H, W, C)
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
        
        image = (image - mean) / std
        
        # 调整通道顺序 (H, W, C) -> (C, H, W)
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image).float()
        
        if mask is not None:
            mask = torch.from_numpy(mask).long()
            return image, mask
        return image

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

def build_transforms(train=True, size=(512, 512)):
    """构建数据增强流水线
    
    参数:
        train (bool): 是否为训练模式
        size (tuple): 输入图像大小 (height, width)
        
    返回:
        transforms (Compose): 数据增强流水线
    """
    transforms = []
    
    # 调整图像大小
    transforms.append(Resize(size))
    
    if train:
        # 训练时的数据增强
        transforms.append(RandomHorizontalFlip())
        transforms.append(ColorJitter())
    
    # 标准化
    transforms.append(Normalize())
    
    return Compose(transforms)