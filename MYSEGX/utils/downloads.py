"""权重文件下载模块"""

import os
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# 预训练模型权重配置
MODEL_URLS = {
    # ResNet系列
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    
    # VGG系列
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    
    # MobileNet系列
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
}

def download_weights(model_name, save_dir=None):
    """下载模型权重文件
    
    参数:
        model_name (str): 模型名称
        save_dir (str, optional): 保存目录
    
    返回:
        str: 权重文件保存路径
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f'不支持的模型: {model_name}')
        
    url = MODEL_URLS[model_name]
    
    # 设置保存目录
    if save_dir is None:
        save_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成保存文件名
    filename = os.path.basename(url)
    save_path = save_dir / filename
    
    # 如果文件已存在则直接返回
    if save_path.exists():
        print(f'权重文件已存在: {save_path}')
        return str(save_path)
    
    # 下载文件
    print(f'正在下载 {model_name} 权重文件...')
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # 获取文件大小
    file_size = int(response.headers.get('content-length', 0))
    
    # 使用tqdm显示下载进度
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f'权重文件已保存到: {save_path}')
    return str(save_path)