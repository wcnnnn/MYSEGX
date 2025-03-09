"""UNet模型实现"""

from .unet import UNet

def build_unet(config):
    """构建UNet模型
    
    参数:
        config: 配置字典，包含模型参数
        
    返回:
        model: UNet模型实例
    """
    model = UNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes'],
        bilinear=config['model'].get('bilinear', True)
    )
    return model

__all__ = ['UNet', 'build_unet']