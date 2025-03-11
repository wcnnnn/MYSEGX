"""SAUNet模型实现"""

from .saunet import SA_UNet

def build_saunet(config):
    """构建SAUNet模型
    
    参数:
        config: 配置字典，包含模型参数
        
    返回:
        model: SAUNet模型实例
    """
    model = SA_UNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes'],
        bilinear=config['model'].get('bilinear', True)
    )
    return model

__all__ = ['SA_UNet', 'build_saunet']