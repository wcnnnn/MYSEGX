"""CNN分割网络模型"""

from .cnn import CNNSegNet

def build_cnn(config):
    """构建CNN分割网络
    
    参数:
        config: 配置字典，包含模型参数
        
    返回:
        model: CNNSegNet模型实例
    """
    model = CNNSegNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes']
    )
    return model

__all__ = ['CNNSegNet', 'build_cnn']
