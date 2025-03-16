"""ProtoNet模块 - 用于生成prototype masks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ProtoNet(nn.Module):
    """ProtoNet模块
    
    生成prototype masks,这些masks将与预测的系数组合生成实例掩码。
    
    参数:
        in_channels (int): 输入特征通道数
        proto_channels (int): prototype特征通道数
        num_protos (int): prototype的数量
        use_dcn (bool): 是否使用可变形卷积
    """
    def __init__(self, in_channels=256, proto_channels=256, num_protos=32, use_dcn=False):
        super().__init__()
        
        self.num_protos = num_protos
        self.proto_channels = proto_channels
        
        # 特征处理网络
        self.proto_net = nn.Sequential(
            nn.Conv2d(in_channels, proto_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(proto_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(proto_channels, proto_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(proto_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(proto_channels, proto_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(proto_channels),
            nn.ReLU(inplace=True),
            
            # 上采样到原始分辨率的1/4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 最后一层生成prototype masks
            nn.Conv2d(proto_channels, num_protos, kernel_size=1)
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化ProtoNet: in_channels={in_channels}, proto_channels={proto_channels}, num_protos={num_protos}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            proto_out (Tensor): prototype masks [B, num_protos, H*2, W*2]
        """
        if torch.is_grad_enabled():
            logger.debug(f"ProtoNet输入: shape={x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")
            
            # 检查输入是否有异常值
            if torch.isnan(x).any():
                logger.warning("输入特征包含NaN值!")
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isinf(x).any():
                logger.warning("输入特征包含Inf值!")
                x = torch.nan_to_num(x, posinf=100.0, neginf=-100.0)
        
        # 生成prototype masks
        proto_out = self.proto_net(x)
        
        # 对输出进行relu激活,确保非负
        proto_out = F.relu(proto_out)
        
        if torch.is_grad_enabled():
            logger.debug(f"ProtoNet输出: shape={proto_out.shape}, 范围=[{proto_out.min():.3f}, {proto_out.max():.3f}]")
            
            # 检查输出是否有异常值
            if torch.isnan(proto_out).any():
                logger.warning("输出包含NaN值!")
                proto_out = torch.nan_to_num(proto_out, nan=0.0)
            if torch.isinf(proto_out).any():
                logger.warning("输出包含Inf值!")
                proto_out = torch.nan_to_num(proto_out, posinf=100.0, neginf=-100.0)
        
        return proto_out
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 