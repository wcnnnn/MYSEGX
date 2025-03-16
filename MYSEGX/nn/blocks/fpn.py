"""FPN模块 - 特征金字塔网络"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class FPN(nn.Module):
    """特征金字塔网络
    
    用于生成多尺度特征图,支持自顶向下的特征融合。
    
    参数:
        in_channels_list (list): 输入特征通道数列表
        out_channels (int): 输出特征通道数
        use_relu (bool): 是否使用ReLU激活
        use_gn (bool): 是否使用GroupNorm替代BatchNorm
    """
    def __init__(self, in_channels_list, out_channels=256, use_relu=True, use_gn=False):
        super().__init__()
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        norm_layer = nn.GroupNorm(32, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
        
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                norm_layer,
                nn.ReLU(inplace=True) if use_relu else nn.Identity()
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化FPN: in_channels={in_channels_list}, out_channels={out_channels}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (list[Tensor]): 输入特征图列表 [B, C, H, W]
            
        返回:
            results (list[Tensor]): 输出特征图列表 [B, out_channels, H, W]
        """
        if torch.is_grad_enabled():
            for i, feat in enumerate(x):
                logger.debug(f"FPN输入{i}: shape={feat.shape}, 范围=[{feat.min():.3f}, {feat.max():.3f}]")
                
                # 检查输入是否有异常值
                if torch.isnan(feat).any():
                    logger.warning(f"输入特征{i}包含NaN值!")
                    x[i] = torch.nan_to_num(feat, nan=0.0)
                if torch.isinf(feat).any():
                    logger.warning(f"输入特征{i}包含Inf值!")
                    x[i] = torch.nan_to_num(feat, posinf=100.0, neginf=-100.0)
        
        # 获取有效的输入特征
        x = [feat for feat in x if feat is not None]
        
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for feat, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_lateral = inner_block(feat)
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:],
                                         mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))
            
        if torch.is_grad_enabled():
            for i, feat in enumerate(results):
                logger.debug(f"FPN输出{i}: shape={feat.shape}, 范围=[{feat.min():.3f}, {feat.max():.3f}]")
                
                # 检查输出是否有异常值
                if torch.isnan(feat).any():
                    logger.warning(f"输出特征{i}包含NaN值!")
                    results[i] = torch.nan_to_num(feat, nan=0.0)
                if torch.isinf(feat).any():
                    logger.warning(f"输出特征{i}包含Inf值!")
                    results[i] = torch.nan_to_num(feat, posinf=100.0, neginf=-100.0)
        
        return results
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 