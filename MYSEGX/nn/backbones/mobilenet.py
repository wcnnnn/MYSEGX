"""MobileNet主干网络模块"""

import torch
import torch.nn as nn
from ...utils.downloads import download_weights

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # 构建网络结构
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # stage 1
            [6, 24, 2, 2],   # stage 2
            [6, 32, 3, 2],   # stage 3
            [6, 64, 4, 2],   # stage 4
            [6, 96, 3, 1],   
            [6, 160, 3, 2],  # stage 5
            [6, 320, 1, 1],
        ]

        # 第一层卷积
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # 构建Inverted Residual块
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 最后一层卷积
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        prev_feature = None
        
        for i, block in enumerate(self.features):
            x = block(x)
            if isinstance(block, InvertedResidual) and block.stride == 2 and prev_feature is not None:
                features.append(prev_feature)
            prev_feature = x
            
        features.append(x)  # 添加最后一层特征
        return features

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # MobileNetV3-Small的实现将在后续版本中添加
        raise NotImplementedError("MobileNetV3-Small将在后续版本中实现")

class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # MobileNetV3-Large的实现将在后续版本中添加
        raise NotImplementedError("MobileNetV3-Large将在后续版本中实现")

def create_mobilenet_v2(pretrained=True):
    model = MobileNetV2()
    if pretrained:
        weights_path = download_weights('mobilenet_v2')
        model.load_state_dict(torch.load(weights_path))
    return model

def create_mobilenet_v3_small(pretrained=True):
    model = MobileNetV3Small()
    if pretrained:
        weights_path = download_weights('mobilenet_v3_small')
        model.load_state_dict(torch.load(weights_path))
    return model

def create_mobilenet_v3_large(pretrained=True):
    model = MobileNetV3Large()
    if pretrained:
        weights_path = download_weights('mobilenet_v3_large')
        model.load_state_dict(torch.load(weights_path))
    return model