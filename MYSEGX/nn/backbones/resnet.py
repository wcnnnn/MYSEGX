"""ResNet主干网络模块"""

import torch
import torch.nn as nn
from ...utils.downloads import download_weights

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, remove_classification=True):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        if not remove_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, 1000)
        else:
            self.avgpool = None
            self.fc = None
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)    # 1/4
        x2 = self.layer2(x1)   # 1/8
        x3 = self.layer3(x2)   # 1/16
        x4 = self.layer4(x3)   # 1/32
        
        if not self.avgpool:
            return [x1, x2, x3, x4]
        
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(pretrained=True, remove_classification=True):
    model = ResNet(BasicBlock, [2, 2, 2, 2], remove_classification)
    if pretrained:
        weights_path = download_weights('resnet18')
        # 加载预训练权重时忽略分类层
        if remove_classification:
            state_dict = torch.load(weights_path)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('avgpool', 'fc'))}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(torch.load(weights_path))
    return model

def ResNet34(pretrained=True, remove_classification=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3], remove_classification)
    if pretrained:
        weights_path = download_weights('resnet34')
        if remove_classification:
            state_dict = torch.load(weights_path)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('avgpool', 'fc'))}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(torch.load(weights_path))
    return model

def ResNet50(pretrained=True, remove_classification=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], remove_classification)
    if pretrained:
        weights_path = download_weights('resnet50')
        if remove_classification:
            state_dict = torch.load(weights_path)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('avgpool', 'fc'))}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(torch.load(weights_path))
    return model

def ResNet101(pretrained=True, remove_classification=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3], remove_classification)
    if pretrained:
        weights_path = download_weights('resnet101')
        if remove_classification:
            state_dict = torch.load(weights_path)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('avgpool', 'fc'))}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(torch.load(weights_path))
    return model