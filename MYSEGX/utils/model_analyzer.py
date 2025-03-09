"""模型分析工具"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from tabulate import tabulate
from ptflops import get_model_complexity_info

def get_module_info(module: nn.Module, input_shape: Union[torch.Size, Tuple]) -> Dict:
    """获取模块的详细信息
    Args:
        module: PyTorch模块
        input_shape: 输入形状
    Returns:
        Dict: 包含模块信息的字典
    """
    # 获取模块类型
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    
    # 计算参数量
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # 获取模块参数配置
    arguments = ""
    if isinstance(module, nn.Conv2d):
        arguments = f"{module.in_channels}x{module.out_channels}x{module.kernel_size[0]}x{module.kernel_size[1]}"
    elif isinstance(module, nn.Linear):
        arguments = f"{module.in_features}x{module.out_features}"
    elif isinstance(module, nn.MaxPool2d):
        arguments = f"{module.kernel_size}x{module.stride}"
    
    # 估算FLOPs
    flops = 0
    if isinstance(module, nn.Conv2d):
        # Conv2d层的FLOPs计算
        output_height = ((input_shape[-2] + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1)
        output_width = ((input_shape[-1] + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1)
        flops = output_height * output_width * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
    elif isinstance(module, nn.Linear):
        # Linear层的FLOPs计算
        flops = module.in_features * module.out_features
    
    return {
        'module': class_name,
        'params': params,
        'arguments': arguments,
        'flops': flops
    }

def analyze_model(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 640, 640), show_details: bool = True) -> Dict:
    """分析模型结构和计算复杂度
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        show_details: 是否显示详细信息
        
    Returns:
        Dict: 包含模型分析结果的字典
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 使用ptflops计算FLOPs
    macs, params = get_model_complexity_info(
        model, input_size[1:], as_strings=False,
        print_per_layer_stat=False, verbose=False
    )
    
    # 准备数据结构存储分析结果
    layers_info = []
    total_params = params
    total_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops = macs * 2  # MACs to FLOPs conversion
    
    def is_key_layer(module: nn.Module, class_name: str) -> bool:
        """判断是否为关键层"""
        key_layers = {
            'ResNet',  # 主干网络作为一个整体
            'Transformer',  # Transformer块
            'TransformerEncoder',  # Transformer编码器
            'TransformerDecoder',  # Transformer解码器
            'MultiheadAttention',  # 多头注意力层
            'PositionalEncoding',  # 位置编码层
            'FPNBlock', 'MaskHead',  # 特征金字塔和掩码头
            'UNet',  # UNet模型
            'DoubleConv',  # 双卷积块
            'Down',  # 下采样块
            'Up',  # 上采样块
            'OutConv'  # 输出卷积
        }
        
        # 确保编码器和解码器层被识别
        if 'Encoder' in class_name or 'Decoder' in class_name:
            return True
        
        # 只过滤掉基础的构建块
        if class_name in {'BasicBlock', 'Bottleneck', 'Conv2d'}:
            return False
        
        return class_name in key_layers
    
    def register_hook(module: nn.Module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            
            # 只记录关键层
            if not is_key_layer(module, class_name):
                return
            
            module_idx = len(layers_info)
            
            # 处理输入
            if isinstance(input, tuple):
                input_shape = input[0].shape if isinstance(input[0], torch.Tensor) else input[0]
            else:
                input_shape = input
            
            m_info = get_module_info(module, input_shape)
            
            # 获取来自哪一层的信息
            from_layers = []
            if hasattr(module, 'from_layers'):
                from_layers = module.from_layers
            
            info = {
                'idx': module_idx,
                'from': str(from_layers) if from_layers else '-',
                'n': 1,
                'params': m_info['params'],
                'module': m_info['module'],
                'arguments': m_info['arguments'],
                'flops': m_info['flops']
            }
            
            layers_info.append(info)
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not isinstance(module, nn.ModuleDict):
            hooks.append(module.register_forward_hook(hook))
    
    # 注册钩子
    hooks = []
    model.apply(register_hook)
    
    # 进行一次前向传播
    device = next(model.parameters()).device
    x = torch.zeros(input_size).to(device)
    try:
        model(x)
    except Exception as e:
        print(f"警告：模型前向传播时出现错误，但将继续分析可用的层信息。错误信息：{str(e)}")
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    if show_details and layers_info:
        # 准备表格数据
        headers = ['Layer', 'From', 'n', 'Parameters', 'Module Type', 'Arguments']
        table_data = []
        for layer in layers_info:
            # 使用科学计数法显示大于1000的参数量
            params_str = f"{layer['params']:,d}" if layer['params'] < 1000 else f"{layer['params']/1000:.1f}K"
            table_data.append([
                f"{layer['idx']:>3d}",  # 右对齐，固定宽度
                f"{layer['from']:<8}",  # 左对齐，固定宽度
                f"{layer['n']:>2d}",   # 右对齐
                f"{params_str:>10}",   # 右对齐，固定宽度
                f"{layer['module']:<20}",  # 左对齐，固定宽度
                f"{layer['arguments']}"
            ])
        
        # 打印表格，使用grid格式提高可读性
        print("\n模型结构:")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 格式化数字显示
    def format_number(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)
    
    # 打印总结信息
    gflops = total_flops / (1024 ** 3)  # 转换为GFLOPs
    print(f"\n模型统计信息:")
    print(f"├─ 层数: {len(layers_info):,d}")
    print(f"├─ 参数量: {format_number(total_params)}")
    print(f"├─ 梯度参数: {format_number(total_gradients)}")
    print(f"└─ 计算量: {gflops:.2f} GFLOPs")
    
    return {
        'num_layers': len(layers_info),
        'total_params': total_params,
        'total_gradients': total_gradients,
        'gflops': gflops,
        'layers_info': layers_info
    }