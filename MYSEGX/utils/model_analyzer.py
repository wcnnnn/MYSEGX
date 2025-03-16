"""分割模型分析工具"""

import torch
import logging
from typing import Dict, Tuple, Union
try:
    from ptflops import get_model_complexity_info
    from torchinfo import summary
    HAVE_ANALYSIS_TOOLS = True
except ImportError:
    HAVE_ANALYSIS_TOOLS = False
    logging.warning("未安装ptflops或torchinfo，请安装：pip install ptflops torchinfo")

def analyze_model(model: torch.nn.Module, 
                 input_size: Tuple[int, ...] = (1, 3, 640, 640), 
                 show_details: bool = False,  # 默认不显示详细信息
                 print_table: bool = True) -> Dict:  # 添加参数控制是否打印表格
    """分析模型结构和计算复杂度
    
    参数:
        model: 要分析的模型
        input_size: 输入尺寸
        show_details: 是否显示每层的详细信息
        print_table: 是否打印统计表格
    """
    if not HAVE_ANALYSIS_TOOLS:
        raise ImportError("请先安装必要的分析工具：pip install ptflops torchinfo")

    # 将模型设置为评估模式
    model.eval()
    
    try:
        # 使用ptflops计算模型复杂度
        macs, params = get_model_complexity_info(
            model, 
            input_size[1:],
            as_strings=False,
            print_per_layer_stat=show_details,
            verbose=show_details  # 根据show_details参数控制是否显示详细信息
        )
        
        # 计算参数量
        total_params = params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算GFLOPs
        gflops = macs * 2 / (1024 ** 3) if macs is not None else None
        
        if print_table:  # 只在需要时打印表格
            # 格式化数字显示
            def format_number(n: Union[int, float]) -> str:
                if n is None:
                    return "N/A"
                if n >= 1e9:
                    return f"{n/1e9:.2f}B"
                elif n >= 1e6:
                    return f"{n/1e6:.2f}M"
                elif n >= 1e3:
                    return f"{n/1e3:.2f}K"
                return str(n)
            
            # 打印表格形式的模型统计信息
            print("\n" + "="*80)
            print(f"{'模型统计信息':^80}")
            print("="*80)
            print(f"| {'指标':<20} | {'数值':<15} | {'单位':<20} |")
            print("-"*80)
            print(f"| {'总参数量':<20} | {format_number(total_params):<15} | {'参数':<20} |")
            print(f"| {'可训练参数':<20} | {format_number(trainable_params):<15} | {'参数':<20} |")
            print(f"| {'MACs':<20} | {format_number(macs):<15} | {'MAC运算':<20} |")
            print(f"| {'计算量':<20} | {format_number(gflops):<15} | {'GFLOPs':<20} |")
            print("="*80)
            
            if show_details:  # 只在需要时打印详细结构
                # 打印模型层级结构表格
                print("\n" + "="*70)
                print(f"{'模型层级结构':^70}")
                print("="*70)
                print(f"| {'模块':<30} | {'参数量':<15} | {'参数占比':<10} |")
                print("-"*70)
                
                def get_module_stats(module):
                    params = sum(p.numel() for p in module.parameters())
                    return params
                
                def print_module_stats(name, module, total_params, total_macs, level=0):
                    prefix = "  " * level
                    params = get_module_stats(module)
                    param_percent = (params / total_params * 100) if total_params > 0 else 0
                    print(f"| {prefix}{name:<{30-len(prefix)}} | {format_number(params):<15} | {param_percent:>9.2f}% |")
                    for child_name, child in module.named_children():
                        print_module_stats(child_name, child, total_params, total_macs, level + 1)
                
                print_module_stats(model.__class__.__name__, model, total_params, macs)
                print("="*70)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'macs': macs,
            'gflops': gflops,
            'model_summary': None
        }
        
    except Exception as e:
        logging.error(f"模型分析过程中出错: {str(e)}")
        # 尝试获取基本的模型信息
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 打印基本信息
            print("\n" + "="*80)
            print(f"{'模型基本信息 (分析过程中出错)':^80}")
            print("="*80)
            print(f"| {'指标':<20} | {'数值':<15} | {'单位':<20} |")
            print("-"*80)
            print(f"| {'总参数量':<20} | {format_number(total_params):<15} | {'参数':<20} |")
            print(f"| {'可训练参数':<20} | {format_number(trainable_params):<15} | {'参数':<20} |")
            print("="*80)
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'macs': None,
                'gflops': None,
                'model_summary': None
            }
        except Exception as inner_e:
            logging.error(f"获取基本模型信息也失败: {str(inner_e)}")
            return {
                'total_params': None,
                'trainable_params': None,
                'macs': None,
                'gflops': None,
                'model_summary': None
            }