"""通用工具函数模块"""
import os
import yaml
import logging
import torch
import platform
from datetime import datetime

# ANSI颜色代码
COLORS = {
    'BLUE': '\033[38;5;39m',      # 浅蓝色
    'PURPLE': '\033[38;5;141m',    # 紫色
    'GREEN': '\033[38;5;48m',      # 翠绿色
    'YELLOW': '\033[38;5;227m',    # 金黄色
    'RED': '\033[38;5;196m',       # 鲜红色
    'ORANGE': '\033[38;5;208m',    # 橙色
    'CYAN': '\033[38;5;51m',       # 青色
    'MAGENTA': '\033[38;5;201m',   # 品红色
    'RESET': '\033[0m',            # 重置颜色
    'BOLD': '\033[1m',             # 粗体
    'DIM': '\033[2m',              # 暗色
}

def center_text(text, width):
    """将文本居中"""
    return text.center(width)

def print_banner():
    """打印项目启动标志"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 获取终端宽度，默认80
    terminal_width = 80
    try:
        terminal_width = os.get_terminal_size().columns
    except:
        pass
    
    # 设置banner的宽度
    banner_width = 70
    padding = (terminal_width - banner_width) // 2 if terminal_width > banner_width else 0
    pad = " " * padding
    
    c = COLORS  # 简化引用
    banner = f"""

{pad}{c['PURPLE']}{c['BOLD']}╔{'═' * (banner_width-2)}╗{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{' ' * (banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{center_text(f"{c['BLUE']}███╗   ███╗██╗   ██╗{c['CYAN']}███████╗███████╗ ██████╗{c['ORANGE']} ██╗  ██╗", banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{center_text(f"{c['BLUE']}████╗ ████║╚██╗ ██╔╝{c['CYAN']}██╔════╝██╔════╝██╔════╝{c['ORANGE']} ╚██╗██╔╝", banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{center_text(f"{c['BLUE']}██╔████╔██║ ╚████╔╝ {c['CYAN']}███████╗█████╗  ██║  ███╗{c['ORANGE']} ╚███╔╝ ", banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{center_text(f"{c['BLUE']}██║╚██╔╝██║  ╚██╔╝  {c['CYAN']}╚════██║██╔══╝  ██║   ██║{c['ORANGE']} ██╔██╗ ", banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{center_text(f"{c['BLUE']}██║ ╚═╝ ██║   ██║   {c['CYAN']}███████║███████╗╚██████╔╝{c['ORANGE']}██╔╝ ██╗", banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{center_text(f"{c['BLUE']}╚═╝     ╚═╝   ╚═╝   {c['CYAN']}╚══════╝╚══════╝ ╚═════╝{c['ORANGE']} ╚═╝  ╚═╝", banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}║{' ' * (banner_width-2)}║{c['RESET']}
{pad}{c['PURPLE']}{c['BOLD']}╚{'═' * (banner_width-2)}╝{c['RESET']}

{center_text(f"{c['YELLOW']}🚀 欢迎使用 {c['BOLD']}{c['BLUE']}MY{c['CYAN']}SEG{c['ORANGE']}X{c['RESET']}{c['YELLOW']} 图像分割框架!{c['RESET']} {c['GREEN']}🎯{c['RESET']}", terminal_width)}

{pad}{c['BLUE']}📌 系统信息:{c['RESET']}
{pad}   {c['DIM']}• 版本: {c['RESET']}{c['GREEN']}1.0.0{c['RESET']}
{pad}   {c['DIM']}• 启动时间: {c['RESET']}{c['GREEN']}{current_time}{c['RESET']}
{pad}   {c['DIM']}• Python版本: {c['RESET']}{c['GREEN']}{platform.python_version()}{c['RESET']}
{pad}   {c['DIM']}• PyTorch版本: {c['RESET']}{c['GREEN']}{torch.__version__}{c['RESET']}

{center_text(f"{c['PURPLE']}🌟 让我们开始图像分割的奇妙旅程吧! {c['ORANGE']}💫{c['RESET']}", terminal_width)}
{pad}{c['DIM']}{'═' * min(banner_width, 70)}{c['RESET']}
"""
    print(banner)

def setup_logger(name, log_file=None, level='INFO'):
    """设置日志记录器
    
    参数:
        name (str): 日志记录器名称
        log_file (str, optional): 日志文件路径
        level (str): 日志级别
    """
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_file):
    """加载配置文件
    
    参数:
        config_file (str): 配置文件路径
        
    返回:
        dict: 配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"加载配置文件 {config_file} 失败: {str(e)}")