"""MYSEGX 高级 API 接口"""
import os
import torch
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from datetime import datetime
import importlib.util
import threading
import socket
import webbrowser
import subprocess
from tensorboard import program
from .models import build_model
from .engine.trainer import Trainer
from .utils.losses import DETRLoss,MaskRCNNLoss,CrossEntropyLoss,OHEMCrossEntropyLoss,YOLACTLoss
from .data.transforms import build_transforms
from .nn.modules.assigners.hungarian_assigner import HungarianAssigner
from .utils.general import load_config, setup_logger, print_banner
from .utils.model_analyzer import analyze_model
from .utils.plots import  plot_segmentation
from .utils.results import ResultSaver
from .data.Semantic_Segmentation.voc import collate_fn as voc_semantic_collate_fn
from .data.Instance_Segmentation.voc import collate_fn as voc_instance_collate_fn
from .data.Semantic_Segmentation.cityscapes import collate_fn as cityscapes_semantic_collate_fn
from .data.Instance_Segmentation.cityscapes import collate_fn as cityscapes_instance_collate_fn
from .data.Semantic_Segmentation.ade20k import collate_fn as ade20k_semantic_collate_fn
from torch.utils.data import Dataset
import sys
import numpy as np
import logging
import warnings
warnings.filterwarnings("ignore")
# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 将默认级别从DEBUG改为WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_debug.log')
    ]
)

def _find_dataset_module(task_type: str, dataset_name: str) -> Optional[str]:
    """查找数据集模块
    
    在指定任务类型目录下查找包含数据集名称的模块文件
    
    参数:
        task_type: 任务类型目录名 (如 'Semantic_Segmentation')
        dataset_name: 数据集名称 (如 'voc')
        
    返回:
        str: 模块路径，如果未找到则返回None
    """
    # 获取data目录的绝对路径
    data_dir = Path(__file__).parent / 'data'
    
    # 标准化任务类型名称
    task_map = {
        'semantic': 'Semantic_Segmentation',
        'instance': 'Instance_Segmentation', 
        'panoptic': 'Panoptic_Segmentation',
        '3d': '3D_Segmentation'
    }
    
    task_dir_name = task_map.get(task_type.lower(), task_type)
    task_dir = data_dir / task_dir_name
    
    if not task_dir.exists():
        return None
        
    # 直接查找对应的数据集文件
    dataset_file = task_dir / f"{dataset_name.lower()}.py"
    if dataset_file.exists():
        return str(dataset_file)
    
    # 回退到模糊匹配
    for file in task_dir.glob('*.py'):
        if dataset_name.lower() in file.stem.lower():
            return str(file)
    
    return None

def _load_dataloader(module_path: str) -> callable:
    """动态加载数据集的dataloader函数
    
    参数:
        module_path: 模块文件路径
        
    返回:
        callable: dataloader函数
    """
    # 获取模块名和规范路径
    module_name = Path(module_path).stem
    abs_path = str(Path(module_path).resolve())
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 获取dataloader函数
    if not hasattr(module, 'dataloader'):
        raise AttributeError(f"模块 {module_name} 中未找到 dataloader 函数")
    
    return module.dataloader

def _start_tensorboard(logdir: str, port: int = 6006) -> Optional[str]:
    from .utils.general import setup_logger
    logger = setup_logger('MYSEGX.tensorboard')
    
    # 设置TensorBoard的日志级别为WARNING，抑制调试信息
    import logging
    logging.getLogger('tensorboard').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    """
    在后台启动TensorBoard服务
    
    参数:
        logdir: TensorBoard日志目录
        port: 要使用的端口号
        
    返回:
        str: TensorBoard的URL，如果启动失败则返回None
    """
    def _is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    # 检查端口是否被占用，找到可用端口
    original_port = port
    while _is_port_in_use(port):
        port += 1
    if port != original_port:
        logger.info(f"端口 {original_port} 已被占用，使用端口 {port}")
    
    # 构建启动命令
    try:
        # 直接使用 tensorboard 模块而不是命令行
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
        url = tb.launch()
        import time
        time.sleep(1)  # 等待服务启动
        webbrowser.open(url)
        return url
    except Exception as e:
        logger.error(f"启动TensorBoard时出错: {str(e)}")
        return None

def train(
    config_path: Optional[Union[str, Path]] = None,
    resume_path: Optional[str] = None,
    # 任务参数
    task_type: str = 'semantic',  # 简化：只需要一个任务类型参数
    # 模型参数
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    nhead: Optional[int] = None,
    num_encoder_layers: Optional[int] = None,
    num_decoder_layers: Optional[int] = None,
    dim_feedforward: Optional[int] = None,
    dropout: Optional[float] = None,
    num_queries: Optional[int] = None,
    backbone_type: Optional[str] = None,
    # 数据集参数
    dataset_root: Optional[str] = None,
    dataset_split: Optional[str] = None,
    image_size: Optional[tuple] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    dataset: str = 'voc',
    # 训练参数
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    epochs: Optional[int] = None,
    lr_drop: Optional[int] = None,
    # 其他参数
    device: Optional[str] = None,
    distributed: bool = False,
    amp: bool = False,
    save_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    eval_interval: int = 1,
    save_interval: int = 1,
    visualize: bool = True,
    seed: Optional[int] = None,
    debug: bool = False,
    metrics_dir: Optional[str] = None,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """训练模型的高级接口"""
    # 打印项目启动banner
    print_banner()
    
    # 设置日志
    logger = setup_logger(
        'MYSEGX',
        log_file=os.path.join('logs', f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )

    # 处理配置
    if config_path is not None:
        config = load_config(config_path)
        logger.info(f"加载配置文件: {config_path}")
    else:
        # 确定任务目录
        task_dir = 'Semantic_Segmentation' if task_type == 'semantic' else 'Instance_Segmentation'
        
        # 查找模型配置模板
        model_config_dir = os.path.join('configs', task_dir, model_name or 'detr')
        if os.path.exists(model_config_dir):
            config_files = [f for f in os.listdir(model_config_dir) if f.endswith('.yaml')]
            if config_files:
                config_path = os.path.join(model_config_dir, config_files[0])
                config = load_config(config_path)
                logger.info(f"使用默认配置文件: {config_path}")
            else:
                raise ValueError(f"在 {model_config_dir} 中未找到配置文件")
        else:
            raise ValueError(f"未找到任务 {task_type} 的模型 {model_name or 'detr'} 配置目录")

    # 确保配置中有dataset部分
    if 'dataset' not in config:
        config['dataset'] = {}

    # 处理数据集路径
    if dataset_root is not None:
        # 使用绝对路径
        config['dataset']['root'] = os.path.abspath(dataset_root)
    elif 'root' in config['dataset']:
        # 如果配置文件中有路径，转换为绝对路径
        config['dataset']['root'] = os.path.abspath(config['dataset']['root'])
    else:
        # 默认路径
        config['dataset']['root'] = os.path.abspath('data/VOC2012')

    # 验证数据集路径是否存在
    if not os.path.exists(config['dataset']['root']):
        raise ValueError(f"数据集路径不存在: {config['dataset']['root']}")

    logger.info(f"使用数据集路径: {config['dataset']['root']}")
    
    # 确保配置中有model部分
    if 'model' not in config:
        config['model'] = {}

    # 更新模型参数
    if model_name is not None:
        config['model']['name'] = model_name

    # 统一设置类别数量
    if num_classes is not None:
        logger.info(f"设置模型类别数为: {num_classes}")
        # 更新所有可能的类别数配置
        config['model']['num_classes'] = num_classes
        config['model']['n_classes'] = num_classes  # 某些模型使用n_classes
        if 'loss' in config:
            config['loss']['num_classes'] = num_classes
    else:
        # 如果未指定，则根据数据集类型设置默认值
        if dataset.lower() == 'cityscapes':
            default_classes = 19
        elif dataset.lower() == 'ade20k':
            default_classes = 151  # ADE20K有150个类别加1个背景类
        else:  # VOC数据集
            default_classes = 21
        logger.info(f"使用默认类别数: {default_classes}")
        config['model']['num_classes'] = default_classes
        config['model']['n_classes'] = default_classes

    # 根据模型类型更新特定参数
    if config['model']['name'] == 'detr':
        if num_classes is not None: 
            config['model']['num_classes'] = num_classes
        if num_queries is not None: 
            config['model']['num_queries'] = num_queries
        if backbone_type is not None: 
            config['model']['backbone_type'] = backbone_type
    elif config['model']['name'] == 'unet':
        if num_classes is not None: 
            config['model']['n_classes'] = num_classes
    elif config['model']['name'] == 'saunet':
        if num_classes is not None: 
            config['model']['num_classes'] = num_classes
    elif config['model']['name'] == 'cnn':
        if num_classes is not None: 
            config['model']['num_classes'] = num_classes
    elif config['model']['name'] in ['deeplabv3', 'deeplabv3plus']:
        if num_classes is not None: 
            config['model']['num_classes'] = num_classes
        else:
            # 确保DeepLab模型使用正确的类别数量
            config['model']['num_classes'] = 19 if dataset.lower() == 'cityscapes' else 21
    
    # 更新数据集参数
    if image_size is not None:
        config['dataset']['size'] = image_size
    if batch_size is not None:
        config['train']['batch_size'] = batch_size
    if num_workers is not None:
        config['train']['num_workers'] = num_workers
    if learning_rate is not None:
        config['train']['learning_rate'] = learning_rate
    if weight_decay is not None:
        config['train']['weight_decay'] = weight_decay
    if epochs is not None:
        config['train']['epochs'] = epochs
    if lr_drop is not None:
        config['train']['lr_drop'] = lr_drop
    
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # 设置设备和分布式训练
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    if distributed and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl')
        model = torch.nn.parallel.DistributedDataParallel(model)
        logger.info(f"使用分布式训练，设备数量: {torch.cuda.device_count()}")
    else:
        logger.info(f"使用设备: {device}")
    
    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if amp else None
    if amp:
        logger.info("启用混合精度训练")
    
    # 设置保存目录
    if save_dir:
        result_saver = ResultSaver(base_dir=save_dir)
    else:
        result_saver = ResultSaver(base_dir=os.path.join('results', config['model']['name']))
    
    # 设置指标保存目录
    metrics_base_dir = metrics_dir or os.path.join(result_saver.exp_dir, 'metrics')
    
    # 获取类别名称
    if class_names:
        names = class_names
    elif 'names' in config.get('dataset', {}):
        names = config['dataset']['names']
    else:
        names = [f'class_{i}' for i in range(config['model'].get('num_classes', config['model'].get('n_classes', 21)))]
    
    # 创建结果保存器
    result_saver = ResultSaver(base_dir=os.path.join('results', config['model']['name']))
    result_saver.save_config(config)
    logger.info(f"结果将保存在: {result_saver.exp_dir}")
    
    # 查找数据集模块
    module_path = _find_dataset_module(task_type, dataset)
    if not module_path:
        raise ValueError(f"未找到数据集 {dataset} 的实现")
    
    # 将数据集目录添加到Python路径
    module_dir = str(Path(module_path).parent.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    # 确保配置中有task部分
    if 'task' not in config:
        config['task'] = {
            'type': task_type,
            'model_type': config['model']['name']
        }

    # 安全获取transform配置
    train_transform_config = config.get('dataset', {}).get('transform', {}).get('train', {})
    val_transform_config = config.get('dataset', {}).get('transform', {}).get('val', {})
    
    # 加载dataloader函数
    try:
        # 根据数据集类型选择正确的dataloader
        if dataset.lower() == 'voc':
            if task_type.lower() == 'semantic':
                from MYSEGX.data.Semantic_Segmentation.voc import dataloader as dataloader_fn
                collate_fn = voc_semantic_collate_fn
            elif task_type.lower() == 'instance':
                from MYSEGX.data.Instance_Segmentation.voc import dataloader as dataloader_fn
                collate_fn = voc_instance_collate_fn
            else:
                raise ValueError(f"VOC数据集不支持 {task_type} 任务类型")
        elif dataset.lower() == 'cityscapes':
            if task_type.lower() == 'semantic':
                from MYSEGX.data.Semantic_Segmentation.cityscapes import dataloader as dataloader_fn
                collate_fn = cityscapes_semantic_collate_fn
            elif task_type.lower() == 'instance':
                from MYSEGX.data.Instance_Segmentation.cityscapes import dataloader as dataloader_fn
                collate_fn = cityscapes_instance_collate_fn
                # 更新类别数为Cityscapes实例分割的8个类别
                if num_classes is None:
                    config['model']['num_classes'] = 8
                    config['model']['n_classes'] = 8
                    logger.info("使用Cityscapes实例分割的8个类别")
            else:
                raise ValueError(f"Cityscapes数据集暂不支持 {task_type} 任务类型")
        elif dataset.lower() == 'ade20k':
            if task_type.lower() == 'semantic':
                from MYSEGX.data.Semantic_Segmentation.ade20k import dataloader as dataloader_fn
                collate_fn = ade20k_semantic_collate_fn
            else:
                raise ValueError(f"ADE20K数据集暂不支持 {task_type} 任务类型")
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
            
        logger.info(f"使用 {dataset} 数据集的 dataloader")
    except ImportError as e:
        logger.error(f"导入数据集模块失败: {str(e)}")
        raise

    # 创建数据加载器
    logger.info("创建数据加载器...")
    logger.info("数据加载器配置:")
    logger.info(f"- 训练集transform配置: {train_transform_config}")
    logger.info(f"- 验证集transform配置: {val_transform_config}")
    logger.info(f"- 模型类型: {config['task']['model_type']}")
    logger.info(f"- 任务类型: {config['task']['type']}")

    try:
        train_dataloader = dataloader_fn(
            root=config['dataset']['root'],
            split='train',
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            transform_config=train_transform_config,
            model_type=config['task']['model_type'],
            task_type=config['task']['type']
        )
        logger.info(f"训练数据集大小: {len(train_dataloader.dataset)}")
    except Exception as e:
        logger.error(f"创建训练数据加载器失败: {str(e)}")
        raise

    try:
        val_dataloader = dataloader_fn(
            root=config['dataset']['root'],
            split='val',
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            transform_config=val_transform_config,
            model_type=config['task']['model_type'],
            task_type=config['task']['type'],
            return_dataset=True if config.get('save', {}).get('metric') == 'mAP' else False
        )

        # 如果是返回的数据集，创建验证集的dataloader
        if isinstance(val_dataloader, Dataset):
            val_dataset = val_dataloader
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['train']['batch_size'],
                shuffle=False,
                num_workers=config['train']['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn
            )
            logger.info(f"验证数据集大小: {len(val_dataset)}")
        else:
            logger.info(f"验证数据集大小: {len(val_dataloader.dataset)}")
    except Exception as e:
        logger.error(f"创建验证数据加载器失败: {str(e)}")
        raise
    
    # 创建模型
    logger.info(f"构建 {config['model']['name']} 模型...")
    model = build_model(
        model_type=config['model']['name'],
        config={
            'model': config['model'],
            'task': config['task']  # 确保传递任务配置
        }
    ).to(device)
    
    # 分析模型结构
    logger.info("分析模型结构...")
    try:
        model_stats = analyze_model(
            model, 
            input_size=(config['train']['batch_size'], 3, *config['dataset']['size']),
            show_details=True
        )
        logger.info(f"模型分析完成:")
        logger.info(f"- 总参数量: {model_stats['total_params']:,}")
        logger.info(f"- 可训练参数: {model_stats['trainable_params']:,}")
        logger.info(f"- MACs: {model_stats['macs']:,}")
        logger.info(f"- GFLOPs: {model_stats['gflops']:.2f}")
    except Exception as e:
        logger.warning(f"模型分析失败: {str(e)}")
        logger.warning("继续训练，但无法显示模型统计信息")
    
    # 创建优化器和损失函数
    optimizer = _create_optimizer(model, config)
    criterion = _create_criterion(config['model']['name'], config)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        model_type=config['model']['name'],
        task_type=task_type,
        num_classes=config['model']['num_classes'],  # 使用配置中的类别数
        save_dir=metrics_base_dir,
        names=names
    )
    logger.info(f"创建训练器 - 类别数: {config['model']['num_classes']}")
    
    # 如果提供了恢复路径，加载检查点
    if resume_path:
        logger.info(f"从检查点恢复: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"恢复到 epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_metrics': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # 设置TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_dir = result_saver.get_tensorboard_dir()
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard日志保存在: {writer.log_dir}")
    
    # 启动TensorBoard服务
    tensorboard_url = _start_tensorboard(tensorboard_dir)
    if tensorboard_url:
        logger.info(f"TensorBoard服务已启动: {tensorboard_url}")
    else:
        logger.warning(f"TensorBoard启动失败，请手动运行: tensorboard --logdir={tensorboard_dir}")
    
    # 记录模型结构图
    dummy_input = torch.randn(1, 3, *config['dataset']['size']).to(device)
    try:
        writer.add_graph(model, dummy_input, use_strict_trace=False)  # 设置strict=False以允许字典输出
        logger.info("成功添加模型结构图到TensorBoard")
    except Exception:
        # 简化错误信息，不输出详细的调试信息
        logger.info("模型结构图未添加到TensorBoard，继续训练")
    
    # 获取或设置model_type
    model_type = model_name.lower() if model_name else config.get('model', {}).get('type', 'unet')
    logger.info(f"使用模型类型: {model_type}")

    # 开始训练
    logger.info("开始训练...")
    for epoch in range(start_epoch, config['train']['epochs']):
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f'\nEpoch {epoch + 1}/{config["train"]["epochs"]}')
        try:

            # 训练一个epoch
            train_loss, train_metrics = trainer.train_epoch(train_dataloader)
            logger.info(f'Train Loss: {train_loss:.4f}')
            
            # 更新训练历史
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            
            # 记录训练指标到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            for name, value in train_metrics.items():
                writer.add_scalar(f'Metrics/train_{name}', value, epoch)
                logger.info(f'Train {name}: {value:.4f}')
                
                # 特别处理实例分割特有指标
                if task_type == 'instance' and name.startswith(('AP', 'mAP', 'AR@')):
                    logger.info(f'实例分割指标 - Train {name}: {value:.4f}')
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
                logger.error(f"GPU内存错误: {str(e)}")
                logger.info("尝试减小批次大小或清理GPU内存...")
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 如果是第一个epoch就失败，可能需要调整批次大小
                if epoch == start_epoch:
                    new_batch_size = max(1, config['train']['batch_size'] // 2)
                    logger.info(f"将批次大小从 {config['train']['batch_size']} 减小到 {new_batch_size}")
                    config['train']['batch_size'] = new_batch_size
                    # 重新创建数据加载器
                    train_dataloader = dataloader_fn(
                        root=config['dataset']['root'],
                        split='train',
                        batch_size=new_batch_size,
                        num_workers=config['train']['num_workers'],
                        transform_config=config['dataset']['transform']['train'],
                        model_type=config['task']['model_type'],
                        task_type=config['task']['type']
                    )
                    continue
            raise  # 如果不是内存相关错误，继续抛出异常
        
        # 验证
        val_loss, val_metrics = trainer.validate(val_dataloader)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        
        # 更新验证历史
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # 记录验证指标到TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f'Metrics/val_{name}', value, epoch)
            logger.info(f'Val {name}: {value:.4f}')
            
            # 特别处理实例分割特有指标
            if task_type == 'instance' and name.startswith(('AP', 'mAP', 'AR@')):
                logger.info(f'实例分割指标 - Val {name}: {value:.4f}')
        
        # 保存当前epoch的指标到CSV文件
        csv_path = result_saver.save_metrics_to_csv(history, epoch)
        logger.info(f'保存当前epoch指标到CSV: {csv_path}')
        
        # 可视化一些分割结果
        if (epoch + 1) % eval_interval == 0:
            batch = next(iter(val_dataloader))
            images = batch['image'].to(device)
            targets = batch['target']
            
            with torch.no_grad():
                outputs = model(images)
            
            # 获取预测结果
            if task_type == 'semantic':
                if isinstance(outputs, torch.Tensor):
                    pred_masks = outputs.argmax(1)
                elif 'pred_masks' in outputs:
                    pred_masks = outputs['pred_masks'].argmax(1)
                elif 'out' in outputs:  # DeepLabV3/DeepLabV3+模型
                    pred_masks = outputs['out'].argmax(1)
                else:
                    raise ValueError(f"不支持的输出格式: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
                print(f"[DEBUG] Pred masks shape: {pred_masks.shape}")
            else:  # instance segmentation
                pred_masks = outputs['pred_masks']  # 已经是 [B, N, 640, 640]
                pred_logits = outputs['pred_logits']  # [B, N, C]
                print(f"[DEBUG] Instance pred masks shape: {pred_masks.shape}")
                print(f"[DEBUG] Instance pred logits shape: {pred_logits.shape}")
            
            # 添加分割结果到TensorBoard
            for i in range(min(4, len(images))):
                # 根据模型类型处理目标掩码
                if isinstance(targets, list):  # DETR或YOLACT模型
                    target_dict = targets[i]
                    if task_type == 'semantic':
                        target_mask = target_dict['semantic_mask'].to(device)
                    else:  # instance
                        target_mask = {
                            'masks': target_dict['masks'].to(device),  # [N, H, W]
                            'labels': target_dict['labels'].to(device)  # [N]
                        }
                    print(f"[DEBUG] Sample {i} ({task_type}):")
                    print(f"- Target shape: {target_mask['masks'].shape if task_type == 'instance' else target_mask.shape}")
                else:  # UNet等模型
                    target_mask = targets[i].to(device)
                    print(f"[DEBUG] Sample {i} (UNet/SAUNet):")
                    print(f"- Target shape: {target_mask.shape}")
                
                # 根据任务类型处理预测结果
                if task_type == 'semantic':
                    pred_mask = pred_masks[i]
                elif task_type == 'instance':
                    if model_type == 'yolact':
                        # YOLACT的预测结果处理
                        cur_pred_masks = outputs[i]['masks']  # [N, H, W]
                        cur_pred_scores = outputs[i]['scores']  # [N]
                        cur_pred_classes = outputs[i]['classes']  # [N]
                        # 创建预测字典
                        pred_mask = {
                            'masks': cur_pred_masks > 0.5,  # 二值化掩码
                            'labels': cur_pred_classes,
                            'scores': cur_pred_scores
                        }
                    else:
                        # 其他实例分割模型的处理
                        cur_pred_masks = pred_masks[i]  # [N, H, W]
                        cur_pred_logits = pred_logits[i]  # [N, C]
                        pred_classes = cur_pred_logits.argmax(dim=1)  # [N]
                        pred_mask = {
                            'masks': cur_pred_masks.sigmoid() > 0.5,
                            'labels': pred_classes
                        }
                else:  # panoptic
                    pred_mask = np.stack([
                        pred_masks[i]['semantic'].argmax(0).cpu().numpy(),
                        pred_masks[i]['instance'].cpu().numpy()
                    ], axis=-1)
                
                img_grid = plot_segmentation(
                    images[i], target_mask, pred_mask,
                    task_type=task_type,
                    class_names=names
                )
                
                # 记录图像形状和类型，用于调试
                if isinstance(img_grid, torch.Tensor):
                    img_shape = img_grid.shape
                    img_type = f"torch.Tensor, dtype={img_grid.dtype}"
                    img_range = f"[{img_grid.min().item()}, {img_grid.max().item()}]"
                else:
                    img_shape = img_grid.shape
                    img_type = f"numpy.ndarray, dtype={img_grid.dtype}"
                    img_range = f"[{img_grid.min()}, {img_grid.max()}]"
                
                logger.info(f"分割结果图像 - 形状: {img_shape}, 类型: {img_type}, 数据范围: {img_range}")
                
                # 添加到TensorBoard
                try:
                    writer.add_image(f'Segmentation/sample_{i}', img_grid, epoch)
                    logger.info(f"成功添加分割结果到TensorBoard: sample_{i}")
                except Exception as e:
                    logger.error(f"添加图像到TensorBoard失败: {str(e)}")
                
                # 同时保存分割结果图像到结果目录
                try:
                    # 确保图像格式正确
                    if isinstance(img_grid, torch.Tensor):
                        # 如果是PyTorch张量，转换为NumPy数组
                        if img_grid.dim() == 3 and img_grid.shape[0] == 3:  # [C, H, W]
                            # 对于TensorBoard格式的图像，需要转换为HWC格式
                            img_to_save = img_grid.permute(1, 2, 0).cpu().numpy()
                        else:
                            img_to_save = img_grid.cpu().numpy()
                    else:
                        # 已经是NumPy数组
                        img_to_save = img_grid
                        
                    # 检查图像尺寸
                    if len(img_to_save.shape) == 3:
                        h, w = img_to_save.shape[:2] if img_to_save.shape[0] != 3 else img_to_save.shape[1:3]
                        logger.info(f"保存图像尺寸: {h}x{w}")
                        
                        # 如果图像太大，调整大小
                        if h > 2000 or w > 2000:
                            logger.warning(f"图像尺寸过大 ({h}x{w})，将被调整")
                    
                    seg_save_path = result_saver.save_segmentation_result(img_to_save, epoch, i)
                    logger.info(f'保存分割结果图像: {seg_save_path}')
                except Exception as e:
                    logger.error(f"保存分割结果图像失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                # 保存原始图像和掩码
                try:
                    # 获取当前样本的原始图像
                    orig_img = images[i].cpu()
                    
                    # 获取当前样本的预测掩码
                    if task_type == 'semantic':
                        pred_mask_img = pred_mask.cpu()
                    elif task_type == 'instance':
                        # 对于实例分割，创建一个合并的掩码图像
                        if model_type == 'yolact':
                            pred_mask_img = torch.zeros_like(pred_mask['masks'][0])
                            for j, (mask, score) in enumerate(zip(pred_mask['masks'], pred_mask['scores'])):
                                if score > config['model'].get('score_threshold', 0.05):
                                    pred_mask_img = torch.logical_or(pred_mask_img, mask)
                        else:
                            pred_mask_img = torch.zeros_like(pred_mask['masks'][0])
                            for j, mask in enumerate(pred_mask['masks']):
                                pred_mask_img = torch.logical_or(pred_mask_img, mask)
                    
                    # 保存图像和掩码
                    img_path, mask_path, overlay_path = result_saver.save_prediction(
                        orig_img, 
                        pred_mask_img, 
                        f"sample_{i}_epoch_{epoch}.png"
                    )
                    logger.info(f"保存原始图像: {img_path}")
                    logger.info(f"保存掩码: {mask_path}")
                    logger.info(f"保存叠加图像: {overlay_path}")
                except Exception as e:
                    logger.error(f"保存原始图像和掩码失败: {str(e)}")
        
        # 保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'history': history
            }
            result_saver.save_checkpoint(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
            logger.info(f'保存检查点: checkpoint_epoch_{epoch + 1}.pth')
        
        # 可视化训练过程
        #if visualize:
            #plot_training_curves(
                #history,
                #save_path=os.path.join(result_saver.exp_dir, f"training_curves_epoch_{epoch+1}.png"),
                #task_type=task_type
            #)
    
    # 训练结束后保存完整的训练历史到CSV
    try:
        final_csv_path = result_saver.save_metrics_to_csv(history)
        logger.info(f'保存完整训练历史到CSV: {final_csv_path}')
    except Exception as e:
        logger.error(f"保存训练历史到CSV失败: {str(e)}")
    
    # 保存训练历史到JSON
    try:
        result_saver.save_training_history(history)
        logger.info(f'保存训练历史到JSON: {os.path.join(result_saver.exp_dir, "history.json")}')
    except Exception as e:
        logger.error(f"保存训练历史到JSON失败: {str(e)}")
    
    # 保存最终指标
    try:
        final_metrics = {}
        if history['val_metrics'] and len(history['val_metrics']) > 0:
            final_metrics = history['val_metrics'][-1]
        result_saver.save_metrics(final_metrics)
        logger.info(f'保存最终评估指标: {os.path.join(result_saver.exp_dir, "metrics.json")}')
    except Exception as e:
        logger.error(f"保存最终评估指标失败: {str(e)}")
    
    writer.close()
    logger.info(f"训练完成! 所有结果已保存到: {result_saver.exp_dir}")
    return history

def _create_optimizer(model, config):
    """创建优化器"""
    optimizer_type = getattr(torch.optim, config['optimizer']['type'])
    param_groups = []
    
    if hasattr(model, 'backbone'):
        param_groups.append({
            'params': model.backbone.parameters(),
            'lr': config['optimizer'].get('backbone_lr', config['train']['learning_rate'] * 0.1)
        })
        param_groups.append({
            'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
            'lr': config['train']['learning_rate']
        })
    else:
        param_groups.append({
            'params': model.parameters(),
            'lr': config['train']['learning_rate']
        })
    
    return optimizer_type(param_groups, weight_decay=config['train'].get('weight_decay', 0))

def _create_criterion(model_type, config):
    """创建损失函数"""
    if model_type == 'detr':
        task_type = config.get('task', {}).get('type', 'semantic')  # 从配置中获取任务类型
        loss_weights = config.get('loss', {})  # 从配置中获取损失权重
        return DETRLoss(
            num_classes=config['model']['num_classes'],
            matcher=HungarianAssigner() if task_type == 'instance' else None,
            weight_dict={
                'ce': loss_weights.get('ce', 1.0),
                'mask': loss_weights.get('mask', 1.0),
                'dice': loss_weights.get('dice', 1.0)
            },
            task_type=task_type
        )
    elif model_type == 'mask_rcnn':
        # 从配置中获取损失权重，如果没有则使用默认值
        loss_weights = config.get('loss', {})
        return MaskRCNNLoss(
            num_classes=config['model']['num_classes'],
            weight_dict={
                'rpn_cls': loss_weights.get('rpn_cls', 1.0),
                'rpn_reg': loss_weights.get('rpn_reg', 1.0),
                'cls': loss_weights.get('cls', 1.0),
                'reg': loss_weights.get('reg', 1.0),
                'mask': loss_weights.get('mask', 1.0)
            }
        )
    elif model_type == 'yolact':
        # YOLACT的损失函数权重
        loss_weights = config.get('loss', {})
        return YOLACTLoss(
            num_classes=config['model']['num_classes'],
            weight_dict={
                'cls': loss_weights.get('cls', 1.0),
                'box': loss_weights.get('box', 1.0),
                'mask': loss_weights.get('mask', 6.125),
                'proto': loss_weights.get('proto', 1.0),
                'semantic': loss_weights.get('semantic', 1.0)
            }
        )
    elif model_type in ['unet', 'saunet', 'cnn', 'deeplabv3', 'deeplabv3plus']:
        return OHEMCrossEntropyLoss(  # 使用我们自定义的CrossEntropyLoss
            num_classes=config['model']['num_classes'],  # 使用配置中的类别数
            ignore_index=config['loss'].get('ignore_index', 255)
        )
    else:
        raise ValueError(f"未实现的模型类型损失函数: {model_type}")
