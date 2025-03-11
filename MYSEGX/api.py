"""MYSEGX 高级 API 接口"""
import os
import torch
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from datetime import datetime
import importlib.util
from .models import build_model
from .engine.trainer import Trainer
from .utils.losses import DETRLoss
from .data.transforms import build_transforms
from .nn.modules.assigners.hungarian_assigner import HungarianAssigner
from .utils.general import load_config, setup_logger, print_banner
from .utils.model_analyzer import analyze_model
from .utils.plots import plot_training_curves, plot_segmentation
from .utils.results import ResultSaver
import sys

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
    """训练模型的高级接口
    
    参数:
        config_path: 配置文件路径
        resume_path: 恢复训练的检查点路径
        task_type: 任务类型，支持 'semantic' 或 'instance'
        device: 设备名称 ('cuda' 或 'cpu')
        distributed: 是否使用分布式训练
        amp: 是否使用混合精度训练
        save_dir: 保存模型和结果的目录
        log_dir: 日志保存目录
        eval_interval: 每隔多少个epoch进行一次评估
        save_interval: 每隔多少个epoch保存一次模型
        visualize: 是否可视化训练过程
        seed: 随机种子，用于复现实验结果
        debug: 是否启用调试模式
        metrics_dir: 指标保存目录
        class_names: 类别名称列表
    """
    # 打印欢迎横幅
    print_banner()
    
    # 处理配置
    if config_path is not None:
        config = load_config(config_path)
    else:
        # 确定任务目录
        task_dir = 'Semantic_Segmentation' if task_type == 'semantic' else 'Instance_Segmentation'
        
        # 查找模型配置模板
        model_config_dir = os.path.join('configs', task_dir, model_name or 'detr')
        if os.path.exists(model_config_dir):
            config_files = [f for f in os.listdir(model_config_dir) if f.endswith('.yaml')]
            if config_files:
                config = load_config(os.path.join(model_config_dir, config_files[0]))
                
                # 更新模型参数
                if 'model' not in config:
                    config['model'] = {}
                config['model']['name'] = model_name or config['model']['name']
                
                # 更新特定模型参数
                if config['model']['name'] == 'detr':
                    if num_classes is not None: config['model']['num_classes'] = num_classes
                    if hidden_dim is not None: config['model']['hidden_dim'] = hidden_dim
                    if nhead is not None: config['model']['nhead'] = nhead
                    if num_encoder_layers is not None: config['model']['num_encoder_layers'] = num_encoder_layers
                    if num_decoder_layers is not None: config['model']['num_decoder_layers'] = num_decoder_layers
                    if dim_feedforward is not None: config['model']['dim_feedforward'] = dim_feedforward
                    if dropout is not None: config['model']['dropout'] = dropout
                    if num_queries is not None: config['model']['num_queries'] = num_queries
                    if backbone_type is not None: config['model']['backbone_type'] = backbone_type
                elif config['model']['name'] == 'unet':
                    if num_classes is not None: config['model']['n_classes'] = num_classes
                
                # 更新数据集参数
                if dataset_root is not None: config['dataset']['root'] = dataset_root
                if image_size is not None: config['dataset']['size'] = image_size
                
                # 更新训练参数
                if batch_size is not None: config['train']['batch_size'] = batch_size
                if num_workers is not None: config['train']['num_workers'] = num_workers
                if learning_rate is not None: config['train']['learning_rate'] = learning_rate
                if weight_decay is not None: config['train']['weight_decay'] = weight_decay
                if epochs is not None: config['train']['epochs'] = epochs
                if lr_drop is not None: config['train']['lr_drop'] = lr_drop
                
                # 更新优化器参数
                if learning_rate is not None and 'optimizer' in config:
                    config['optimizer']['backbone_lr'] = learning_rate * 0.1
            else:
                raise ValueError(f"在 {model_config_dir} 中未找到配置文件")
        else:
            raise ValueError(f"未找到任务 {task_type} 的模型 {model_name or 'detr'} 配置目录")
    
    # 设置日志
    logger = setup_logger(
        'MYSEGX',
        log_file=os.path.join('logs', f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    logger.info(f"加载配置文件: {config_path}")
    
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
    
    # 加载dataloader函数
    try:
        # 从包中导入模块
        if task_type.lower() == 'semantic':
            from MYSEGX.data.Semantic_Segmentation.voc import dataloader as dataloader_fn
        elif task_type.lower() == 'instance':
            from MYSEGX.data.Instance_Segmentation.voc import dataloader as dataloader_fn
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    except ImportError as e:
        logger.error(f"导入数据集模块失败: {str(e)}")
        raise
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_transform = build_transforms(train=True, size=config['dataset']['size'])
    train_dataloader = dataloader_fn(
        root=config['dataset']['root'],
        split='train',  # 使用默认的训练集划分
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=train_transform,
        model_type=config['model']['name'],
        task_type=task_type  # 传递任务类型到数据加载器
    )
    logger.info(f"训练数据集大小: {len(train_dataloader.dataset)}")
    
    val_transform = build_transforms(train=False, size=config['dataset']['size'])
    val_dataloader = dataloader_fn(
        root=config['dataset']['root'],
        split='val',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=val_transform,
        model_type=config['model']['name'],
        task_type=task_type  # 传递任务类型到数据加载器
    )
    logger.info(f"验证数据集大小: {len(val_dataloader.dataset)}")
    
    # 创建模型
    logger.info(f"构建 {config['model']['name']} 模型...")
    model = build_model(model_type=config['model']['name'], config=config).to(device)
    
    # 分析模型结构
    logger.info("分析模型结构...")
    analyze_model(model)
    
    # 创建优化器和损失函数
    optimizer = _create_optimizer(model, config)
    criterion = _create_criterion(config['model']['name'], config)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        model_type=config['model']['name'],
        task_type=task_type,  # 传递任务类型到训练器
        num_classes=config['model'].get('num_classes', config['model'].get('n_classes', 21)),
        save_dir=metrics_base_dir,  # 使用指标保存目录
        names=names  # 传递类别名称
    )
    
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
    
    # 开始训练
    logger.info("开始训练...")
    for epoch in range(start_epoch, config['train']['epochs']):
        logger.info(f'\nEpoch {epoch + 1}/{config["train"]["epochs"]}')
        
        # 训练一个epoch
        train_loss, train_metrics = trainer.train_epoch(train_dataloader)
        logger.info(f'Train Loss: {train_loss:.4f}')
        
        # 更新训练历史
        history['train_loss'].append(train_loss)
        history['train_metrics'].append(train_metrics)
        
        # 打印训练指标
        for name, value in train_metrics.items():
            logger.info(f'Train {name}: {value:.4f}')
        
        # 验证
        val_loss, val_metrics = trainer.validate(val_dataloader)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        
        # 更新验证历史
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # 打印验证指标
        for name, value in val_metrics.items():
            logger.info(f'Val {name}: {value:.4f}')
        
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
        if visualize:
            plot_training_curves(history, result_saver.exp_dir)
    
    logger.info("训练完成!")
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
        return DETRLoss(
            num_classes=config['model']['num_classes'],
            matcher=HungarianAssigner() if config['task']['type'] == 'instance' else None,
            weight_dict={'ce': 1.0, 'mask': 1.0, 'dice': 1.0},
            task_type=config['task']['type']
        )
    elif model_type in ['unet','saunet','cnn']:
        return torch.nn.CrossEntropyLoss(
            ignore_index=config['loss'].get('ignore_index', 255)
        )
    else:
        raise ValueError(f"未实现的模型类型损失函数: {model_type}")
