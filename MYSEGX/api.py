"""MYSEGX 高级 API 接口"""
import os
import torch
from typing import Optional, Union, Dict, Any
from pathlib import Path
from datetime import datetime

from .models import build_model
from .engine.trainer import Trainer
from .data.voc import create_voc_dataloader
from .utils.losses import DETRLoss
from .data.transforms import build_transforms
from .nn.modules.assigners.hungarian_assigner import HungarianAssigner
from .utils.general import load_config, setup_logger, print_banner
from .utils.model_analyzer import analyze_model
from .utils.plots import plot_training_curves, plot_segmentation
from .utils.results import ResultSaver

def train(
    config_path: Union[str, Path],
    resume_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """训练模型的高级接口
    
    参数:
        config_path: 配置文件路径
        resume_path: 恢复训练的检查点路径
        device: 设备名称 ('cuda' 或 'cpu')
        
    返回:
        包含训练历史的字典
    """
    # 打印欢迎横幅
    print_banner()
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    logger = setup_logger(
        'MYSEGX',
        log_file=os.path.join('logs', f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    logger.info(f"加载配置文件: {config_path}")
    
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"使用设备: {device}")
    
    # 创建结果保存器
    result_saver = ResultSaver(base_dir=os.path.join('results', config['model']['name']))
    result_saver.save_config(config)
    logger.info(f"结果将保存在: {result_saver.exp_dir}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_transform = build_transforms(train=True, size=config['dataset']['size'])
    train_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split=config['dataset']['image_set'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=train_transform,
        model_type=config['model']['name']
    )
    logger.info(f"训练数据集大小: {len(train_dataloader.dataset)}")
    
    val_transform = build_transforms(train=False, size=config['dataset']['size'])
    val_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split='val',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=val_transform,
        model_type=config['model']['name']
    )
    logger.info(f"验证数据集大小: {len(val_dataloader.dataset)}")
    
    # 创建模型
    logger.info(f"构建 {config['model']['name']} 模型...")
    model = build_model(config['model']['name'], config).to(device)
    
    # 如果提供了恢复路径，加载检查点
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logger.info(f"恢复训练自检查点: {resume_path}")
    
    # 分析模型结构
    logger.info("分析模型结构...")
    analyze_model(model, input_size=(1, 3, *config['dataset']['size']))
    
    # 创建优化器
    optimizer = _create_optimizer(model, config)
    if resume_path and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("已恢复优化器状态")
    
    # 创建损失函数
    criterion = _create_criterion(config['model']['name'], config)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        model_type=config['model']['name']
    )
    
    # 记录最佳验证损失和训练历史
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': {}, 'val_metrics': {}
    }
    
    # 开始训练
    logger.info(f"\n{'='*20} 开始训练 {'='*20}")
    for epoch in range(config['train']['epochs']):
        logger.info(f'\nEpoch {epoch + 1}/{config["train"]["epochs"]}')
        
        # 训练一个epoch
        train_loss, train_metrics = trainer.train_epoch(train_dataloader)
        logger.info(f'Train Loss: {train_loss:.4f}')
        
        # 更新训练历史
        history['train_loss'].append(train_loss)
        for name, value in train_metrics.items():
            if name not in history['train_metrics']:
                history['train_metrics'][name] = []
            history['train_metrics'][name].append(value)
            logger.info(f'Train {name}: {value:.4f}')
        
        # 验证
        val_loss, val_metrics = trainer.validate(val_dataloader)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        
        # 更新验证历史
        history['val_loss'].append(val_loss)
        for name, value in val_metrics.items():
            if name not in history['val_metrics']:
                history['val_metrics'][name] = []
            history['val_metrics'][name].append(value)
            logger.info(f'Validation {name}: {value:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(result_saver.exp_dir, f'{config["model"]["name"]}_best.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, save_path)
            logger.info(f'保存最佳模型到: {save_path}，验证损失: {val_loss:.4f}')
        
        # 绘制训练曲线
        plot_path = result_saver.save_plot(f'training_curves_epoch_{epoch+1}')
        plot_training_curves(
            losses={'Train Loss': history['train_loss'], 'Val Loss': history['val_loss']},
            metrics={
                f'Train {k}': v for k, v in history['train_metrics'].items()
            } | {
                f'Val {k}': v for k, v in history['val_metrics'].items()
            },
            save_path=plot_path
        )
        logger.info(f'保存训练曲线到: {plot_path}')
    
    logger.info(f"\n{'='*20} 训练完成 {'='*20}")
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
        default_weight_dict = {'loss_ce': 1.0, 'loss_mask': 1.0, 'loss_dice': 1.0}
        weight_dict = {
            'loss_ce': config['loss_weights'].get('ce', default_weight_dict['loss_ce']),
            'loss_mask': config['loss_weights'].get('mask', default_weight_dict['loss_mask']),
            'loss_dice': config['loss_weights'].get('dice', default_weight_dict['loss_dice'])
        }
        return DETRLoss(
            num_classes=config['model']['num_classes'],
            matcher=HungarianAssigner(),
            weight_dict=weight_dict
        )
    elif model_type in ['unet', 'cnn']:
        return torch.nn.CrossEntropyLoss(
            ignore_index=config['loss'].get('ignore_index', 255)
        )
    else:
        raise ValueError(f"未实现的模型类型损失函数: {model_type}")
