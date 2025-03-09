"""统一的模型训练脚本"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from MYSEGX.models import build_model
from MYSEGX.engine.trainer import Trainer
from MYSEGX.data.voc import create_voc_dataloader
from MYSEGX.utils.losses import DETRLoss
from MYSEGX.data.transforms import build_transforms
from MYSEGX.nn.modules.assigners.hungarian_assigner import HungarianAssigner
from MYSEGX.utils.general import print_banner, setup_logger, load_config
from MYSEGX.utils.model_analyzer import analyze_model
from MYSEGX.utils.plots import plot_training_curves, plot_segmentation
from MYSEGX.utils.results import ResultSaver
import warnings
warnings.filterwarnings("ignore")

def create_optimizer(model, config):
    """创建优化器
    
    根据配置创建适当的优化器，支持不同的参数组
    """
    # 获取优化器类型
    optimizer_type = getattr(torch.optim, config['optimizer']['type'])
    
    # 准备参数组
    param_groups = []
    
    # 如果模型有backbone，使用不同的学习率
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
        # 如果没有backbone，使用统一的学习率
        param_groups.append({
            'params': model.parameters(),
            'lr': config['train']['learning_rate']
        })
    
    # 创建优化器
    return optimizer_type(param_groups, weight_decay=config['train'].get('weight_decay', 0))

def create_criterion(model_type, config):
    """创建损失函数
    
    根据模型类型和配置创建适当的损失函数
    """
    if model_type == 'detr':
        # 设置默认的损失权重
        default_weight_dict = {
            'loss_ce': 1.0,
            'loss_mask': 1.0,
            'loss_dice': 1.0
        }
        # 使用配置中的权重，如果没有则使用默认权重
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
        # UNet和CNN都使用交叉熵损失函数
        return torch.nn.CrossEntropyLoss(
            ignore_index=config['loss'].get('ignore_index', 255)  # 忽略的标签值，默认255
        )
    else:
        raise ValueError(f"未实现的模型类型损失函数: {model_type}")

def main():
    # 显示启动标志
    print_banner()
    
    # 设置日志
    logger = setup_logger('MySegX', 'logs/train.log')
    
    # 加载配置文件
    import argparse
    parser = argparse.ArgumentParser(description='统一训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger.info(f"加载配置文件: {args.config}")
    
    # 创建结果保存器
    result_saver = ResultSaver(base_dir=os.path.join('results', config['model']['name']))
    # 保存配置文件
    result_saver.save_config(config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_transform = build_transforms(train=True, size=config['dataset']['size'])
    train_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split=config['dataset']['image_set'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=train_transform,
        model_type=config['model']['name']
    )
    
    val_transform = build_transforms(train=False, size=config['dataset']['size'])
    val_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split='val',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=val_transform,
        model_type=config['model']['name']
    )
    
    # 创建模型
    model = build_model(config['model']['name'], config).to(device)
    
    # 分析模型结构
    logger.info("模型结构分析：")
    analyze_model(model, input_size=(1, 3, config['dataset']['size'][0], config['dataset']['size'][1]))
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    
    # 创建损失函数
    criterion = create_criterion(config['model']['name'], config)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        model_type=config['model']['name']
    )
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {},
        'val_metrics': {}
    }
    
    # 开始训练
    for epoch in range(config['train']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["train"]["epochs"]}')
        
        # 训练一个epoch
        train_loss, train_metrics = trainer.train_epoch(train_dataloader)
        print(f'Train Loss: {train_loss:.4f}')
        for metric_name, metric_value in train_metrics.items():
            print(f'Train {metric_name}: {metric_value:.4f}')
            
        # 更新训练历史
        history['train_loss'].append(train_loss)
        for metric_name, metric_value in train_metrics.items():
            if metric_name not in history['train_metrics']:
                history['train_metrics'][metric_name] = []
            history['train_metrics'][metric_name].append(metric_value)
        
        # 验证
        val_loss, val_metrics = trainer.validate(val_dataloader)
        print(f'Validation Loss: {val_loss:.4f}')
        for metric_name, metric_value in val_metrics.items():
            print(f'Validation {metric_name}: {metric_value:.4f}')
            
        # 更新验证历史
        history['val_loss'].append(val_loss)
        for metric_name, metric_value in val_metrics.items():
            if metric_name not in history['val_metrics']:
                history['val_metrics'][metric_name] = []
            history['val_metrics'][metric_name].append(metric_value)
        
        # 绘制训练曲线
        plot_training_curves(
            losses={
                'Train Loss': history['train_loss'],
                'Val Loss': history['val_loss']
            },
            metrics={
                f'Train {k}': v for k, v in history['train_metrics'].items()
            } | {
                f'Val {k}': v for k, v in history['val_metrics'].items()
            },
            save_path=result_saver.save_plot(f'training_curves_epoch_{epoch+1}')
        )
        
        # 保存训练历史
        result_saver.save_training_history(history)
        
        # 可视化一些验证结果
        if hasattr(trainer, 'last_val_batch'):
            images, masks, predictions = trainer.last_val_batch
            for i in range(min(3, len(images))):  # 只保存前3张图片
                plot_segmentation(
                    images[i].cpu().numpy(),
                    predictions[i].cpu().numpy(),
                    save_path=result_saver.save_plot(f'val_result_epoch_{epoch+1}_sample_{i+1}')
                )
                # 同时保存原始图像和预测结果
                result_saver.save_prediction(
                    images[i].cpu().numpy(),
                    predictions[i].cpu().numpy(),
                    f'epoch_{epoch+1}_sample_{i+1}.png'
                )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(result_saver.exp_dir, f'{config["model"]["name"]}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'history': history
            }, save_path)
            print(f'保存最佳模型到 {save_path}，验证损失: {val_loss:.4f}')
            
            # 保存最终评估指标
            result_saver.save_metrics({
                'best_epoch': epoch,
                'best_val_loss': float(best_val_loss),
                **{f'best_val_{k}': float(v[-1]) for k, v in history['val_metrics'].items()},
                **{f'final_train_{k}': float(v[-1]) for k, v in history['train_metrics'].items()}
            })
        
        # 学习率衰减
        if epoch == config['train'].get('lr_drop', float('inf')):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

if __name__ == '__main__':
    main()