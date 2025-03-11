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
            'loss_dice': 1.0,
            'loss_focal': 1.0
        }
        # 使用配置中的权重，如果没有则使用默认权重
        weight_dict = config.get('loss_weights', default_weight_dict)
        return DETRLoss(
            num_classes=config['model']['num_classes'],
            matcher=HungarianAssigner(),
            weight_dict=weight_dict
        )
    # 可以添加其他模型的损失函数
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_transform = build_transforms(train=True, size=config['dataset']['size'])
    train_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split=config['dataset']['image_set'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=train_transform
    )
    
    val_transform = build_transforms(train=False, size=config['dataset']['size'])
    val_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split='val',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=val_transform
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
    trainer = Trainer(model, optimizer, criterion)
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 开始训练
    for epoch in range(config['train']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["train"]["epochs"]}')
        
        # 训练一个epoch
        train_loss, train_metrics = trainer.train_epoch(train_dataloader)
        print(f'Train Loss: {train_loss:.4f}')
        for metric_name, metric_value in train_metrics.items():
            print(f'Train {metric_name}: {metric_value:.4f}')
        
        # 验证
        val_loss, val_metrics = trainer.validate(val_dataloader)
        print(f'Validation Loss: {val_loss:.4f}')
        for metric_name, metric_value in val_metrics.items():
            print(f'Validation {metric_name}: {metric_value:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'weights/{config["model"]["name"]}_best.pth'
            os.makedirs('weights', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, save_path)
            print(f'保存最佳模型到 {save_path}，验证损失: {val_loss:.4f}')
        
        # 学习率衰减
        if epoch == config['train'].get('lr_drop', float('inf')):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

if __name__ == '__main__':
    main()