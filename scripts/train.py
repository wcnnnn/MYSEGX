"""DETR模型训练脚本"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from MYSEGX.models.detr.detr import DETR
from MYSEGX.engine.trainer import Trainer
from MYSEGX.data.voc import VOCSegmentation, create_voc_dataloader
from MYSEGX.utils.losses import DETRLoss
from MYSEGX.data.transforms import build_transforms
from MYSEGX.nn.modules.assigners.hungarian_assigner import HungarianAssigner
from MYSEGX.utils.general import print_banner, setup_logger, load_config
from MYSEGX.utils.model_analyzer import analyze_model
import warnings
warnings.filterwarnings("ignore")

def main():
    # 显示启动标志
    print_banner()
    
    # 设置日志
    logger = setup_logger('MySegX', 'logs/train.log')
    
    # 加载配置文件
    config_path = 'configs/models/detr/detr_r18.yaml'
    config = load_config(config_path)
    logger.info(f"加载配置文件: {config_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建训练数据集和数据加载器
    train_transform = build_transforms(train=True, size=config['dataset']['size'])
    train_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split=config['dataset']['image_set'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=train_transform
    )
    
    # 创建验证数据集和数据加载器
    val_transform = build_transforms(train=False, size=config['dataset']['size'])
    val_dataloader = create_voc_dataloader(
        root=config['dataset']['root'],
        split='val',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        transform=val_transform
    )
    
    # 创建模型
    model = DETR(
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        num_queries=config['model']['num_queries'],
        backbone_type=config['model']['backbone_type']
    ).to(device)
    
    # 分析模型结构
    logger.info("模型结构分析：")
    analyze_model(model, input_size=(1, 3, config['dataset']['size'][0], config['dataset']['size'][1]))
    
    # 创建优化器
    optimizer = getattr(torch.optim, config['optimizer']['type'])(
        [
            {'params': model.backbone.parameters(), 
             'lr': config['optimizer']['backbone_lr']},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
             'lr': config['train']['learning_rate']}
        ],
        weight_decay=config['train']['weight_decay']
    )
    
    # 创建损失函数
    criterion = DETRLoss(
        num_classes=config['model']['num_classes'],
        matcher=HungarianAssigner(), 
        weight_dict={
            'loss_ce': config['loss_weights']['ce'],
            'loss_mask': config['loss_weights']['mask'],
            'loss_dice': config['loss_weights']['dice']
        }
    )
    
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, 'best_model.pth')
            print(f'保存最佳模型，验证损失: {val_loss:.4f}')
        
        # 学习率衰减
        if epoch == config['train']['lr_drop']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

if __name__ == '__main__':
    main()