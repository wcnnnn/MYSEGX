"""MYSEGX 训练器模块"""

import torch
import logging
from tqdm import tqdm
from pathlib import Path
from ..utils.metrics import SegmentationMetrics
import numpy as np
import time

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_debug.log')
    ]
)

class Trainer:
    """模型训练器类"""
    
    def __init__(self, model, optimizer, criterion, model_type='detr', task_type='semantic', num_classes=21, save_dir='./results', names=None):
        """初始化训练器
        
        参数:
            model: 待训练的模型
            optimizer: 优化器
            criterion: 损失函数
            model_type: 模型类型，支持 'detr'、'unet'、'saunet' 和 'cnn'
            task_type: 任务类型，支持 'semantic'、'instance' 和 'panoptic'
            num_classes: 类别数量
            save_dir: 结果保存路径
            names: 类别名称列表，如果为None则自动生成
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        self.task_type = task_type
        self.device = next(model.parameters()).device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有提供类别名称，则自动生成
        if names is None:
            names = [f'class_{i}' for i in range(num_classes)]
        
        # 初始化训练和验证指标计算器
        self.train_metrics = SegmentationMetrics(
            num_classes=num_classes,
            save_dir=self.save_dir / 'train',
            names=names
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=num_classes,
            save_dir=self.save_dir / 'val',
            names=names
        )
        
        self.logger = logging.getLogger(__name__)
        self.last_val_batch = None  # 用于存储最后一个验证批次的结果
        self.rng = np.random.RandomState()  # 随机数生成器
        
    def _prepare_batch(self, batch):
        """准备批次数据
        
        将批次数据移动到正确的设备上，并处理目标字典
        
        参数:
            batch: 包含 'image' 和 'target' 的字典
                  对于DETR: target是字典列表，每个字典包含 'masks', 'labels', 'boxes' 和 'image_id'
                  对于UNet/CNN: target是分割掩码张量
        """
        try:
            if not isinstance(batch, dict):
                raise ValueError(f"Expected batch to be a dict, got {type(batch)}")
                
            if 'image' not in batch or 'target' not in batch:
                raise ValueError("Batch must contain 'image' and 'target' keys")
            
            # 将图像移动到设备
            images = batch['image'].to(self.device)
            
            if self.model_type in ['unet', 'saunet', 'cnn']:  
                # UNet和CNN模型的目标处理
                if not isinstance(batch['target'], torch.Tensor):
                    raise ValueError(f"Expected batch['target'] to be a tensor for {self.model_type}, got {type(batch['target'])}")
                targets = batch['target'].to(self.device)
                return images, targets
            
            # DETR模型的目标处理
            if not isinstance(batch['target'], list):
                raise ValueError(f"Expected batch['target'] to be a list for DETR, got {type(batch['target'])}")
                
            targets = []
            for target in batch['target']:
                if not isinstance(target, dict):
                    raise ValueError(f"Expected target to be a dict, got {type(target)}")
                    
                # 将字典中的张量移动到设备
                processed_target = {}
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        processed_target[k] = v.to(self.device)
                        
                        # 确保掩码是二值的
                        if k == 'masks' and self.task_type == 'instance':
                            if len(v.shape) != 3:
                                raise ValueError(f"Expected masks to be 3D (N, H, W), got shape {v.shape}")
                            v = (v > 0.5).float()
                            processed_target[k] = v
                    else:
                        processed_target[k] = v
                targets.append(processed_target)
            
            return images, targets
            
        except Exception as e:
            self.logger.error(f"Error in _prepare_batch: {str(e)}")
            raise
        
    def train_epoch(self, dataloader):
        """训练一个epoch
        
        参数:
            dataloader: 训练数据加载器
            
        返回:
            avg_loss: 平均损失值
            metrics: 评价指标字典
        """
        self.model.train()
        total_loss = 0
        
        try:
            self.logger.info(f"Starting training epoch with {len(dataloader)} batches")
            pbar = tqdm(dataloader, desc='Training')
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Debug batch structure
                    self.logger.debug(f"Batch {batch_idx} keys: {batch.keys()}")
                    self.logger.debug(f"Image shape: {batch['image'].shape}")
                    if isinstance(batch['target'], torch.Tensor):
                        self.logger.debug(f"Target shape: {batch['target'].shape}")
                    elif isinstance(batch['target'], list):
                        self.logger.debug(f"Target list length: {len(batch['target'])}")
                    
                    # 准备数据
                    images, targets = self._prepare_batch(batch)
                    self.logger.debug(f"Prepared images shape: {images.shape}, device: {images.device}")
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    self.logger.debug("Starting forward pass")
                    outputs = self.model(images)
                    self.logger.debug("Completed forward pass")
                    
                    # 计算损失
                    if self.model_type == 'detr':
                        loss_dict = self.criterion(outputs, targets)
                        loss = sum(loss_dict.values())
                        self.logger.debug(f"DETR loss components: {loss_dict}")
                        # 获取预测掩码并确保是浮点类型而不是布尔类型
                        if self.task_type == 'instance':
                            pred_masks = (outputs['pred_masks'].sigmoid() > 0.5).float()
                            target_masks = torch.cat([t['masks'] for t in targets], dim=0)
                        elif self.task_type == 'semantic':
                            pred_masks = torch.argmax(outputs['pred_masks'], dim=1)
                            # 不需要连接，直接堆叠语义分割掩码
                            target_masks = torch.stack([t['semantic_mask'] for t in targets])
                        else:  # panoptic
                            pred_masks = outputs['pred_panoptic']
                            target_masks = torch.cat([t['panoptic_mask'] for t in targets], dim=0)
                    else:
                        loss = self.criterion(outputs, targets)
                        self.logger.debug(f"Loss value: {loss.item()}")
                        # 获取预测掩码
                        pred_masks = torch.argmax(outputs, dim=1)
                        target_masks = targets
                    
                    # 反向传播和优化
                    self.logger.debug("Starting backward pass")
                    loss.backward()
                    self.logger.debug("Completed backward pass")
                    self.optimizer.step()
                    self.logger.debug("Completed optimizer step")
                    
                    # 更新进度条
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # 计算指标
                    self.train_metrics.update(pred_masks, target_masks)
                    
                    if batch_idx == 0:  # Print detailed info for first batch
                        self.logger.info(f"First batch completed successfully:")
                        self.logger.info(f"- Loss: {loss.item()}")
                        self.logger.info(f"- Pred masks shape: {pred_masks.shape}")
                        self.logger.info(f"- Target masks shape: {target_masks.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    self.logger.error(f"Batch structure: {type(batch)}")
                    if isinstance(batch, dict):
                        for k, v in batch.items():
                            self.logger.error(f"Key '{k}': type={type(v)}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error in train_epoch: {str(e)}")
            raise
            
        # 计算平均损失和指标
        avg_loss = total_loss / len(dataloader)
        metrics = self.train_metrics.compute()
        
        # 绘制训练指标图表
        #self.train_metrics.plot()
        self.train_metrics.reset()
        
        return avg_loss, metrics
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        try:
            # 使用当前时间作为种子以确保每次都不同
            self.rng.seed(int(time.time()))
            # 随机选择一个批次用于可视化
            vis_batch_idx = self.rng.randint(0, len(dataloader))
            
            with torch.no_grad():
                pbar = tqdm(dataloader, desc='Validating')
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # 准备数据
                        images, targets = self._prepare_batch(batch)
                        
                        # 前向传播
                        outputs = self.model(images)
                        
                        # 计算损失
                        if self.model_type == 'detr':
                            loss_dict = self.criterion(outputs, targets)
                            loss = sum(loss_dict.values())
                            # 获取预测掩码
                            if self.task_type == 'instance':
                                pred_masks = (outputs['pred_masks'].sigmoid() > 0.5).float()
                                target_masks = torch.cat([t['masks'] for t in targets], dim=0)
                            elif self.task_type == 'semantic':
                                pred_masks = torch.argmax(outputs['pred_masks'], dim=1)
                                # 不需要连接，直接堆叠语义分割掩码
                                target_masks = torch.stack([t['semantic_mask'] for t in targets])
                            else:  # panoptic
                                pred_masks = outputs['pred_panoptic']
                                target_masks = torch.cat([t['panoptic_mask'] for t in targets], dim=0)
                        else:
                            loss = self.criterion(outputs, targets)
                            # 获取预测掩码
                            pred_masks = torch.argmax(outputs, dim=1)
                            target_masks = targets
                        
                        # 更新进度条
                        total_loss += loss.item()
                        pbar.set_postfix({'loss': loss.item()})
                        
                        # 计算指标
                        self.val_metrics.update(pred_masks, target_masks)
                        
                        # 保存随机选择的批次结果用于可视化
                        if batch_idx == vis_batch_idx:
                            self.last_val_batch = (
                                images.detach(),  # 原始图像
                                target_masks.detach(),  # 真实掩码
                                pred_masks.detach()  # 预测掩码
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error in validate: {str(e)}")
            raise
            
        # 计算平均损失和指标
        avg_loss = total_loss / len(dataloader)
        metrics = self.val_metrics.compute()
        
        # 绘制验证指标图表
        #self.val_metrics.plot()
        self.val_metrics.reset()
        
        return avg_loss, metrics