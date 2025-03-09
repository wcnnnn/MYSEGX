"""MYSEGX 训练器模块"""

import torch
import logging
from tqdm import tqdm
from ..utils.metrics import MetricCalculator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    """模型训练器类"""
    
    def __init__(self, model, optimizer, criterion, model_type='detr'):
        """初始化训练器
        
        参数:
            model: 待训练的模型
            optimizer: 优化器
            criterion: 损失函数
            model_type: 模型类型，支持 'detr'、'unet' 和 'cnn'
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        self.device = next(model.parameters()).device
        self.metric_calculator = MetricCalculator(model_type=model_type)
        self.logger = logging.getLogger(__name__)
        self.last_val_batch = None  # 用于存储最后一个验证批次的结果

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
                
            # 记录批次结构以便调试
            # self.logger.info(f"Batch keys: {batch.keys()}")
            # self.logger.info(f"Target type: {type(batch['target'])}")
            # if isinstance(batch['target'], list):
            #     self.logger.info(f"Target length: {len(batch['target'])}")
            #     if len(batch['target']) > 0:
            #         self.logger.info(f"First target keys: {batch['target'][0].keys()}")
            
            # 将图像移动到设备并记录形状
            images = batch['image'].to(self.device)
            
            if self.model_type in ['unet', 'cnn']:  
                # UNet和CNN模型的目标处理
                if not isinstance(batch['target'], torch.Tensor):
                    raise ValueError(f"Expected batch['target'] to be a tensor for {self.model_type}, got {type(batch['target'])}")
                targets = batch['target'].to(self.device)
                return images, targets
            
            # DETR模型的目标处理
            if not isinstance(batch['target'], list):
                raise ValueError(f"Expected batch['target'] to be a list for DETR, got {type(batch['target'])}")
                
            targets = []
            for i, target in enumerate(batch['target']):
                if not isinstance(target, dict):
                    raise ValueError(f"Expected target to be a dict, got {type(target)}")
                    
                # 将字典中的张量移动到设备
                processed_target = {}
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        processed_target[k] = v.to(self.device)
                        # 记录每个张量的形状
                        # self.logger.info(f"Target {i}, {k} shape: {v.shape}")
                        
                        # 确保掩码是3D张量 (N, H, W)
                        if k == 'masks':
                            if len(v.shape) != 3:
                                raise ValueError(f"Expected masks to be 3D (N, H, W), got shape {v.shape}")
                            
                            # 确保掩码是二值的
                            if not torch.all(torch.logical_or(v == 0, v == 1)):
                                # self.logger.warning(f"Masks should be binary (0 or 1), got values: {torch.unique(v)}")
                                v = (v > 0.5).float()
                                processed_target[k] = v
                                
                        # 确保标签是1D张量
                        elif k == 'labels':
                            if len(v.shape) != 1:
                                raise ValueError(f"Expected labels to be 1D, got shape {v.shape}")
                            
                            # 确保标签在正确的范围内 (0-19)
                            if torch.any(v < 0) or torch.any(v >= 20):
                                raise ValueError(f"Labels must be in range [0, 19], got values: {torch.unique(v)}")
                                
                        # 确保边界框是2D张量 (N, 4)
                        elif k == 'boxes':
                            if len(v.shape) != 2 or v.shape[-1] != 4:
                                raise ValueError(f"Expected boxes to be 2D (N, 4), got shape {v.shape}")
                    else:
                        processed_target[k] = v
                targets.append(processed_target)
            
            return images, targets
            
        except Exception as e:
            # self.logger.error(f"Error in _prepare_batch: {str(e)}")
            # self.logger.error(f"Batch structure: {batch}")
            raise
        
    def train_epoch(self, dataloader):
        """训练一个epoch
        
        参数:
            dataloader: 训练数据加载器
        """
        self.model.train()
        total_loss = 0
        
        try:
            pbar = tqdm(dataloader, desc='Training')
            for batch_idx, batch in enumerate(pbar):
                try:
                    # 准备数据
                    # self.logger.info(f"Processing batch {batch_idx}")
                    images, targets = self._prepare_batch(batch)
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    
                    # 计算损失
                    if self.model_type == 'detr':
                        loss_dict = self.criterion(outputs, targets)
                        loss = sum(loss_dict.values())
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    # 更新进度条
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # 计算指标
                    self.metric_calculator.update(outputs, targets)
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error in train_epoch: {str(e)}")
            raise
            
        # 计算平均损失和指标
        avg_loss = total_loss / len(dataloader)
        metrics = self.metric_calculator.compute()
        self.metric_calculator.reset()
        
        return avg_loss, metrics
    
    def validate(self, dataloader):
        """验证模型
        
        参数:
            dataloader: 验证数据加载器
        """
        self.model.eval()
        total_loss = 0
        
        try:
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
                            pred_masks = outputs['pred_masks'].sigmoid() > 0.5
                        else:
                            loss = self.criterion(outputs, targets)
                            # 获取预测掩码
                            pred_masks = torch.argmax(outputs, dim=1)
                        
                        # 更新进度条
                        total_loss += loss.item()
                        pbar.set_postfix({'loss': loss.item()})
                        
                        # 计算指标
                        self.metric_calculator.update(outputs, targets)
                        
                        # 保存最后一个批次的结果用于可视化
                        if batch_idx == len(dataloader) - 1:
                            self.last_val_batch = (
                                images.detach(),  # 原始图像
                                targets if isinstance(targets, torch.Tensor) else targets[0]['masks'],  # 真实掩码
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
        metrics = self.metric_calculator.compute()
        self.metric_calculator.reset()
        
        return avg_loss, metrics