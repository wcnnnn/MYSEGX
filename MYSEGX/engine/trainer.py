"""MYSEGX 训练器模块"""

import torch
import logging
from tqdm import tqdm
from pathlib import Path
from ..utils.metrics import SegmentationMetrics
import numpy as np
import time
import torch.nn.functional as F
from typing import Dict, List, Optional

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
            model_type: 模型类型，支持 'detr'、'unet'、'saunet'、'cnn'、'deeplabv3'、'deeplabv3plus'、'mask_rcnn'、'yolact'
            task_type: 任务类型，支持 'semantic'、'instance' 和 'panoptic'
            num_classes: 类别数量
            save_dir: 结果保存路径
            names: 类别名称列表，如果为None则自动生成
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type.lower()
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
            names=names,
            task_type=task_type
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=num_classes,
            save_dir=self.save_dir / 'val',
            names=names,
            task_type=task_type
        )
        
        # 打印任务类型信息
        logging.info(f"初始化{task_type}分割训练器")
        if task_type == 'instance':
            logging.info(f"实例分割任务 - 模型类型: {model_type}, 类别数: {num_classes}")
        
        self.logger = logging.getLogger(__name__)
        self.last_val_batch = None  # 用于存储最后一个验证批次的结果
        self.rng = np.random.RandomState()  # 随机数生成器
        
    def _prepare_batch(self, batch):
        """准备批次数据
        
        将批次数据移动到正确的设备上，并处理目标字典
        
        参数:
            batch: 包含 'image' 和 'target' 的字典
                  对于DETR/Mask R-CNN: target是字典列表，每个字典包含 'masks', 'labels', 'boxes' 等
                  对于UNet/CNN/DeepLabV3: target是分割掩码张量
        """
        try:
            if not isinstance(batch, dict):
                raise ValueError(f"Expected batch to be a dict, got {type(batch)}")
                
            if 'image' not in batch or 'target' not in batch:
                raise ValueError("Batch must contain 'image' and 'target' keys")
            
            # 将图像移动到设备并确保维度正确
            images = batch['image']
            self.logger.debug(f"批次图像: shape={images.shape}, dtype={images.dtype}")
            self.logger.debug(f"图像值范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
            
            # 检查图像是否已经正确归一化（应该在dataloader中完成）
            if images.max() > 10.0 or images.min() < -10.0:
                self.logger.warning("图像值范围异常，请检查dataloader中的归一化步骤")
            
            # 处理图像维度
            if images.dim() == 5:  # [B, N, C, H, W] -> [B*N, C, H, W]
                B, N, C, H, W = images.shape
                if B == 1:  # 如果批次大小为1，直接取第一个批次
                    images = images.squeeze(0)  # [N, C, H, W]
                else:
                    images = images.view(B * N, C, H, W)
            elif images.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                images = images.unsqueeze(0)
            
            # 移动到设备
            images = images.to(self.device)
            self.logger.debug(f"处理后图像: shape={images.shape}, device={images.device}")
            
            if self.model_type in ['unet', 'cnn', 'deeplabv3', 'deeplabv3plus']:  
                # UNet、CNN和DeepLabV3模型的目标处理
                if not isinstance(batch['target'], torch.Tensor):
                    raise ValueError(f"Expected batch['target'] to be a tensor for {self.model_type}, got {type(batch['target'])}")
                targets = batch['target'].to(self.device)
                self.logger.debug(f"目标掩码: shape={targets.shape}, unique values={torch.unique(targets).tolist()}")
                return images, targets
            
            # DETR和Mask R-CNN模型的目标处理
            if not isinstance(batch['target'], list):
                raise ValueError(f"Expected batch['target'] to be a list for {self.model_type}, got {type(batch['target'])}")
                
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
            
            # 打印调试信息
            self.logger.debug(f"处理后目标数量: {len(targets)}")
            if targets:
                self.logger.debug(f"第一个目标键: {list(targets[0].keys())}")
                for k, v in targets[0].items():
                    if isinstance(v, torch.Tensor):
                        self.logger.debug(f"  {k}: shape={v.shape}")
            
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
            self.logger.info(f"开始训练epoch，共{len(dataloader)}个批次")
            pbar = tqdm(dataloader, desc='Training')
            for batch_idx, batch in enumerate(pbar):
                try:
                    # 记录批次数据信息
                    self.logger.debug(f"批次 {batch_idx} 数据:")
                    self.logger.debug(f"- 图像形状: {batch['image'].shape}")
                    self.logger.debug(f"- 图像值范围: [{batch['image'].min().item():.4f}, {batch['image'].max().item():.4f}]")
                    if isinstance(batch['target'], torch.Tensor):
                        self.logger.debug(f"- 目标形状: {batch['target'].shape}")
                        self.logger.debug(f"- 目标值范围: [{batch['target'].min().item()}, {batch['target'].max().item()}]")
                    
                    # 准备数据
                    images, targets = self._prepare_batch(batch)
                    self.logger.debug(f"处理后的数据:")
                    self.logger.debug(f"- 图像形状: {images.shape}")
                    self.logger.debug(f"- 图像值范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
                    self.logger.debug(f"- 图像设备: {images.device}")
                    
                    # 前向传播前检查模型参数
                    if batch_idx == 0:
                        self.logger.info("检查模型参数:")
                        total_params = 0
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                self.logger.debug(f"- {name}: shape={param.shape}, 范围=[{param.min().item():.4f}, {param.max().item():.4f}]")
                                total_params += param.numel()
                        self.logger.info(f"模型总参数量: {total_params:,}")
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    self.logger.debug("开始前向传播")
                    if self.model_type in ['mask_rcnn', 'detr']:
                        outputs = self.model(images, targets)
                    else:
                        outputs = self.model(images)
                    self.logger.debug("完成前向传播")
                    
                    # 检查模型输出
                    if isinstance(outputs, dict):
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor):
                                self.logger.debug(f"模型输出 '{k}':")
                                self.logger.debug(f"- 形状: {v.shape}")
                                self.logger.debug(f"- 值范围: [{v.min().item():.4f}, {v.max().item():.4f}]")
                                if torch.isnan(v).any():
                                    self.logger.warning(f"- 包含NaN值!")
                                if torch.isinf(v).any():
                                    self.logger.warning(f"- 包含Inf值!")
                    
                    # 计算损失
                    if self.model_type == 'detr':
                        loss_dict = self.criterion(outputs, targets)
                        loss = sum(loss_dict.values())
                        self.logger.debug(f"DETR损失组件: {loss_dict}")
                        # 获取预测掩码
                        if self.task_type == 'instance':
                            pred_masks = (outputs['pred_masks'].sigmoid() > 0.5).float()
                            target_masks = torch.cat([t['masks'] for t in targets], dim=0)
                        elif self.task_type == 'semantic':
                            pred_masks = torch.argmax(outputs['pred_masks'], dim=1)
                            target_masks = torch.stack([t['semantic_mask'] for t in targets])
                        else:  # panoptic
                            pred_masks = outputs['pred_panoptic']
                            target_masks = torch.cat([t['panoptic_mask'] for t in targets], dim=0)
                    elif self.model_type == 'mask_rcnn':
                        loss = sum(outputs['losses'].values()) if 'losses' in outputs else outputs['loss']
                        self.logger.debug(f"Mask R-CNN损失: {loss.item()}")
                        pred_masks = outputs['pred_masks']
                        target_masks = torch.cat([t['masks'] for t in targets], dim=0)
                    elif self.model_type in ['deeplabv3', 'deeplabv3plus']:
                        if isinstance(outputs, dict) and 'out' in outputs:
                            self.logger.debug(f"DeepLabV3输出:")
                            self.logger.debug(f"- 格式: {list(outputs.keys())}")
                            self.logger.debug(f"- 'out'形状: {outputs['out'].shape}")
                            self.logger.debug(f"- 'out'范围: [{outputs['out'].min().item():.4f}, {outputs['out'].max().item():.4f}]")
                            
                            # 检查输出是否有异常值
                            if torch.isnan(outputs['out']).any():
                                self.logger.warning("输出中包含NaN值!")
                            if torch.isinf(outputs['out']).any():
                                self.logger.warning("输出中包含Inf值!")
                            
                            # 确保targets格式正确
                            if isinstance(targets, list) and 'semantic_mask' in targets[0]:
                                target_masks = torch.stack([t['semantic_mask'] for t in targets])
                                self.logger.debug(f"目标掩码:")
                                self.logger.debug(f"- 形状: {target_masks.shape}")
                                self.logger.debug(f"- 范围: [{target_masks.min().item()}, {target_masks.max().item()}]")
                                loss = self.criterion(outputs['out'], target_masks)
                            else:
                                self.logger.debug(f"目标格式: {type(targets)}")
                                if isinstance(targets, torch.Tensor):
                                    self.logger.debug(f"- 形状: {targets.shape}")
                                    self.logger.debug(f"- 范围: [{targets.min().item()}, {targets.max().item()}]")
                                    unique_targets = torch.unique(targets)
                                    self.logger.debug(f"- 唯一类别: {unique_targets}")
                                loss = self.criterion(outputs['out'], targets)
                            
                            # 获取预测掩码
                            pred_masks = torch.argmax(outputs['out'], dim=1)
                            self.logger.debug(f"预测掩码:")
                            self.logger.debug(f"- 形状: {pred_masks.shape}")
                            self.logger.debug(f"- 范围: [{pred_masks.min().item()}, {pred_masks.max().item()}]")
                            self.logger.debug(f"- 唯一值: {torch.unique(pred_masks)}")
                            
                            # 检查类别匹配情况
                            if isinstance(targets, torch.Tensor):
                                target_classes = torch.unique(targets).cpu().numpy()
                                pred_classes = torch.unique(pred_masks).cpu().numpy()
                                missing_classes = set(target_classes) - set(pred_classes)
                                extra_classes = set(pred_classes) - set(target_classes)
                                if missing_classes:
                                    self.logger.warning(f"缺失的类别: {missing_classes}")
                                if extra_classes:
                                    self.logger.warning(f"多余的类别: {extra_classes}")
                            
                            if isinstance(targets, list) and 'semantic_mask' in targets[0]:
                                target_masks = torch.stack([t['semantic_mask'] for t in targets])
                            else:
                                target_masks = targets
                        else:
                            self.logger.warning(f"DeepLabV3输出格式不符合预期: {type(outputs)}")
                            if isinstance(outputs, dict):
                                self.logger.warning(f"可用键: {list(outputs.keys())}")
                            loss = self.criterion(outputs, targets)
                            if isinstance(outputs, dict) and 'pred_masks' in outputs:
                                pred_masks = torch.argmax(outputs['pred_masks'], dim=1)
                            else:
                                pred_masks = torch.argmax(outputs, dim=1)
                            target_masks = targets
                    
                    # 检查损失值
                    if torch.isnan(loss).any():
                        self.logger.error("损失值包含NaN!")
                        raise ValueError("损失值包含NaN")
                    if torch.isinf(loss).any():
                        self.logger.error("损失值包含Inf!")
                        raise ValueError("损失值包含Inf")
                    
                    # 反向传播和优化
                    self.logger.debug("开始反向传播")
                    loss.backward()
                    
                    # 检查梯度
                    if batch_idx % 10 == 0:  # 每10个批次检查一次梯度
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_min = param.grad.min().item()
                                grad_max = param.grad.max().item()
                                if grad_min < -1e3 or grad_max > 1e3:
                                    self.logger.warning(f"梯度范围过大 - {name}: [{grad_min:.4f}, {grad_max:.4f}]")
                                if torch.isnan(param.grad).any():
                                    self.logger.error(f"梯度包含NaN - {name}")
                                if torch.isinf(param.grad).any():
                                    self.logger.error(f"梯度包含Inf - {name}")
                    
                    self.logger.debug("完成反向传播")
                    self.optimizer.step()
                    self.logger.debug("完成优化器步骤")
                    
                    # 更新进度条
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # 计算指标
                    self.train_metrics.update(pred_masks, target_masks)
                    
                    if batch_idx == 0:  # 打印第一个批次的详细信息
                        self.logger.info(f"第一个批次完成:")
                        self.logger.info(f"- 损失: {loss.item()}")
                        self.logger.info(f"- 预测掩码形状: {pred_masks.shape}")
                        self.logger.info(f"- 目标掩码形状: {target_masks.shape}")
                    
                except Exception as e:
                    self.logger.error(f"批次 {batch_idx} 处理出错: {str(e)}")
                    self.logger.error(f"批次结构: {type(batch)}")
                    if isinstance(batch, dict):
                        for k, v in batch.items():
                            self.logger.error(f"键 '{k}': type={type(v)}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"train_epoch出错: {str(e)}")
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
                        if self.model_type in ['mask_rcnn', 'detr']:
                            outputs = self.model(images, targets)
                        else:
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
                                target_masks = torch.stack([t['semantic_mask'] for t in targets])
                            else:  # panoptic
                                pred_masks = outputs['pred_panoptic']
                                target_masks = torch.cat([t['panoptic_mask'] for t in targets], dim=0)
                        elif self.model_type == 'mask_rcnn':
                            # Mask R-CNN的输出已经包含了损失
                            loss = sum(outputs['losses'].values()) if 'losses' in outputs else outputs['loss']
                            # 获取预测掩码
                            pred_masks = outputs['pred_masks']
                            target_masks = torch.cat([t['masks'] for t in targets], dim=0)
                        elif self.model_type in ['deeplabv3', 'deeplabv3plus']:
                            # DeepLabV3/DeepLabV3+的输出是字典，包含'out'键
                            if isinstance(outputs, dict) and 'out' in outputs:
                                print(f"[DEBUG] DeepLabV3输出格式: {list(outputs.keys())}")
                                print(f"[DEBUG] DeepLabV3 'out'形状: {outputs['out'].shape}")
                                print(f"[DEBUG] DeepLabV3 'out'范围: [{outputs['out'].min().item():.3f}, {outputs['out'].max().item():.3f}]")
                                
                                # 确保targets是正确的格式
                                if isinstance(targets, list) and 'semantic_mask' in targets[0]:
                                    target_masks = torch.stack([t['semantic_mask'] for t in targets])
                                    print(f"[DEBUG] 目标掩码形状: {target_masks.shape}")
                                    print(f"[DEBUG] 目标掩码范围: [{target_masks.min().item()}, {target_masks.max().item()}]")
                                    loss = self.criterion(outputs['out'], target_masks)
                                else:
                                    print(f"[DEBUG] 目标格式: {type(targets)}")
                                    if isinstance(targets, torch.Tensor):
                                        print(f"[DEBUG] 目标张量形状: {targets.shape}")
                                        print(f"[DEBUG] 目标张量范围: [{targets.min().item()}, {targets.max().item()}]")
                                        # 记录目标中的唯一类别
                                        unique_targets = torch.unique(targets)
                                        print(f"[DEBUG] 目标唯一类别: {unique_targets}")
                                    loss = self.criterion(outputs['out'], targets)
                                
                                # 获取预测掩码
                                pred_masks = torch.argmax(outputs['out'], dim=1)
                                print(f"[DEBUG] 预测掩码形状: {pred_masks.shape}")
                                print(f"[DEBUG] 预测掩码范围: [{pred_masks.min().item()}, {pred_masks.max().item()}]")
                                print(f"[DEBUG] 预测掩码唯一值: {torch.unique(pred_masks)}")
                                
                                # 检查预测类别与目标类别的匹配情况
                                if isinstance(targets, torch.Tensor):
                                    target_classes = torch.unique(targets).cpu().numpy()
                                    pred_classes = torch.unique(pred_masks).cpu().numpy()
                                    missing_classes = set(target_classes) - set(pred_classes)
                                    extra_classes = set(pred_classes) - set(target_classes)
                                    print(f"[DEBUG] 目标中存在但预测中缺失的类别: {missing_classes}")
                                    print(f"[DEBUG] 预测中存在但目标中缺失的类别: {extra_classes}")
                                
                                if isinstance(targets, list) and 'semantic_mask' in targets[0]:
                                    target_masks = torch.stack([t['semantic_mask'] for t in targets])
                                else:
                                    target_masks = targets
                            else:
                                print(f"[WARNING] DeepLabV3输出格式不符合预期: {type(outputs)}")
                                if isinstance(outputs, dict):
                                    print(f"[WARNING] 可用键: {list(outputs.keys())}")
                                loss = self.criterion(outputs, targets)
                                # 尝试获取预测掩码
                                if isinstance(outputs, dict) and 'pred_masks' in outputs:
                                    pred_masks = torch.argmax(outputs['pred_masks'], dim=1)
                                else:
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