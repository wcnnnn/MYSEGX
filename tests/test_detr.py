"""DETR模型测试模块"""

import unittest
import torch
from segfra.models.detr.detr import DETR

class TestDETR(unittest.TestCase):
    """测试DETR模型的构建和前向传播"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.batch_size = 2
        self.num_classes = 20
        self.num_queries = 100
        self.hidden_dim = 256
        self.model = DETR(
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            hidden_dim=self.hidden_dim
        )
        
    def test_model_initialization(self):
        """测试模型初始化"""
        # 验证主干网络
        self.assertIsNotNone(self.model.backbone)
        
        # 验证Transformer编码器和解码器
        self.assertEqual(len(self.model.encoder), 6)
        self.assertEqual(len(self.model.decoder), 6)
        
        # 验证目标查询嵌入
        self.assertEqual(
            self.model.query_embed.weight.shape,
            (self.num_queries, self.hidden_dim)
        )
        
        # 验证分类头
        self.assertEqual(
            self.model.class_embed.out_features,
            self.num_classes + 1  # +1表示背景类
        )
        
    def test_forward_pass(self):
        """测试前向传播"""
        # 创建模拟输入
        x = torch.randn(self.batch_size, 3, 800, 800)
        
        # 前向传播
        outputs = self.model(x)
        
        # 验证输出字典的键
        self.assertIn('pred_logits', outputs)
        self.assertIn('pred_masks', outputs)
        
        # 验证分类输出的形状
        self.assertEqual(
            outputs['pred_logits'].shape,
            (self.batch_size, self.num_queries, self.num_classes + 1)
        )
        
        # 验证掩码输出的形状
        pred_masks = outputs['pred_masks']
        self.assertEqual(pred_masks.shape[0], self.batch_size)
        self.assertEqual(pred_masks.shape[1], self.num_queries)
        
    def test_backbone_feature_extraction(self):
        """测试主干网络的特征提取"""
        x = torch.randn(self.batch_size, 3, 800, 800)
        features = self.model.backbone(x)
        
        # 验证特征金字塔输出
        self.assertEqual(len(features), 4)
        
        # 验证最后一层特征的通道数
        self.assertEqual(features[-1].shape[1], 2048)

if __name__ == '__main__':
    unittest.main()