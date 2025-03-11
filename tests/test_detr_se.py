"""DETR模型测试模块"""

import unittest
import torch
from MYSEGX.models.detr import DETR
import warnings
warnings.filterwarnings("ignore")


class TestDETR(unittest.TestCase):
    """测试DETR模型的语义分割能力"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置基本参数
        self.batch_size = 2  # 批次大小
        self.num_classes = 20  # 类别数量
        self.num_queries = 100  # 目标查询数量
        self.hidden_dim = 256  # 隐藏层维度
        self.input_size = (800, 800)  # 输入图像大小
        
        # 生成模拟输入数据
        self.mock_input = torch.randn(self.batch_size, 3, *self.input_size)  # 模拟RGB图像输入
        
    def test_semantic_segmentation(self):
        """测试语义分割模式"""
        # 初始化语义分割模型
        model = DETR(
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            hidden_dim=self.hidden_dim,
            task_type='semantic'
        )
        
        # 验证模型组件
        self.assertIsNotNone(model.backbone)
        self.assertIsNotNone(model.encoder)
        self.assertIsNone(model.decoder)  # 语义分割不需要解码器
        self.assertIsNone(model.query_embed)  # 语义分割不需要目标查询
        self.assertIsNone(model.class_embed)  # 语义分割不需要分类头
        self.assertIsNotNone(model.mask_head)  # 验证掩aeda存在
        
        # 使用模拟数据测试前向传播
        outputs = model(self.mock_input)
        
        # 验证输出
        self.assertIn('pred_masks', outputs)
        self.assertNotIn('pred_logits', outputs)  # 语义分割不输出类别logits
        
        # 验证掩码输出形状
        pred_masks = outputs['pred_masks']
        self.assertEqual(
            pred_masks.shape,
            (self.batch_size, self.num_classes, self.input_size[0]//4, self.input_size[1]//4)
        )
        
    def test_different_backbones(self):
        """测试语义分割在不同主干网络下的表现"""
        backbones = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19',
                    ]
        task_type = 'semantic'  # 使用语义分割任务
        
        for backbone in backbones:
            # 初始化模型
            model = DETR(
                num_classes=self.num_classes,
                num_queries=self.num_queries,
                hidden_dim=self.hidden_dim,
                backbone_type=backbone,
                task_type=task_type
            )
            
            # 验证主干网络类型和基本组件
            self.assertIsNotNone(model.backbone)
            self.assertIsNotNone(model.encoder)
            self.assertIsNone(model.decoder)  # 语义分割不需要解码器
            self.assertIsNone(model.query_embed)  # 语义分割不需要目标查询
            self.assertIsNone(model.class_embed)  # 语义分割不需要分类头
            self.assertIsNotNone(model.mask_head)
            
            # 使用模拟数据测试前向传播
            outputs = model(self.mock_input)
            
            # 验证输出
            self.assertIn('pred_masks', outputs)
            self.assertNotIn('pred_logits', outputs)  # 语义分割不输出类别logits
            
            # 验证输出形状
            pred_masks = outputs['pred_masks']
            self.assertEqual(
                pred_masks.shape,
                (self.batch_size, self.num_classes, self.input_size[0]//4, self.input_size[1]//4)
            )
            
            # 验证输出数值范围
            self.assertTrue(torch.isfinite(pred_masks).all(),
                          f"{backbone}的预测掩码包含无效值")
            
            # 验证掩码输出在有效范围内
            self.assertTrue((pred_masks >= -1).all() and 
                          (pred_masks <= 1).all(),
                          f"{backbone}的掩码预测值超出[-1,1]范围")

if __name__ == '__main__':
    unittest.main()