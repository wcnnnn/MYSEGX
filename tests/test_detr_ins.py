"""DETR模型实例分割测试模块"""

import unittest
import torch
from MYSEGX.models.detr import DETR
import warnings
warnings.filterwarnings("ignore")


class TestDETRInstance(unittest.TestCase):
    """测试DETR模型的实例分割能力"""
    
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
        
    def test_instance_segmentation(self):
        """测试实例分割模式"""
        # 初始化实例分割模型
        model = DETR(
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            hidden_dim=self.hidden_dim,
            task_type='instance'
        )
        
        # 验证模型组件
        self.assertIsNotNone(model.backbone)
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.decoder)  # 实例分割需要解码器
        self.assertIsNotNone(model.query_embed)  # 实例分割需要目标查询
        self.assertIsNotNone(model.class_embed)  # 实例分割需要分类头
        self.assertIsNotNone(model.mask_head)  # 验证掩码头存在
        
        # 使用模拟数据测试前向传播
        outputs = model(self.mock_input)
        
        # 验证输出
        self.assertIn('pred_masks', outputs)
        self.assertIn('pred_logits', outputs)  # 实例分割需要输出类别logits
        
        # 验证输出形状
        pred_masks = outputs['pred_masks']
        pred_logits = outputs['pred_logits']
        
        # 验证掩码输出形状 (batch_size, num_queries, H/4, W/4)
        self.assertEqual(
            pred_masks.shape,
            (self.batch_size, self.num_queries, self.input_size[0]//4, self.input_size[1]//4)
        )
        
        # 验证类别logits输出形状 (batch_size, num_queries, num_classes + 1)
        self.assertEqual(
            pred_logits.shape,
            (self.batch_size, self.num_queries, self.num_classes + 1)  # +1表示背景类
        )
        
        # 验证掩码输出在有效范围内
        self.assertTrue((pred_masks >= -1).all() and 
                      (pred_masks <= 1).all(),
                      "掩码预测值超出[-1,1]范围")
        
        # 验证logits输出为有效值
        self.assertTrue(torch.isfinite(pred_logits).all(),
                      "类别logits包含无效值")
        
    def test_different_backbones(self):
        """测试实例分割在不同主干网络下的表现"""
        backbones = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19']
        task_type = 'instance'  # 使用实例分割任务
        
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
            self.assertIsNotNone(model.decoder)  # 实例分割需要解码器
            self.assertIsNotNone(model.query_embed)  # 实例分割需要目标查询
            self.assertIsNotNone(model.class_embed)  # 实例分割需要分类头
            self.assertIsNotNone(model.mask_head)
            
            # 使用模拟数据测试前向传播
            outputs = model(self.mock_input)
            
            # 验证输出
            self.assertIn('pred_masks', outputs)
            self.assertIn('pred_logits', outputs)  # 实例分割需要输出类别logits
            
            # 验证输出形状
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits']
            
            # 验证掩码输出形状
            self.assertEqual(
                pred_masks.shape,
                (self.batch_size, self.num_queries, self.input_size[0]//4, self.input_size[1]//4)
            )
            
            # 验证类别logits输出形状
            self.assertEqual(
                pred_logits.shape,
                (self.batch_size, self.num_queries, self.num_classes + 1)  # +1表示背景类
            )
            
            # 验证输出数值范围
            self.assertTrue(torch.isfinite(pred_masks).all(),
                          f"{backbone}的预测掩码包含无效值")
            self.assertTrue(torch.isfinite(pred_logits).all(),
                          f"{backbone}的类别logits包含无效值")
            
            # 验证掩码输出在有效范围内
            self.assertTrue((pred_masks >= -1).all() and 
                          (pred_masks <= 1).all(),
                          f"{backbone}的掩码预测值超出[-1,1]范围")

if __name__ == '__main__':
    unittest.main()