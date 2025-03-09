"""VOC数据集测试模块"""

import unittest
import torch
from segfra.data import VOCSegmentation, create_voc_dataloader
from segfra.data.transforms import get_training_transform

class TestVOCDataset(unittest.TestCase):
    """测试VOC数据集的加载和预处理"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.root = 'VOC2012'
        self.transform = get_training_transform()
        self.dataset = VOCSegmentation(
            root=self.root,
            split='train',
            transform=self.transform
        )
        
    def test_dataset_initialization(self):
        """测试数据集初始化"""
        self.assertIsNotNone(self.dataset)
        self.assertTrue(len(self.dataset) > 0)
        
    def test_sample_format(self):
        """测试单个样本的格式"""
        sample = self.dataset[0]
        
        # 验证返回的是字典
        self.assertIsInstance(sample, dict)
        
        # 验证必要的键
        self.assertIn('image', sample)
        self.assertIn('target', sample)
        self.assertIn('image_id', sample)
        
        # 验证图像格式
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(len(sample['image'].shape), 3)  # (C, H, W)
        self.assertEqual(sample['image'].shape[0], 3)    # RGB通道
        
        # 验证目标字典格式
        target = sample['target']
        self.assertIn('masks', target)
        self.assertIn('labels', target)
        self.assertIn('boxes', target)
        self.assertIn('image_id', target)
        
        # 验证掩码格式
        self.assertIsInstance(target['masks'], torch.Tensor)
        self.assertEqual(len(target['masks'].shape), 3)  # (N, H, W)
        
        # 验证标签格式
        self.assertIsInstance(target['labels'], torch.Tensor)
        self.assertEqual(target['labels'].dtype, torch.long)
        
        # 验证边界框格式
        self.assertIsInstance(target['boxes'], torch.Tensor)
        self.assertEqual(target['boxes'].shape[-1], 4)  # (x1, y1, x2, y2)
        
    def test_dataloader(self):
        """测试数据加载器"""
        batch_size = 4
        dataloader = create_voc_dataloader(
            root=self.root,
            split='train',
            batch_size=batch_size,
            num_workers=0  # 测试时使用单进程
        )
        
        # 获取一个批次
        batch = next(iter(dataloader))
        
        # 验证批次格式
        self.assertIsInstance(batch, dict)
        self.assertIn('image', batch)
        self.assertIn('target', batch)
        self.assertIn('image_id', batch)
        
        # 验证批次大小
        self.assertEqual(batch['image'].shape[0], batch_size)
        self.assertEqual(len(batch['target']), batch_size)
        
        # 验证每个样本的目标字典
        for target in batch['target']:
            self.assertIn('masks', target)
            self.assertIn('labels', target)
            self.assertIn('boxes', target)
            self.assertIn('image_id', target)
            
            # 验证掩码形状
            self.assertEqual(len(target['masks'].shape), 3)  # (N, H, W)
            
            # 验证标签和边界框数量匹配
            self.assertEqual(len(target['labels']), len(target['boxes']))
            
            # 验证边界框格式
            self.assertEqual(target['boxes'].shape[-1], 4)

if __name__ == '__main__':
    unittest.main()