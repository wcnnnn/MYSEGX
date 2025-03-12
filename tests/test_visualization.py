import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import tempfile
from MYSEGX.utils.plots import plot_segmentation

class TestVisualization(unittest.TestCase):
    def setUp(self):
        # 创建一个临时目录来保存测试图像
        self.test_dir = tempfile.mkdtemp()
        
        # 创建一个简单的测试图像（黑色背景上的彩色矩形）
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 添加一个红色矩形（人）
        self.image[20:40, 20:40] = [255, 0, 0]
        # 添加一个绿色矩形（车）
        self.image[60:80, 60:80] = [0, 255, 0]
        
        # 创建类别名称
        self.class_names = ['background', 'person', 'car', 'tree']
        
        # 创建类别颜色
        self.class_colors = {
            0: (0, 0, 0),      # 背景（黑色）
            1: (255, 0, 0),    # 人（红色）
            2: (0, 255, 0),    # 车（绿色）
            3: (0, 0, 255)     # 树（蓝色）
        }
    
    def test_semantic_segmentation(self):
        """测试语义分割的可视化"""
        # 创建语义分割掩码（与图像中的矩形对应）
        target = np.zeros((100, 100), dtype=np.uint8)
        target[20:40, 20:40] = 1  # 人（红色矩形区域）
        target[60:80, 60:80] = 2  # 车（绿色矩形区域）
        
        # 创建预测掩码（故意偏移一点，以显示差异）
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[25:45, 25:45] = 1  # 预测的人位置略有偏移
        pred[60:80, 60:80] = 2  # 车的预测正确
        
        # 测试 NumPy 数组输入
        save_path = os.path.join(self.test_dir, 'semantic.png')
        result = plot_segmentation(
            self.image, target, pred,
            task_type='semantic',
            class_colors=self.class_colors,
            class_names=self.class_names,
            save_path=save_path
        )
        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(result.shape, (3, 100, 100))
        
        # 测试 PyTorch 张量输入
        torch_image = torch.from_numpy(self.image.transpose(2, 0, 1))
        torch_target = torch.from_numpy(target)
        torch_pred = torch.from_numpy(pred)
        
        result = plot_segmentation(
            torch_image, torch_target, torch_pred,
            task_type='semantic',
            class_colors=self.class_colors,
            class_names=self.class_names
        )
        self.assertEqual(result.shape, (3, 100, 100))
    
    def test_instance_segmentation(self):
        """测试实例分割的可视化"""
        # 创建实例分割掩码
        target = np.zeros((100, 100), dtype=np.uint8)
        target[20:40, 20:40] = 1  # 实例1
        target[60:80, 60:80] = 2  # 实例2
        
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[25:45, 25:45] = 1  # 预测的实例1位置略有偏移
        pred[60:80, 60:80] = 2  # 实例2的预测正确
        
        save_path = os.path.join(self.test_dir, 'instance.png')
        result = plot_segmentation(
            self.image, target, pred,
            task_type='instance',
            save_path=save_path
        )
        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(result.shape, (3, 100, 100))
    
    def test_panoptic_segmentation(self):
        """测试全景分割的可视化"""
        # 创建全景分割掩码（语义+实例）
        target_semantic = np.zeros((100, 100), dtype=np.uint8)
        target_semantic[20:40, 20:40] = 1  # 人
        target_semantic[60:80, 60:80] = 2  # 车
        
        target_instance = np.zeros((100, 100), dtype=np.uint8)
        target_instance[20:40, 20:40] = 1  # 实例1
        target_instance[60:80, 60:80] = 2  # 实例2
        
        target = np.stack([target_semantic, target_instance], axis=-1)
        
        pred_semantic = np.zeros((100, 100), dtype=np.uint8)
        pred_semantic[25:45, 25:45] = 1  # 预测的人位置略有偏移
        pred_semantic[60:80, 60:80] = 2  # 车的预测正确
        
        pred_instance = np.zeros((100, 100), dtype=np.uint8)
        pred_instance[25:45, 25:45] = 1  # 预测的实例1位置略有偏移
        pred_instance[60:80, 60:80] = 2  # 实例2的预测正确
        
        pred = np.stack([pred_semantic, pred_instance], axis=-1)
        
        save_path = os.path.join(self.test_dir, 'panoptic.png')
        result = plot_segmentation(
            self.image, target, pred,
            task_type='panoptic',
            class_colors=self.class_colors,
            class_names=self.class_names,
            save_path=save_path
        )
        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(result.shape, (3, 100, 100))
    
    def test_invalid_inputs(self):
        """测试无效输入的处理"""
        # 创建一个简单的掩码
        target = np.zeros((100, 100), dtype=np.uint8)
        pred = np.zeros((100, 100), dtype=np.uint8)
        
        # 测试无效的任务类型
        with self.assertRaises(ValueError):
            plot_segmentation(
                self.image, target, pred,
                task_type='invalid_type'
            )
        
        # 测试形状不匹配的掩码
        invalid_mask = np.zeros((50, 50), dtype=np.uint8)
        with self.assertRaises(ValueError):
            plot_segmentation(
                self.image, invalid_mask, pred,
                task_type='semantic'
            )
    
    def tearDown(self):
        # 清理临时文件
        for filename in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, filename))
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()