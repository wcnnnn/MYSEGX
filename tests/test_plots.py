"""测试绘图模块"""
import unittest
import torch
import numpy as np
import os
import sys
import tempfile

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MYSEGX.utils.plots import plot_segmentation, plot_training_curves, plot_comparison

class TestPlots(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录用于保存测试生成的图片
        self.test_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.mask = np.random.randint(0, 2, (224, 224), dtype=np.uint8)
        
        # 创建GPU和CPU上的tensor数据
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.tensor_image = torch.from_numpy(self.image).to(self.device)
        self.tensor_mask = torch.from_numpy(self.mask).to(self.device)
        
        # 创建训练历史数据
        self.losses = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4]
        }
        self.metrics = {
            'accuracy': [0.8, 0.85, 0.9],
            'iou': [0.7, 0.75, 0.8]
        }
        
        # 转换部分数据为tensor
        self.tensor_losses = {
            'train_loss': torch.tensor([0.5, 0.4, 0.3]).to(self.device),
            'val_loss': torch.tensor([0.6, 0.5, 0.4]).to(self.device)
        }
        
    def tearDown(self):
        """清理测试环境"""
        # 删除测试过程中创建的文件
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
        
    def test_plot_segmentation_numpy(self):
        """测试使用numpy数组绘制分割结果"""
        save_path = os.path.join(self.test_dir, 'seg_numpy.png')
        plot_segmentation(self.image, self.mask, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
        
    def test_plot_segmentation_tensor(self):
        """测试使用tensor绘制分割结果"""
        save_path = os.path.join(self.test_dir, 'seg_tensor.png')
        plot_segmentation(
            self.tensor_image.cpu().numpy(), 
            self.tensor_mask.cpu().numpy(), 
            save_path=save_path
        )
        self.assertTrue(os.path.exists(save_path))
        
    def test_plot_training_curves_list(self):
        """测试使用列表数据绘制训练曲线"""
        save_path = os.path.join(self.test_dir, 'curves_list.png')
        plot_training_curves(self.losses, self.metrics, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
        
    def test_plot_training_curves_tensor(self):
        """测试使用tensor数据绘制训练曲线"""
        save_path = os.path.join(self.test_dir, 'curves_tensor.png')
        plot_training_curves(self.tensor_losses, self.metrics, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
        
    def test_plot_comparison(self):
        """测试图像对比功能"""
        save_path = os.path.join(self.test_dir, 'comparison.png')
        images = [self.image, self.mask, self.mask]
        titles = ['Image', 'Mask 1', 'Mask 2']
        plot_comparison(images, titles, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
        
    def test_plot_comparison_tensor(self):
        """测试使用tensor进行图像对比"""
        save_path = os.path.join(self.test_dir, 'comparison_tensor.png')
        images = [
            self.tensor_image.cpu().numpy(),
            self.tensor_mask.cpu().numpy(),
            self.tensor_mask.cpu().numpy()
        ]
        titles = ['Image', 'Mask 1', 'Mask 2']
        plot_comparison(images, titles, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

if __name__ == '__main__':
    unittest.main()
