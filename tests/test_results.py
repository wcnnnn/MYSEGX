"""测试结果保存模块"""
import unittest
import os
import sys
import shutil
import tempfile
import numpy as np
import torch

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MYSEGX.utils.results import ResultSaver

class TestResultSaver(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.result_saver = ResultSaver(base_dir=self.test_dir)
        
        # 创建测试数据
        self.config = {
            'model': {'name': 'test_model', 'num_classes': 2},
            'train': {'batch_size': 8, 'epochs': 10}
        }
        
        self.metrics = {
            'accuracy': 0.85,
            'iou': 0.75,
            'loss': 0.25
        }
        
        self.history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'train_metrics': {
                'accuracy': [0.8, 0.85, 0.9],
                'iou': [0.7, 0.75, 0.8]
            },
            'val_metrics': {
                'accuracy': [0.75, 0.8, 0.85],
                'iou': [0.65, 0.7, 0.75]
            }
        }
        
        # 创建测试图像和掩码
        self.image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.mask = np.random.randint(0, 2, (224, 224), dtype=np.uint8)
        
        # GPU tensor
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.tensor_image = torch.from_numpy(self.image).to(self.device)
        self.tensor_mask = torch.from_numpy(self.mask).to(self.device)
        
    def tearDown(self):
        """清理测试环境"""
        # 删除测试目录及其内容
        shutil.rmtree(self.test_dir)
        
    def test_directory_creation(self):
        """测试目录创建"""
        self.assertTrue(os.path.exists(self.result_saver.exp_dir))
        self.assertTrue(os.path.exists(self.result_saver.img_dir))
        self.assertTrue(os.path.exists(self.result_saver.mask_dir))
        self.assertTrue(os.path.exists(self.result_saver.plot_dir))
        
    def test_save_config(self):
        """测试配置保存"""
        self.result_saver.save_config(self.config)
        config_path = os.path.join(self.result_saver.exp_dir, 'config.json')
        self.assertTrue(os.path.exists(config_path))
        
    def test_save_metrics(self):
        """测试指标保存"""
        self.result_saver.save_metrics(self.metrics)
        metrics_path = os.path.join(self.result_saver.exp_dir, 'metrics.json')
        self.assertTrue(os.path.exists(metrics_path))
        
    def test_save_training_history(self):
        """测试训练历史保存"""
        self.result_saver.save_training_history(self.history)
        history_path = os.path.join(self.result_saver.exp_dir, 'history.json')
        self.assertTrue(os.path.exists(history_path))
        
    def test_save_prediction_numpy(self):
        """测试使用numpy数组保存预测结果"""
        filename = 'test_pred.png'
        self.result_saver.save_prediction(self.image, self.mask, filename)
        
        # 检查是否生成了所有预期的文件
        img_path = os.path.join(self.result_saver.img_dir, filename)
        mask_path = os.path.join(self.result_saver.mask_dir, 'test_pred_mask.png')
        overlay_path = os.path.join(self.result_saver.img_dir, 'test_pred_overlay.png')
        
        self.assertTrue(os.path.exists(img_path))
        self.assertTrue(os.path.exists(mask_path))
        self.assertTrue(os.path.exists(overlay_path))
        
    def test_save_prediction_tensor(self):
        """测试使用tensor保存预测结果"""
        filename = 'test_pred_tensor.png'
        self.result_saver.save_prediction(
            self.tensor_image.cpu().numpy(),
            self.tensor_mask.cpu().numpy(),
            filename
        )
        
        # 检查是否生成了所有预期的文件
        img_path = os.path.join(self.result_saver.img_dir, filename)
        mask_path = os.path.join(self.result_saver.mask_dir, 'test_pred_tensor_mask.png')
        overlay_path = os.path.join(self.result_saver.img_dir, 'test_pred_tensor_overlay.png')
        
        self.assertTrue(os.path.exists(img_path))
        self.assertTrue(os.path.exists(mask_path))
        self.assertTrue(os.path.exists(overlay_path))
        
    def test_save_plot(self):
        """测试获取绘图保存路径"""
        plot_name = 'test_plot'
        plot_path = self.result_saver.save_plot(plot_name)
        expected_path = os.path.join(self.result_saver.plot_dir, 'test_plot.png')
        self.assertEqual(plot_path, expected_path)

if __name__ == '__main__':
    unittest.main()
