"""SA-UNet模型测试模块"""

import unittest
import torch
from MYSEGX.models.saunet import SA_UNet
import warnings
warnings.filterwarnings("ignore")


class TestSAUNet(unittest.TestCase):
    """测试SA-UNet模型的语义分割能力"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置基本参数
        self.batch_size = 2  # 批次大小
        self.num_classes = 20  # 类别数量
        self.input_size = (512, 512)  # 输入图像大小
        
        # 生成模拟输入数据
        self.mock_input = torch.randn(self.batch_size, 3, *self.input_size)  # 模拟RGB图像输入
        
    def test_semantic_segmentation(self):
        """测试语义分割模式"""
        # 初始化语义分割模型
        model = SA_UNet(
            n_channels=3,
            n_classes=self.num_classes,
            bilinear=True
        )
        
        # 验证模型组件
        self.assertIsNotNone(model.conv1)  # 初始卷积层
        self.assertIsNotNone(model.down1)  # 下采样路径
        self.assertIsNotNone(model.down2)
        self.assertIsNotNone(model.down3)
        self.assertIsNotNone(model.down4)
        self.assertIsNotNone(model.attn)  # 注意力模块
        self.assertIsNotNone(model.conv2)  # 注意力后的卷积
        self.assertIsNotNone(model.up1)  # 上采样路径
        self.assertIsNotNone(model.up2)
        self.assertIsNotNone(model.up3)
        self.assertIsNotNone(model.up4)
        self.assertIsNotNone(model.out_conv)  # 输出层
        
        # 使用模拟数据测试前向传播
        outputs = model(self.mock_input)
        
        # 验证输出形状
        self.assertEqual(
            outputs.shape,
            (self.batch_size, self.num_classes, self.input_size[0], self.input_size[1])
        )
        
        # 验证输出数值范围
        self.assertTrue(torch.isfinite(outputs).all(),
                      "预测输出包含无效值")
        
        # 验证训练模式下的输出范围
        self.assertTrue((outputs >= -20).all() and 
                      (outputs <= 20).all(),
                      "训练模式下的logits超出合理范围")
        
        # 测试推理模式
        model.eval()
        with torch.no_grad():
            outputs = model(self.mock_input)
            # 验证softmax后的输出范围
            self.assertTrue((outputs >= 0).all() and 
                          (outputs <= 1).all(),
                          "推理模式下的softmax输出超出[0,1]范围")
            
    def test_different_configurations(self):
        """测试不同配置下的SA-UNet模型"""
        configs = [
            {'bilinear': True, 'n_channels': 3, 'n_classes': 2},  # 二分类
            {'bilinear': False, 'n_channels': 3, 'n_classes': 20},  # 多分类
            {'bilinear': True, 'n_channels': 1, 'n_classes': 1},  # 单通道输入
        ]
        
        for config in configs:
            # 初始化模型
            model = SA_UNet(**config)
            
            # 调整输入通道数以匹配配置
            if config['n_channels'] == 1:
                mock_input = torch.randn(self.batch_size, 1, *self.input_size)
            else:
                mock_input = self.mock_input
            
            # 使用模拟数据测试前向传播
            outputs = model(mock_input)
            
            # 验证输出形状
            self.assertEqual(
                outputs.shape,
                (self.batch_size, config['n_classes'], self.input_size[0], self.input_size[1])
            )
            
            # 验证输出数值范围
            self.assertTrue(torch.isfinite(outputs).all(),
                          f"配置{config}的预测输出包含无效值")
            
            # 验证训练模式下的输出范围
            self.assertTrue((outputs >= -20).all() and 
                          (outputs <= 20).all(),
                          f"配置{config}的训练模式logits超出合理范围")
            
            # 测试推理模式
            model.eval()
            with torch.no_grad():
                outputs = model(mock_input)
                # 验证softmax后的输出范围
                self.assertTrue((outputs >= 0).all() and 
                              (outputs <= 1).all(),
                              f"配置{config}的推理模式softmax输出超出[0,1]范围")
                
    def test_feature_ranges(self):
        """测试特征值范围"""
        model = SA_UNet(
            n_channels=3,
            n_classes=self.num_classes,
            bilinear=True
        )
        
        # 使用较小的输入进行测试
        test_input = torch.randn(1, 3, 64, 64)
        
        # 获取中间特征
        x1 = model.conv1(test_input)
        x2 = model.down1(x1)
        x3 = model.down2(x2)
        x4 = model.down3(x3)
        x5 = model.down4(x4)
        
        # 验证下采样路径的特征范围
        self.assertTrue((x1 >= -10).all() and (x1 <= 10).all(),
                      "初始特征超出合理范围")
        self.assertTrue((x2 >= -10).all() and (x2 <= 10).all(),
                      "down1特征超出合理范围")
        self.assertTrue((x3 >= -10).all() and (x3 <= 10).all(),
                      "down2特征超出合理范围")
        self.assertTrue((x4 >= -10).all() and (x4 <= 10).all(),
                      "down3特征超出合理范围")
        self.assertTrue((x5 >= -10).all() and (x5 <= 10).all(),
                      "down4特征超出合理范围")
        
        # 测试注意力模块
        x6 = model.attn(x5)
        self.assertTrue((x6 >= -10).all() and (x6 <= 10).all(),
                      "注意力模块输出超出合理范围")
        
        x7 = model.conv2(x6)
        self.assertTrue((x7 >= -10).all() and (x7 <= 10).all(),
                      "注意力后卷积输出超出合理范围")
        
        # 测试上采样路径
        x = model.up1(x7, x4)
        self.assertTrue((x >= -10).all() and (x <= 10).all(),
                      "up1特征超出合理范围")
        
        x = model.up2(x, x3)
        self.assertTrue((x >= -10).all() and (x <= 10).all(),
                      "up2特征超出合理范围")
        
        x = model.up3(x, x2)
        self.assertTrue((x >= -10).all() and (x <= 10).all(),
                      "up3特征超出合理范围")
        
        x = model.up4(x, x1)
        self.assertTrue((x >= -10).all() and (x <= 10).all(),
                      "up4特征超出合理范围")
        
        # 测试最终归一化
        x = model.final_norm1(x)
        x = model.final_norm2(x)
        x = model.final_act(x)
        x = model.pre_out_norm(x)
        self.assertTrue((x >= -10).all() and (x <= 10).all(),
                      "最终归一化输出超出合理范围")


if __name__ == '__main__':
    unittest.main() 