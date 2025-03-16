"""Mask R-CNN模型实例分割测试模块"""

import unittest
import torch
from MYSEGX.models.mask_rcnn.mask_rcnn import MaskRCNN
import warnings
warnings.filterwarnings("ignore")


class TestMaskRCNN(unittest.TestCase):
    """测试Mask R-CNN模型的实例分割能力"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置基本参数
        self.batch_size = 2  # 批次大小
        self.num_classes = 20  # 类别数量（包括背景）
        self.input_size = (512, 512)  # 输入图像大小
        
        # 生成模拟输入数据
        self.mock_input = torch.randn(self.batch_size, 3, *self.input_size)  # 模拟RGB图像输入
        
        # 生成模拟目标数据（用于训练模式测试）
        self.mock_targets = []
        for i in range(self.batch_size):
            # 每个图像包含1-3个目标
            num_objects = torch.randint(1, 4, (1,)).item()
            
            # 随机生成边界框 [x1, y1, x2, y2]
            boxes = torch.rand(num_objects, 4)
            # 确保 x1 < x2, y1 < y2
            boxes[:, 2:] = boxes[:, :2] + torch.rand(num_objects, 2) * 0.5
            # 缩放到图像尺寸
            boxes[:, [0, 2]] *= self.input_size[1]
            boxes[:, [1, 3]] *= self.input_size[0]
            
            # 随机生成类别标签（1到num_classes-1，0为背景）
            labels = torch.randint(1, self.num_classes, (num_objects,))
            
            # 随机生成掩码（简化为边界框内的矩形掩码）
            masks = torch.zeros(num_objects, self.input_size[0], self.input_size[1])
            for j in range(num_objects):
                x1, y1, x2, y2 = boxes[j].int().tolist()
                masks[j, y1:y2, x1:x2] = 1
            
            self.mock_targets.append({
                'boxes': boxes,
                'labels': labels,
                'masks': masks
            })
    
    def test_model_initialization(self):
        """测试模型初始化"""
        # 测试不同的主干网络
        backbone_types = ['resnet18']
        
        for backbone_type in backbone_types:
            # 初始化模型
            model = MaskRCNN(
                num_classes=self.num_classes,
                backbone_type=backbone_type
            )
            
            # 验证模型组件
            self.assertIsNotNone(model.backbone)
            self.assertIsNotNone(model.fpn)
            self.assertIsNotNone(model.rpn)
            self.assertIsNotNone(model.box_roi_pool)
            self.assertIsNotNone(model.box_head)
            self.assertIsNotNone(model.mask_roi_pool)
            self.assertIsNotNone(model.mask_head)
            
            # 验证模型可以切换到评估模式
            model.eval()
            self.assertFalse(model.training)
    
    def test_inference_mode(self):
        """测试推理模式"""
        # 初始化模型
        model = MaskRCNN(num_classes=self.num_classes)
        model.eval()  # 设置为评估模式
        
        # 使用模拟数据测试前向传播
        with torch.no_grad():
            outputs = model(self.mock_input)
        
        # 验证输出字典包含必要的键
        self.assertIn('pred_boxes', outputs)
        self.assertIn('pred_scores', outputs)
        self.assertIn('pred_labels', outputs)
        self.assertIn('pred_masks', outputs)
        
        # 验证输出形状和类型
        self.assertIsInstance(outputs['pred_boxes'], torch.Tensor)
        self.assertIsInstance(outputs['pred_scores'], torch.Tensor)
        self.assertIsInstance(outputs['pred_labels'], torch.Tensor)
        self.assertIsInstance(outputs['pred_masks'], torch.Tensor)
        
        # 验证预测分数在有效范围内
        if outputs['pred_scores'].numel() > 0:
            self.assertTrue((outputs['pred_scores'] >= 0).all() and 
                          (outputs['pred_scores'] <= 1).all(),
                          "预测分数超出[0,1]范围")
        
        # 验证预测掩码在有效范围内
        if outputs['pred_masks'].numel() > 0:
            self.assertTrue((outputs['pred_masks'] >= 0).all() and 
                          (outputs['pred_masks'] <= 1).all(),
                          "预测掩码超出[0,1]范围")
    
    def test_training_mode(self):
        """测试训练模式"""
        # 初始化模型
        model = MaskRCNN(num_classes=self.num_classes)
        model.train()  # 设置为训练模式
        
        # 使用模拟数据和目标测试前向传播
        outputs = model(self.mock_input, self.mock_targets)
        
        # 验证输出字典包含必要的键
        self.assertIn('pred_logits', outputs)
        self.assertIn('pred_boxes', outputs)
        self.assertIn('pred_masks', outputs)
        self.assertIn('rpn_logits', outputs)
        self.assertIn('rpn_boxes', outputs)
        self.assertIn('targets', outputs)
        
        # 验证输出是有效的张量
        self.assertTrue(torch.isfinite(outputs['pred_logits']).all(),
                      "预测logits包含无效值")
        self.assertTrue(torch.isfinite(outputs['pred_boxes']).all(),
                      "预测边界框包含无效值")
        self.assertTrue(torch.isfinite(outputs['pred_masks']).all(),
                      "预测掩码包含无效值")
    
    def test_predict_method(self):
        """测试predict方法"""
        # 初始化模型
        model = MaskRCNN(num_classes=self.num_classes)
        model.eval()  # 设置为评估模式
        
        # 使用模拟数据测试predict方法
        with torch.no_grad():
            predictions = model.predict([self.mock_input[0]])
        
        # 验证返回的是列表
        self.assertIsInstance(predictions, list)
        
        # 验证每个预测结果是字典
        for pred in predictions:
            self.assertIsInstance(pred, dict)
            self.assertIn('boxes', pred)
            self.assertIn('labels', pred)
            self.assertIn('scores', pred)
            self.assertIn('masks', pred)
    
    def test_different_backbones(self):
        """测试不同主干网络的表现"""
        backbones = ['resnet18']
        #'resnet50', 'vgg16', 'mobilenetv2'
        for backbone in backbones:
            # 初始化模型
            model = MaskRCNN(
                num_classes=self.num_classes,
                backbone_type=backbone
            )
            model.eval()  # 设置为评估模式
            
            # 使用模拟数据测试前向传播
            with torch.no_grad():
                outputs = model(self.mock_input)
            
            # 验证输出字典包含必要的键
            self.assertIn('pred_boxes', outputs)
            self.assertIn('pred_scores', outputs)
            self.assertIn('pred_labels', outputs)
            self.assertIn('pred_masks', outputs)
            
            # 验证输出是有效的张量
            if outputs['pred_scores'].numel() > 0:
                self.assertTrue(torch.isfinite(outputs['pred_scores']).all(),
                              f"{backbone}的预测分数包含无效值")
            
            if outputs['pred_masks'].numel() > 0:
                self.assertTrue(torch.isfinite(outputs['pred_masks']).all(),
                              f"{backbone}的预测掩码包含无效值")
    
    def test_input_formats(self):
        """测试不同输入格式"""
        # 初始化模型
        model = MaskRCNN(num_classes=self.num_classes)
        model.eval()  # 设置为评估模式
        
        # 测试单张图像输入（3D张量）
        single_image = torch.randn(3, *self.input_size)
        with torch.no_grad():
            outputs_single = model(single_image)
        
        # 验证输出
        self.assertIn('pred_boxes', outputs_single)
        self.assertIn('pred_scores', outputs_single)
        self.assertIn('pred_labels', outputs_single)
        self.assertIn('pred_masks', outputs_single)
        
        # 测试字典输入
        dict_input = {'image': self.mock_input}
        with torch.no_grad():
            outputs_dict = model(dict_input)
        
        # 验证输出
        self.assertIn('pred_boxes', outputs_dict)
        self.assertIn('pred_scores', outputs_dict)
        self.assertIn('pred_labels', outputs_dict)
        self.assertIn('pred_masks', outputs_dict)
        
        # 测试图像列表输入
        image_list = [self.mock_input[0], self.mock_input[1]]
        with torch.no_grad():
            outputs_list = model(image_list)
        
        # 验证输出
        self.assertIn('pred_boxes', outputs_list)
        self.assertIn('pred_scores', outputs_list)
        self.assertIn('pred_labels', outputs_list)
        self.assertIn('pred_masks', outputs_list)


if __name__ == '__main__':
    unittest.main() 