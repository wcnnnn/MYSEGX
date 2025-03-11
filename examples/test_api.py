"""测试 MYSEGX 简单训练脚本"""
import os
from pathlib import Path
from MYSEGX import train

def test_semantic_segmentation():
    """测试语义分割训练"""
    print("\n=== 测试语义分割训练 ===")
    history = train(
        config_path='configs/Semantic_Segmentation/detr/detr_r18.yaml',
        task_type='semantic',
        dataset='voc',
        dataset_root='data/VOC2012',  # 请替换为实际的VOC数据集路径
        model_name='detr',
        batch_size=8,  # 小批次用于测试
        num_workers=8  # 禁用多进程加载用于测试
    )
    assert isinstance(history, dict)
    assert 'train_loss' in history
    print("语义分割训练测试通过！")

def test_instance_segmentation():
    """测试实例分割训练"""
    print("\n=== 测试实例分割训练 ===")
    history = train(
        config_path='configs/Semantic_Segmentation/detr/detr_r18.yaml',
        task_type='instance',
        dataset='voc',
        dataset_root='data/VOC2012',  # 请替换为实际的VOC数据集路径
        model_name='detr',
        batch_size=8,  # 小批次用于测试
        num_workers=8  # 禁用多进程加载用于测试
    )
    assert isinstance(history, dict)
    assert 'train_loss' in history
    print("实例分割训练测试通过！")



if __name__ == '__main__':
    # 设置数据集路径
    voc_root = os.getenv('VOC_ROOT', 'data/VOC2012')  # 从环境变量获取或使用默认值
    if not os.path.exists(voc_root):
        print(f"警告: VOC数据集路径 {voc_root} 不存在")
        print("请设置正确的VOC_ROOT环境变量或直接修改脚本中的路径")
        exit(1)
        
    # 运行所有测试
    try:
        test_semantic_segmentation()
        test_instance_segmentation()
        print("\n所有测试完成！")
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        raise
