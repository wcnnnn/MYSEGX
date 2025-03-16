"""测试 MYSEGX 简单训练脚本"""
import os
from pathlib import Path
from MYSEGX import train

def test_semantic_segmentation():
    """测试语义分割训练"""
    print("\n=== 测试语义分割训练 ===")
    
    # 获取数据集路径
    dataset_root = os.path.abspath(os.path.join('datasets', 'Cityspaces'))
    print(f"使用数据集路径: {dataset_root}")
    
    history = train(
        config_path='configs/Semantic_Segmentation/deeplabv3/deeplabv3_r18.yaml',
        task_type='semantic',
        dataset='cityscapes',
        dataset_root=dataset_root,  
        model_name='deeplabv3',
        num_classes=19, 
        batch_size=4,
        num_workers=8
    )
    assert isinstance(history, dict)
    assert 'train_loss' in history
    print("语义分割训练测试通过！")

def test_instance_segmentation():
    """测试实例分割训练"""
    print("\n=== 测试实例分割训练 ===")
    
    # 获取数据集路径
    dataset_root = os.path.abspath(os.path.join('datasets', 'VOC2012'))
    print(f"使用数据集路径: {dataset_root}")

    history = train(
        config_path='configs/Instance_Segmentation/yolact/yolact_r18.yaml',
        task_type='instance',
        dataset='cityscapes',
        dataset_root=dataset_root,
        model_name='yolact',
        num_classes=8, 
        batch_size=4,
        num_workers=8
    )
    assert isinstance(history, dict)
    assert 'train_loss' in history
    print("实例分割训练测试通过！")

if __name__ == '__main__':
    # 设置数据集路径
    voc_root = os.path.abspath(os.path.join('datasets', 'VOC2012'))
    if not os.path.exists(voc_root):
        print(f"警告: VOC数据集路径 {voc_root} 不存在")
        print("请确保数据集位于 datasets/VOC2012 目录下")
        exit(1)
        
    # 运行所有测试
    try:
        test_instance_segmentation()
        print("\n所有测试完成！")
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        raise
