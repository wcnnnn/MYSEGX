"""测试训练模式的参数传递功能"""

from MYSEGX import train

def test_params_mode():
    """测试直接参数模式"""
    print("\n=== 测试直接参数模式 ===")
    history = train(
        # 模型参数
        model_name='detr',  
        num_classes=21,     # VOC数据集21个类别
        # 数据集参数
        dataset_root='data/VOC2012',
        dataset_split='train',
        image_size=(512, 512),
        batch_size=4,
        num_workers=2,
        # 训练参数
        learning_rate=1e-4,
        epochs=2,          # 为了快速测试，只训练2个epoch
        # 评价指标参数
        class_names=[
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ],
        metrics_dir='results/test_metrics',
        # 其他参数
        save_dir='results/test_unet',
        visualize=True,
        debug=True
    )
    return history

def test_config_mode():
    """测试配置文件模式"""
    print("\n=== 测试配置文件模式 ===")
    history = train(
        config_path='configs/models/unet/unet_voc.yaml',
        # 覆盖一些配置参数
        epochs=2,          # 为了快速测试，只训练2个epoch
        batch_size=4,
        save_dir='results/test_unet_config',
        metrics_dir='results/test_metrics_config',
        debug=True
    )
    return history

def test_mixed_mode():
    """测试混合模式（配置文件+参数）"""
    print("\n=== 测试混合模式 ===")
    history = train(
        # 基础配置
        config_path='configs/models/unet/unet_voc.yaml',
        # 覆盖模型参数
        num_classes=21,
        # 覆盖训练参数
        learning_rate=1e-4,
        epochs=2,          # 为了快速测试，只训练2个epoch
        batch_size=4,
        # 评价指标参数
        class_names=[
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ],
        metrics_dir='results/test_metrics_mixed',
        # 其他参数
        save_dir='results/test_unet_mixed',
        debug=True
    )
    return history

def main():
    """主函数"""
    print("开始测试训练模式...")
    
    # 测试直接参数模式
    params_history = test_params_mode()
    print("\n直接参数模式测试完成")
    print(f"训练历史: {params_history.keys()}")
    
    # 测试配置文件模式
    config_history = test_config_mode()
    print("\n配置文件模式测试完成")
    print(f"训练历史: {config_history.keys()}")
    
    # 测试混合模式
    mixed_history = test_mixed_mode()
    print("\n混合模式测试完成")
    print(f"训练历史: {mixed_history.keys()}")
    
    print("\n所有测试完成!")

if __name__ == '__main__':
    main()