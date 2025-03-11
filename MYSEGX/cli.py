"""命令行接口模块"""
import argparse
import sys
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MYSEGX: 现代化图像分割框架')
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练模型')
    # 基本配置
    train_parser.add_argument('--config', type=str, help='配置文件路径')
    train_parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    # 模型参数
    train_parser.add_argument('--model', type=str, default='detr', help='模型类型 (detr, unet, cnn)')
    train_parser.add_argument('--num-classes', type=int, help='类别数量')
    train_parser.add_argument('--backbone', type=str, help='主干网络类型')
    # 数据集参数
    train_parser.add_argument('--data-root', type=str, help='数据集根目录')
    train_parser.add_argument('--dataset', type=str, help='数据集类型')
    train_parser.add_argument('--image-size', type=int, nargs=2, help='输入图像尺寸 (高度 宽度)')
    train_parser.add_argument('--batch-size', type=int, help='批次大小')
    train_parser.add_argument('--num-workers', type=int, help='数据加载线程数')
    # 训练参数
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--lr', type=float, help='学习率')
    train_parser.add_argument('--weight-decay', type=float, help='权重衰减')
    train_parser.add_argument('--lr-drop', type=int, help='学习率衰减轮数')
    # 设备参数
    train_parser.add_argument('--device', type=str, help='训练设备 (cuda 或 cpu)')
    train_parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    train_parser.add_argument('--amp', action='store_true', help='是否使用混合精度训练')
    # 日志和保存参数
    train_parser.add_argument('--save-dir', type=str, help='保存目录')
    train_parser.add_argument('--log-dir', type=str, help='日志目录')
    train_parser.add_argument('--eval-interval', type=int, default=1, help='评估间隔')
    train_parser.add_argument('--save-interval', type=int, default=1, help='保存间隔')
    train_parser.add_argument('--no-visualize', action='store_true', help='禁用可视化')
    train_parser.add_argument('--seed', type=int, help='随机种子')
    train_parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    # 评价指标参数
    train_parser.add_argument('--class-names', type=str, nargs='+', help='类别名称列表')
    train_parser.add_argument('--metrics-dir', type=str, help='指标保存目录，默认为save-dir下的metrics子目录')
    
    # 评估模式
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    eval_parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    eval_parser.add_argument('--device', type=str, help='评估设备 (cuda 或 cpu)')
    eval_parser.add_argument('--batch-size', type=int, help='批次大小')
    eval_parser.add_argument('--save-dir', type=str, help='结果保存目录')
    eval_parser.add_argument('--class-names', type=str, nargs='+', help='类别名称列表')
    eval_parser.add_argument('--metrics-dir', type=str, help='指标保存目录')
    
    # 预测模式
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    predict_parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    predict_parser.add_argument('--source', type=str, required=True, help='输入图像或目录路径')
    predict_parser.add_argument('--save-dir', type=str, default='results', help='保存结果的目录')
    predict_parser.add_argument('--device', type=str, help='预测设备 (cuda 或 cpu)')
    predict_parser.add_argument('--batch-size', type=int, help='批次大小')
    predict_parser.add_argument('--conf-thres', type=float, default=0.7, help='置信度阈值')
    predict_parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    predict_parser.add_argument('--class-names', type=str, nargs='+', help='类别名称列表')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'train':
        from MYSEGX.api import train
        # 构建训练参数字典
        train_kwargs = {
            'config_path': args.config,
            'resume_path': args.resume,
            # 模型参数
            'model_name': args.model,
            'num_classes': args.num_classes,
            'backbone_type': args.backbone,
            # 数据集参数
            'dataset_root': args.data_root,
            'dataset_split': args.dataset,
            'image_size': tuple(args.image_size) if args.image_size else None,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            # 训练参数
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'lr_drop': args.lr_drop,
            # 设备参数
            'device': args.device,
            'distributed': args.distributed,
            'amp': args.amp,
            # 日志和保存参数
            'save_dir': args.save_dir,
            'log_dir': args.log_dir,
            'eval_interval': args.eval_interval,
            'save_interval': args.save_interval,
            'visualize': not args.no_visualize,
            'seed': args.seed,
            'debug': args.debug,
            # 评价指标参数
            'class_names': args.class_names,
            'metrics_dir': args.metrics_dir
        }
        # 移除None值参数
        train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
        train(**train_kwargs)
        
    elif args.mode == 'eval':
        from MYSEGX.engine.evaluator import evaluate
        eval_kwargs = {
            'config': args.config,
            'weights': args.weights,
            'device': args.device,
            'batch_size': args.batch_size,
            'save_dir': args.save_dir,
            'class_names': args.class_names,
            'metrics_dir': args.metrics_dir
        }
        eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
        evaluate(**eval_kwargs)
        
    elif args.mode == 'predict':
        from MYSEGX.engine.predictor import predict
        predict_kwargs = {
            'config': args.config,
            'weights': args.weights,
            'source': args.source,
            'save_dir': args.save_dir,
            'device': args.device,
            'batch_size': args.batch_size,
            'conf_thres': args.conf_thres,
            'visualize': args.visualize,
            'class_names': args.class_names
        }
        predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}
        predict(**predict_kwargs)
        
    else:
        print('请指定运行模式: train, eval, 或 predict')
        sys.exit(1)

if __name__ == '__main__':
    main()
