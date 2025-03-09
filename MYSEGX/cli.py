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
    train_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    train_parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    # 评估模式
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    eval_parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    
    # 预测模式
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    predict_parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    predict_parser.add_argument('--source', type=str, required=True, help='输入图像或目录路径')
    predict_parser.add_argument('--save-dir', type=str, default='results', help='保存结果的目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'train':
        from MYSEGX.engine.trainer import train
        train(args.config, args.resume)
    elif args.mode == 'eval':
        from MYSEGX.engine.evaluator import evaluate
        evaluate(args.config, args.weights)
    elif args.mode == 'predict':
        from MYSEGX.engine.predictor import predict
        predict(args.config, args.weights, args.source, args.save_dir)
    else:
        print('请指定运行模式: train, eval, 或 predict')
        sys.exit(1)

if __name__ == '__main__':
    main()
