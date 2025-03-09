"""测试 MYSEGX 简单训练脚本"""
from MYSEGX import train

if __name__ == '__main__':
    config_path = 'configs/models/detr/detr_r18.yaml'
    
    print("开始训练...")
    history = train(config_path)
    print("训练完成！")
