# MYSEGX 图像分割框架

<div align="center">
    <img src="MYSEGX.png" alt="MYSEGX Logo" width="600"/>
</div>

## 📌 简介

MYSEGX 是一个图像分割框架，专注于提供高效、易用的分割解决方案。框架支持多种分割模型，包括 DETR、UNet并提供完整的训练和评估流程。

## ✨ 特性

- 🚀 支持多种主流分割模型：
  - DETR（End-to-End Object Detection）
  - UNet（经典U型网络）
  - CNN（轻量级卷积网络）

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/MYSEGX.git
cd MYSEGX

# 安装依赖
pip install -r requirements.txt
```

## 📚 使用方法

### 1. 准备数据
将数据集组织为以下结构：
```
data/VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/
            └── Segmentation/
                ├── train.txt
                └── val.txt
```

### 2. 配置模型
在 `configs/models/` 目录下创建或修改模型配置文件：
```yaml
model:
  name: 'cnn'  # 或 'unet', 'detr'
  n_channels: 3
  n_classes: 21

train:
  batch_size: 4
  learning_rate: 0.001
  epochs: 100
```

### 3. 训练模型
```bash
python scripts/train_unified.py --config configs/models/detr/detr_r18.yaml
```

### 4. 评估和预测
> 🚧 正在开发中

## 📊 性能对比
> 🚧 正在测试中

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。