<div align="center">
    <img src="LOGO.jpg" alt="MYSEGX Logo" width="600"/>
    <p>
        <em>🚀 简单、高效、易用的图像分割框架</em>
    </p>
    <p>
        <a href="LICENSE">
            <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
        </a>
        <img alt="Python" src="https://img.shields.io/badge/python-3.7%2B-blue"/>
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.7%2B-orange"/>
    </p>
</div>

## 📌 简介

MYSEGX 是一个图像分割框架，专注于提供高效、易用的分割解决方案。框架支持多种分割模型。

### 🎯 分割任务以及参考训练脚本

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Semantic Segmentation</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Panoptic Segmentation</b>
      </td>
      <td>
        <b>3D Segmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="docs/Semantic_Segmentation/detr.md">DETR (ECCV'2020)</a></li>
        <li><a href="docs/Semantic_Segmentation/unet.md">UNet (MICCAI'2015)</a></li>
        <li><a href="docs/Semantic_Segmentation/deeplabv3.md">DeepLabV3 (ArXiv'2017)</a></li>
        <li><a href="docs/Semantic_Segmentation/deeplabv3plus.md">DeepLabV3+ (ECCV'2018)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/Instance_Segmentation/detr.md">DETR (ECCV'2020)</a></li>
          <li><a href="docs/Instance_Segmentation/yolact.md">YOLACT (ICCV'2019)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>开发中</li>
        </ul>
      </td>
      <td>
        <ul>
        <li>开发中</li>
        </ul>
      </td>
  </tbody>
</table>

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/wcnnnn/MYSEGX.git
cd MYSEGX

# 安装依赖
pip install -r requirements.txt

# 安装 MYSEGX
pip install -e .
```

## 📚 使用指南

### 1. 准备数据

本项目目前支持 [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 数据集。请确保您的数据集目录结构如下：

```
VOC2012/
    ├── JPEGImages/
    ├── SegmentationClass/
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

更多详情请参阅 [数据集文档](docs/dataset.md)。

## 📊 性能对比
> 🚧 正在测试中

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
