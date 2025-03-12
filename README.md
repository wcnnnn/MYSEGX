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

<table width="100%">
<tr>
<td align="center" width="25%">
<b>Semantic<br/>Segmentation</b><br/>
</td>
<td align="center" width="25%">
<b>Panoptic<br/>Segmentation</b><br/>
🚧 开发中
</td>
<td align="center" width="25%">
<b>Instance<br/>Segmentation</b><br/>
🚧 开发中
</td>
<td align="center" width="25%">
<b>3D<br/>Segmentation</b><br/>
🚧 开发中
</td>
</tr>
<tr>
<td align="center" width="25%">
<ul style="text-align: left;">
<li><a href="docs/Semantic_Segmentation/detr.md">DETR</a></li>
<li>UNet</li>
</ul>
</td>
<td align="center" width="25%">
<ul style="text-align: left;">
<li>开发中</li>
</ul>
</td>
<td align="center" width="25%">
<ul style="text-align: left;">
<li>开发中</li>
</ul>
</td>
<td align="center" width="25%">
<ul style="text-align: left;">
<li>开发中</li>
</ul>
</td>
</tr>
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