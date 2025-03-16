<div align="center">
    <img src="LOGO.jpg" alt="MYSEGX Logo" width="600"/>
    <p>
        <em>🚀 A simple, efficient, and easy-to-use image segmentation framework</em>
    </p>
    <p>
        <a href="LICENSE">
            <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
        </a>
        <img alt="Python" src="https://img.shields.io/badge/python-3.7%2B-blue"/>
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.7%2B-orange"/>
    </p>
    <p>
        <a href="README_zh_CN.md">中文</a> | <strong>English</strong>
    </p>
</div>

## 📌 Introduction

<div align="center">
    MYSEGX is an image segmentation framework focused on providing efficient and user-friendly segmentation solutions. The framework supports various segmentation models.
</div>

### 🎯 Segmentation Tasks and Reference Training Scripts

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
          <li>In development</li>
        </ul>
      </td>
      <td>
        <ul>
        <li>In development</li>
        </ul>
      </td>
  </tbody>
</table>

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/wcnnnn/MYSEGX.git
cd MYSEGX

# Install dependencies
pip install -r requirements.txt

# Install MYSEGX
pip install -e .
```

## 📚 User Guide

### 1. Prepare Data

This project currently supports the [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset. Ensure your dataset directory structure is as follows:

```
VOC2012/
    ├── JPEGImages/
    ├── SegmentationClass/
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

For more details, refer to the [Dataset Documentation](docs/dataset.md).

## 📊 Performance Comparison
> 🚧 Under testing

## 🤝 Contributing

Feel free to submit Issues and Pull Requests!

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.