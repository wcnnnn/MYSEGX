# 数据集

本项目目前支持 [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 数据集。VOC2012 是一个广泛使用的图像分割和目标检测数据集，包含多种类别的标注。

## VOC2012

[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 数据集。VOC2012 是一个广泛使用的图像分割和目标检测数据集，包含多种类别的标注。请确保您的数据集目录结构如下：

```
VOC2012/
    ├── JPEGImages/
    ├── SegmentationClass/
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

- `JPEGImages/`：存放原始图像。
- `SegmentationClass/`：存放分割标注图像。
- `ImageSets/Segmentation/`：包含训练和验证集的文件列表。

确保数据集按照上述结构组织，以便于模型的训练和评估。