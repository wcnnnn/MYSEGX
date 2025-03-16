"""NMS操作模块"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def box_iou(box1, box2):
    """计算两组边界框之间的IoU
    
    参数:
        box1 (Tensor): 第一组边界框 [N, 4]
        box2 (Tensor): 第二组边界框 [M, 4]
        
    返回:
        iou (Tensor): IoU矩阵 [N, M]
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]
    
    # 计算交集
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # 计算IoU
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou

def nms(boxes, scores, iou_threshold):
    """传统NMS
    
    参数:
        boxes (Tensor): 边界框 [N, 4]
        scores (Tensor): 置信度分数 [N]
        iou_threshold (float): IoU阈值
        
    返回:
        keep (Tensor): 保留的边界框索引
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    # 按分数降序排序
    scores, idx = scores.sort(descending=True)
    boxes = boxes[idx]
    
    # 计算IoU矩阵
    iou = box_iou(boxes, boxes)
    
    keep = []
    while idx.numel() > 0:
        if idx.numel() == 1:
            keep.append(idx[0])
            break
        else:
            # 保留分数最高的框
            keep.append(idx[0])
            
            # 计算其他框与当前框的IoU
            other_boxes = idx[1:]
            ious = iou[0, 1:]
            
            # 过滤掉IoU大于阈值的框
            idx = other_boxes[ious <= iou_threshold]
            if idx.numel() == 0:
                break
            iou = iou[1:, 1:][ious <= iou_threshold][:, ious <= iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def fast_nms(boxes, scores, iou_threshold, top_k, score_threshold=0.05):
    """Fast NMS - 并行处理所有类别
    
    参数:
        boxes (Tensor): 边界框 [N, num_classes, 4]
        scores (Tensor): 置信度分数 [N, num_classes]
        iou_threshold (float): IoU阈值
        top_k (int): 每个类别保留的最大检测数
        score_threshold (float): 分数阈值
        
    返回:
        keep (Tensor): 保留的边界框索引 [N']
        scores (Tensor): 保留的分数 [N']
        classes (Tensor): 对应的类别 [N']
    """
    scores = scores.clone()
    
    # 过滤低分数的检测
    scores[scores < score_threshold] = 0
    
    # 获取每个anchor box的最高分数和对应类别
    num_classes = scores.shape[1]
    max_scores, classes = scores.max(dim=1)
    
    # 按分数降序排序
    _, idx = max_scores.sort(descending=True)
    idx = idx[:top_k]
    
    boxes = boxes[idx]
    max_scores = max_scores[idx]
    classes = classes[idx]
    
    # 计算IoU矩阵
    iou = box_iou(boxes, boxes)
    
    # 创建保留掩码
    keep = torch.zeros(len(idx), dtype=torch.bool, device=boxes.device)
    
    for i in range(len(idx)):
        if not keep[i]:
            continue
            
        # 获取当前框的类别
        cls = classes[i]
        
        # 找到相同类别的其他框
        same_class = classes == cls
        
        # 计算IoU
        ious = iou[i, same_class]
        
        # 过滤掉IoU大于阈值的框
        keep[same_class] = ious <= iou_threshold
    
    idx = idx[keep]
    return idx, max_scores[keep], classes[keep]

class FastNMSModule(torch.nn.Module):
    """Fast NMS模块
    
    参数:
        iou_threshold (float): IoU阈值
        top_k (int): 每个类别保留的最大检测数
        score_threshold (float): 分数阈值
    """
    def __init__(self, iou_threshold=0.5, top_k=200, score_threshold=0.05):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.score_threshold = score_threshold
        
    def forward(self, boxes, scores):
        """前向传播"""
        return fast_nms(boxes, scores, self.iou_threshold, self.top_k, self.score_threshold)
