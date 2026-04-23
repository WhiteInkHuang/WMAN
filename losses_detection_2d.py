import torch
import torch.nn as nn
import torch.nn.functional as F


class BBoxLoss(nn.Module):
    
    def __init__(self, use_iou_loss=True, iou_weight=2.0, l1_weight=0.5):
        super(BBoxLoss, self).__init__()
        self.use_iou_loss = use_iou_loss
        self.iou_weight = iou_weight
        self.l1_weight = l1_weight
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, pred_bbox, target_bbox):
       
        
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])  # [B, 4]
        
       
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]  # [B, 4]
        
        
        l1_loss = self.smooth_l1(pred_bbox, target_bbox)
        
        total_loss = self.l1_weight * l1_loss
        
        # IoU Loss
        if self.use_iou_loss:
            iou_loss = 1.0 - bbox_iou_2d(pred_bbox, target_bbox).mean()
            total_loss += self.iou_weight * iou_loss
        
        return total_loss


class GIoULoss(nn.Module):
   
    def __init__(self):
        super(GIoULoss, self).__init__()
    
    def forward(self, pred_bbox, target_bbox):
       
       
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
        
       
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]
        
        giou = bbox_giou_2d(pred_bbox, target_bbox)
        loss = 1.0 - giou.mean()
        return loss


class CIoULoss(nn.Module):
    
    def __init__(self):
        super(CIoULoss, self).__init__()
    
    def forward(self, pred_bbox, target_bbox):
       
        
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
        
        
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]
        
        ciou = bbox_ciou_2d(pred_bbox, target_bbox)
        loss = 1.0 - ciou.mean()
        return loss


def bbox_iou_2d(bbox1, bbox2):
    
   
    if bbox1.dim() == 1:
        bbox1 = bbox1.unsqueeze(0)
    if bbox2.dim() == 1:
        bbox2 = bbox2.unsqueeze(0)
    
    # 提取坐标
    x1_min, y1_min = bbox1[:, 0], bbox1[:, 1]
    x1_max, y1_max = bbox1[:, 2], bbox1[:, 3]
    
    x2_min, y2_min = bbox2[:, 0], bbox2[:, 1]
    x2_max, y2_max = bbox2[:, 2], bbox2[:, 3]
    
    # 计算交集
    inter_x_min = torch.max(x1_min, x2_min)
    inter_y_min = torch.max(y1_min, y2_min)
    
    inter_x_max = torch.min(x1_max, x2_max)
    inter_y_max = torch.min(y1_max, y2_max)
    
    # 交集面积
    inter_w = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_h = torch.clamp(inter_y_max - inter_y_min, min=0)
    
    inter_area = inter_w * inter_h
    
    # 各自面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 并集面积
    union_area = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def bbox_giou_2d(bbox1, bbox2):
    
   
    if bbox1.dim() == 1:
        bbox1 = bbox1.unsqueeze(0)
    if bbox2.dim() == 1:
        bbox2 = bbox2.unsqueeze(0)
    
    # 计算IoU
    iou = bbox_iou_2d(bbox1, bbox2)
    
    # 提取坐标
    x1_min, y1_min = bbox1[:, 0], bbox1[:, 1]
    x1_max, y1_max = bbox1[:, 2], bbox1[:, 3]
    
    x2_min, y2_min = bbox2[:, 0], bbox2[:, 1]
    x2_max, y2_max = bbox2[:, 2], bbox2[:, 3]
    
    # 计算最小包围框
    c_x_min = torch.min(x1_min, x2_min)
    c_y_min = torch.min(y1_min, y2_min)
    
    c_x_max = torch.max(x1_max, x2_max)
    c_y_max = torch.max(y1_max, y2_max)
    
    # 包围框面积
    c_area = (c_x_max - c_x_min) * (c_y_max - c_y_min)
    
    # 各自面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 并集面积
    inter_area = iou * (area1 + area2) / (1 + iou + 1e-6)
    union_area = area1 + area2 - inter_area
    
    # GIoU
    giou = iou - (c_area - union_area) / (c_area + 1e-6)
    
    return giou


def bbox_ciou_2d(bbox1, bbox2):
    """
    计算2D边界框的CIoU (Complete IoU)
    CIoU = IoU - (ρ²(b, b_gt) / c²) - αv
    其中 ρ 是中心点距离，c 是对角线距离，v 是宽高比一致性
    
    bbox: [B, 4]
    返回: [B]
    """
    # 确保是2D tensor
    if bbox1.dim() == 1:
        bbox1 = bbox1.unsqueeze(0)
    if bbox2.dim() == 1:
        bbox2 = bbox2.unsqueeze(0)
    
    # 计算IoU
    iou = bbox_iou_2d(bbox1, bbox2)
    
    # 提取坐标
    x1_min, y1_min = bbox1[:, 0], bbox1[:, 1]
    x1_max, y1_max = bbox1[:, 2], bbox1[:, 3]
    
    x2_min, y2_min = bbox2[:, 0], bbox2[:, 1]
    x2_max, y2_max = bbox2[:, 2], bbox2[:, 3]
    
    # 计算中心点
    x1_center = (x1_min + x1_max) / 2
    y1_center = (y1_min + y1_max) / 2
    x2_center = (x2_min + x2_max) / 2
    y2_center = (y2_min + y2_max) / 2
    
    # 中心点距离的平方
    center_dist_sq = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
    
    # 最小包围框
    c_x_min = torch.min(x1_min, x2_min)
    c_y_min = torch.min(y1_min, y2_min)
    c_x_max = torch.max(x1_max, x2_max)
    c_y_max = torch.max(y1_max, y2_max)
    
    # 对角线距离的平方
    c_diag_sq = (c_x_max - c_x_min) ** 2 + (c_y_max - c_y_min) ** 2
    
    # 宽高
    w1 = x1_max - x1_min
    h1 = y1_max - y1_min
    w2 = x2_max - x2_min
    h2 = y2_max - y2_min
    
    # 宽高比一致性
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6)), 2)
    
    # alpha 参数
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    
    # CIoU
    ciou = iou - (center_dist_sq / (c_diag_sq + 1e-6)) - alpha * v
    
    return ciou


def compute_detection_metrics(pred_bbox, target_bbox):
    
    
    if isinstance(target_bbox, list):
        target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
    
   
    if target_bbox.dim() == 3:
        target_bbox = target_bbox[:, 0, :]
    
    iou = bbox_iou_2d(pred_bbox, target_bbox)
    giou = bbox_giou_2d(pred_bbox, target_bbox)
    
    # 计算中心点距离
    pred_center = (pred_bbox[:, :2] + pred_bbox[:, 2:]) / 2
    target_center = (target_bbox[:, :2] + target_bbox[:, 2:]) / 2
    center_dist = torch.norm(pred_center - target_center, dim=1)
    
    metrics = {
        'iou': iou.mean().item(),
        'giou': giou.mean().item(),
        'center_dist': center_dist.mean().item()
    }
    
    return metrics

