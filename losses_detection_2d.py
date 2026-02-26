"""
2D检测任务的损失函数和评估指标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BBoxLoss(nn.Module):
    """
    边界框回归损失（2D版本）
    结合 Smooth L1 Loss 和 IoU Loss
    """
    def __init__(self, use_iou_loss=True, iou_weight=2.0, l1_weight=0.5):
        super(BBoxLoss, self).__init__()
        self.use_iou_loss = use_iou_loss
        self.iou_weight = iou_weight
        self.l1_weight = l1_weight
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, pred_bbox, target_bbox):
        """
        pred_bbox: [B, 4] - 预测的边界框
        target_bbox: list of [N_i, 4] - 真实的边界框（每个样本可能有多个）
        """
        # 如果target是list，取每个样本的第一个框
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])  # [B, 4]
        
        # 如果target有多个框，取第一个
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]  # [B, 4]
        
        # Smooth L1 Loss（坐标回归）
        l1_loss = self.smooth_l1(pred_bbox, target_bbox)
        
        total_loss = self.l1_weight * l1_loss
        
        # IoU Loss
        if self.use_iou_loss:
            iou_loss = 1.0 - bbox_iou_2d(pred_bbox, target_bbox).mean()
            total_loss += self.iou_weight * iou_loss
        
        return total_loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss（2D版本）
    更好地处理不重叠的边界框
    """
    def __init__(self):
        super(GIoULoss, self).__init__()
    
    def forward(self, pred_bbox, target_bbox):
        """
        pred_bbox: [B, 4]
        target_bbox: list of [N_i, 4] 或 [B, N, 4] 或 [B, 4]
        """
        # 如果target是list，取每个样本的第一个框
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
        
        # 如果target有多个框，取第一个
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]
        
        giou = bbox_giou_2d(pred_bbox, target_bbox)
        loss = 1.0 - giou.mean()
        return loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss（2D版本）
    考虑中心点距离和宽高比，更适合小目标
    """
    def __init__(self):
        super(CIoULoss, self).__init__()
    
    def forward(self, pred_bbox, target_bbox):
        """
        pred_bbox: [B, 4]
        target_bbox: list of [N_i, 4] 或 [B, 4]
        """
        # 如果target是list，取每个样本的第一个框
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
        
        # 如果target有多个框，取第一个
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]
        
        ciou = bbox_ciou_2d(pred_bbox, target_bbox)
        loss = 1.0 - ciou.mean()
        return loss


def bbox_iou_2d(bbox1, bbox2):
    """
    计算2D边界框的IoU
    
    bbox: [B, 4] - [x_min, y_min, x_max, y_max]
    返回: [B] - 每个样本的IoU
    """
    # 确保是2D tensor
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
    """
    计算2D边界框的GIoU (Generalized IoU)
    
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
    """
    计算检测任务的评估指标（2D版本）
    
    返回: dict with IoU, GIoU, etc.
    """
    # 如果target是list，取每个样本的第一个框
    if isinstance(target_bbox, list):
        target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
    
    # 如果target有多个框，取第一个
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


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试2D检测损失函数和评估指标")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    pred_bbox = torch.rand(batch_size, 4)  # 随机预测
    target_bbox = torch.rand(batch_size, 4)  # 随机真实值
    
    # 确保边界框格式正确 (min < max)
    pred_bbox[:, 2:] = pred_bbox[:, :2] + torch.abs(pred_bbox[:, 2:] - pred_bbox[:, :2])
    target_bbox[:, 2:] = target_bbox[:, :2] + torch.abs(target_bbox[:, 2:] - target_bbox[:, :2])
    
    print(f"\n预测边界框形状: {pred_bbox.shape}")
    print(f"真实边界框形状: {target_bbox.shape}")
    print(f"\n第一个样本:")
    print(f"  预测: {pred_bbox[0]}")
    print(f"  真实: {target_bbox[0]}")
    
    # 测试IoU
    print("\n" + "=" * 60)
    print("测试IoU计算")
    print("=" * 60)
    iou = bbox_iou_2d(pred_bbox, target_bbox)
    print(f"IoU: {iou}")
    print(f"平均IoU: {iou.mean():.4f}")
    
    # 测试GIoU
    print("\n" + "=" * 60)
    print("测试GIoU计算")
    print("=" * 60)
    giou = bbox_giou_2d(pred_bbox, target_bbox)
    print(f"GIoU: {giou}")
    print(f"平均GIoU: {giou.mean():.4f}")
    
    # 测试损失函数
    print("\n" + "=" * 60)
    print("测试损失函数")
    print("=" * 60)
    
    bbox_loss = BBoxLoss(use_iou_loss=True, iou_weight=1.0, l1_weight=1.0)
    loss = bbox_loss(pred_bbox, target_bbox)
    print(f"BBox Loss: {loss:.4f}")
    
    giou_loss_fn = GIoULoss()
    giou_loss = giou_loss_fn(pred_bbox, target_bbox)
    print(f"GIoU Loss: {giou_loss:.4f}")
    
    # 测试评估指标
    print("\n" + "=" * 60)
    print("测试评估指标")
    print("=" * 60)
    metrics = compute_detection_metrics(pred_bbox, target_bbox)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    print("\n✓ 所有测试通过!")
