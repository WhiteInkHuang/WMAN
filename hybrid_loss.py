"""
混合损失函数 - 结合多种损失以提升性能
"""
import torch
import torch.nn as nn
from losses_detection_2d import CIoULoss, bbox_iou_2d


class HybridDetectionLoss(nn.Module):
    """
    混合检测损失
    = CIoU Loss + Focal L1 Loss + Aspect Ratio Loss
    """
    def __init__(self, ciou_weight=3.0, l1_weight=1.0, aspect_weight=0.5):
        super(HybridDetectionLoss, self).__init__()
        self.ciou_weight = ciou_weight
        self.l1_weight = l1_weight
        self.aspect_weight = aspect_weight
        self.ciou_loss = CIoULoss()
    
    def forward(self, pred_bbox, target_bbox):
        """
        pred_bbox: [B, 4]
        target_bbox: list of [N_i, 4] 或 [B, 4]
        """
        # 统一格式
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]
        
        # 1. CIoU Loss（主要损失）
        ciou_loss = self.ciou_loss(pred_bbox, target_bbox)
        
        # 2. Focal L1 Loss（对难样本加权）
        l1_diff = torch.abs(pred_bbox - target_bbox)
        iou = bbox_iou_2d(pred_bbox, target_bbox)
        focal_weight = (1 - iou).unsqueeze(1)  # IoU越低，权重越高
        focal_l1_loss = (focal_weight * l1_diff).mean()
        
        # 3. Aspect Ratio Loss（保持宽高比）
        pred_w = pred_bbox[:, 2] - pred_bbox[:, 0]
        pred_h = pred_bbox[:, 3] - pred_bbox[:, 1]
        target_w = target_bbox[:, 2] - target_bbox[:, 0]
        target_h = target_bbox[:, 3] - target_bbox[:, 1]
        
        pred_aspect = pred_w / (pred_h + 1e-6)
        target_aspect = target_w / (target_h + 1e-6)
        aspect_loss = torch.abs(pred_aspect - target_aspect).mean()
        
        # 总损失
        total_loss = (self.ciou_weight * ciou_loss + 
                     self.l1_weight * focal_l1_loss + 
                     self.aspect_weight * aspect_loss)
        
        return total_loss


class AdaptiveDetectionLoss(nn.Module):
    """
    自适应检测损失 - 根据训练阶段动态调整权重
    """
    def __init__(self):
        super(AdaptiveDetectionLoss, self).__init__()
        self.ciou_loss = CIoULoss()
        self.epoch = 0
    
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.epoch = epoch
    
    def forward(self, pred_bbox, target_bbox):
        """
        pred_bbox: [B, 4]
        target_bbox: list of [N_i, 4] 或 [B, 4]
        """
        # 统一格式
        if isinstance(target_bbox, list):
            target_bbox = torch.stack([bbox[0] for bbox in target_bbox])
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, 0, :]
        
        # CIoU Loss
        ciou_loss = self.ciou_loss(pred_bbox, target_bbox)
        
        # L1 Loss
        l1_loss = torch.abs(pred_bbox - target_bbox).mean()
        
        # 动态权重：早期更关注L1（粗定位），后期更关注CIoU（精细调整）
        if self.epoch < 50:
            # 早期：L1权重高
            ciou_weight = 2.0
            l1_weight = 1.5
        elif self.epoch < 100:
            # 中期：平衡
            ciou_weight = 3.0
            l1_weight = 1.0
        else:
            # 后期：CIoU权重高
            ciou_weight = 4.0
            l1_weight = 0.5
        
        total_loss = ciou_weight * ciou_loss + l1_weight * l1_loss
        
        return total_loss


# 测试
if __name__ == "__main__":
    print("测试混合损失函数")
    
    pred = torch.rand(4, 4)
    target = torch.rand(4, 4)
    
    # 确保格式正确
    pred[:, 2:] = pred[:, :2] + torch.abs(pred[:, 2:] - pred[:, :2])
    target[:, 2:] = target[:, :2] + torch.abs(target[:, 2:] - target[:, :2])
    
    # 测试混合损失
    hybrid_loss = HybridDetectionLoss()
    loss = hybrid_loss(pred, target)
    print(f"Hybrid Loss: {loss:.4f}")
    
    # 测试自适应损失
    adaptive_loss = AdaptiveDetectionLoss()
    adaptive_loss.set_epoch(10)
    loss = adaptive_loss(pred, target)
    print(f"Adaptive Loss (epoch 10): {loss:.4f}")
    
    adaptive_loss.set_epoch(150)
    loss = adaptive_loss(pred, target)
    print(f"Adaptive Loss (epoch 150): {loss:.4f}")
    
    print("✓ 测试通过!")
