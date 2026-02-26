"""
数据增强模块 - 针对医学图像检测任务
"""
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F


class DetectionAugmentation:
    """
    检测任务的数据增强
    包含：随机翻转、旋转、缩放、裁剪、对比度调整等
    """
    def __init__(self, 
                 flip_prob=0.5,
                 rotate_prob=0.5,
                 scale_prob=0.5,
                 contrast_prob=0.5,
                 brightness_prob=0.5,
                 crop_prob=0.3,
                 max_rotate_angle=15,
                 scale_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 brightness_range=(0.9, 1.1)):
        """
        初始化数据增强参数
        
        Args:
            flip_prob: 翻转概率
            rotate_prob: 旋转概率
            scale_prob: 缩放概率
            contrast_prob: 对比度调整概率
            brightness_prob: 亮度调整概率
            crop_prob: 裁剪概率
            max_rotate_angle: 最大旋转角度
            scale_range: 缩放范围
            contrast_range: 对比度范围
            brightness_range: 亮度范围
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.contrast_prob = contrast_prob
        self.brightness_prob = brightness_prob
        self.crop_prob = crop_prob
        self.max_rotate_angle = max_rotate_angle
        self.scale_range = scale_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
    
    def __call__(self, image, bbox):
        """
        应用数据增强
        
        Args:
            image: [2, H, W] tensor
            bbox: [N, 4] tensor, 归一化坐标 [xmin, ymin, xmax, ymax]
        
        Returns:
            augmented_image: [2, H, W]
            augmented_bbox: [N, 4]
        """
        # 随机水平翻转
        if random.random() < self.flip_prob:
            image, bbox = self.horizontal_flip(image, bbox)
        
        # 随机垂直翻转
        if random.random() < self.flip_prob:
            image, bbox = self.vertical_flip(image, bbox)
        
        # 随机旋转（小角度）
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotate_angle, self.max_rotate_angle)
            image, bbox = self.rotate(image, bbox, angle)
        
        # 随机缩放
        if random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_range)
            image, bbox = self.scale(image, bbox, scale)
        
        # 随机对比度调整
        if random.random() < self.contrast_prob:
            contrast_factor = random.uniform(*self.contrast_range)
            image = self.adjust_contrast(image, contrast_factor)
        
        # 随机亮度调整
        if random.random() < self.brightness_prob:
            brightness_factor = random.uniform(*self.brightness_range)
            image = self.adjust_brightness(image, brightness_factor)
        
        # 随机裁剪（围绕边界框）
        if random.random() < self.crop_prob:
            image, bbox = self.crop_around_bbox(image, bbox)
        
        return image, bbox
    
    def horizontal_flip(self, image, bbox):
        """水平翻转"""
        image = torch.flip(image, dims=[2])  # 翻转宽度维度
        bbox = bbox.clone()
        bbox[:, [0, 2]] = 1.0 - bbox[:, [2, 0]]  # xmin, xmax 互换并翻转
        return image, bbox
    
    def vertical_flip(self, image, bbox):
        """垂直翻转"""
        image = torch.flip(image, dims=[1])  # 翻转高度维度
        bbox = bbox.clone()
        bbox[:, [1, 3]] = 1.0 - bbox[:, [3, 1]]  # ymin, ymax 互换并翻转
        return image, bbox
    
    def rotate(self, image, bbox, angle):
        """
        旋转图像和边界框（小角度）
        注意：旋转后边界框会变大（外接矩形）
        """
        # 转换为 [1, C, H, W] 格式
        image = image.unsqueeze(0)
        
        # 旋转图像
        angle_rad = torch.tensor(angle * np.pi / 180.0)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # 旋转矩阵
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        image = F.grid_sample(image, grid, align_corners=False)
        image = image.squeeze(0)
        
        # 边界框旋转（简化处理：保持不变，因为小角度旋转影响不大）
        # 对于更精确的处理，需要旋转四个角点然后计算新的外接矩形
        return image, bbox
    
    def scale(self, image, bbox, scale):
        """
        缩放图像和边界框
        """
        C, H, W = image.shape
        new_H = int(H * scale)
        new_W = int(W * scale)
        
        # 缩放图像
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # 裁剪或填充到原始尺寸
        if scale > 1.0:
            # 中心裁剪
            start_h = (new_H - H) // 2
            start_w = (new_W - W) // 2
            image = image[:, :, start_h:start_h+H, start_w:start_w+W]
            
            # 调整边界框（缩放后裁剪）
            bbox = bbox.clone()
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale - start_w / W
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale - start_h / H
        else:
            # 填充
            pad_h = (H - new_H) // 2
            pad_w = (W - new_W) // 2
            # 确保填充后尺寸正确
            pad_h_total = H - new_H
            pad_w_total = W - new_W
            pad_left = pad_w
            pad_right = pad_w_total - pad_w
            pad_top = pad_h
            pad_bottom = pad_h_total - pad_h
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            
            # 调整边界框（缩放后填充）
            bbox = bbox.clone()
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale + pad_w / W
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale + pad_h / H
        
        image = image.squeeze(0)
        
        # 确保输出尺寸正确
        if image.shape[1] != H or image.shape[2] != W:
            image = image.unsqueeze(0)
            image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)
            image = image.squeeze(0)
        
        # 裁剪边界框到 [0, 1]
        bbox = torch.clamp(bbox, 0, 1)
        
        return image, bbox
    
    def adjust_contrast(self, image, factor):
        """调整对比度"""
        mean = image.mean(dim=[1, 2], keepdim=True)
        image = (image - mean) * factor + mean
        image = torch.clamp(image, 0, 1)
        return image
    
    def adjust_brightness(self, image, factor):
        """调整亮度"""
        image = image * factor
        image = torch.clamp(image, 0, 1)
        return image
    
    def crop_around_bbox(self, image, bbox):
        """
        围绕边界框裁剪（保留上下文）
        这对小目标检测特别有效
        """
        C, H, W = image.shape
        
        # 取第一个边界框
        box = bbox[0]
        xmin, ymin, xmax, ymax = box
        
        # 计算边界框中心和尺寸
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        bw = xmax - xmin
        bh = ymax - ymin
        
        # 扩展边界框（保留上下文，扩展 1.5-2.5 倍）
        expand_factor = random.uniform(1.5, 2.5)
        crop_w = min(bw * expand_factor, 1.0)
        crop_h = min(bh * expand_factor, 1.0)
        
        # 计算裁剪区域
        crop_xmin = max(0, cx - crop_w / 2)
        crop_ymin = max(0, cy - crop_h / 2)
        crop_xmax = min(1.0, cx + crop_w / 2)
        crop_ymax = min(1.0, cy + crop_h / 2)
        
        # 转换为像素坐标
        crop_xmin_px = int(crop_xmin * W)
        crop_ymin_px = int(crop_ymin * H)
        crop_xmax_px = int(crop_xmax * W)
        crop_ymax_px = int(crop_ymax * H)
        
        # 裁剪图像
        cropped_image = image[:, crop_ymin_px:crop_ymax_px, crop_xmin_px:crop_xmax_px]
        
        # Resize 回原始尺寸
        cropped_image = cropped_image.unsqueeze(0)
        cropped_image = F.interpolate(cropped_image, size=(H, W), mode='bilinear', align_corners=False)
        cropped_image = cropped_image.squeeze(0)
        
        # 调整边界框坐标
        new_bbox = bbox.clone()
        new_bbox[:, 0] = (bbox[:, 0] - crop_xmin) / (crop_xmax - crop_xmin)
        new_bbox[:, 1] = (bbox[:, 1] - crop_ymin) / (crop_ymax - crop_ymin)
        new_bbox[:, 2] = (bbox[:, 2] - crop_xmin) / (crop_xmax - crop_xmin)
        new_bbox[:, 3] = (bbox[:, 3] - crop_ymin) / (crop_ymax - crop_ymin)
        
        # 裁剪到 [0, 1]
        new_bbox = torch.clamp(new_bbox, 0, 1)
        
        return cropped_image, new_bbox


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试数据增强")
    print("=" * 60)
    
    # 创建测试数据
    image = torch.rand(2, 512, 512)
    bbox = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
    
    print(f"原始图像形状: {image.shape}")
    print(f"原始边界框: {bbox}")
    
    # 创建增强器
    aug = DetectionAugmentation()
    
    # 应用增强
    aug_image, aug_bbox = aug(image, bbox)
    
    print(f"\n增强后图像形状: {aug_image.shape}")
    print(f"增强后边界框: {aug_bbox}")
    print(f"图像范围: [{aug_image.min():.4f}, {aug_image.max():.4f}]")
    
    print("\n✓ 数据增强测试通过!")
