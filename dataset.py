import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import torch.nn.functional as F
from pathlib import Path
from augmentation import DetectionAugmentation


class BrainInfarctDataset(Dataset):
    """
    脑梗检测数据集 - 支持DCM格式
    输出: 图像 + 2D边界框 [x_min, y_min, x_max, y_max]
    
    数据格式：
    splits/
    ├── train/
    │   ├── 320997/
    │   │   ├── label/
    │   │   │   └── CT_slice_018.json  # 标注文件（路径写的是.jpg）
    │   │   └── registered/
    │   │       ├── CT/
    │   │       │   └── slice_0018.dcm  # 实际是.dcm文件
    │   │       └── MRI/
    │   │           └── slice_0018.dcm
    """
    
    def __init__(self, data_prefix="./splits", split='train', target_size=(512, 512), use_augmentation=True):
        """
        初始化数据集
        
        :param data_prefix: splits文件夹路径
        :param split: 'train', 'val', 或 'test'
        :param target_size: 目标尺寸 (H, W)
        :param use_augmentation: 是否使用数据增强（仅训练集）
        """
        self.data_prefix = Path(data_prefix)
        self.split = split
        self.target_size = target_size
        self.use_augmentation = use_augmentation and (split == 'train')
        
        # 初始化数据增强
        if self.use_augmentation:
            self.augmentation = DetectionAugmentation(
                flip_prob=0.7,          # 增加翻转概率
                rotate_prob=0.5,        # 增加旋转概率
                scale_prob=0.7,         # 增加缩放概率
                contrast_prob=0.7,      # 增加对比度调整概率
                brightness_prob=0.7,    # 增加亮度调整概率
                crop_prob=0.6,          # 增加裁剪概率（关键！对小目标最有效）
                max_rotate_angle=15,    # 增加旋转角度
                scale_range=(0.8, 1.2), # 增加缩放范围
                contrast_range=(0.8, 1.2),
                brightness_range=(0.85, 1.15)
            )
        
        self.samples = self._load_samples()
        
        aug_info = " (含数据增强)" if self.use_augmentation else ""
        print(f"[{split}] 加载了 {len(self.samples)} 个样本（DCM格式）{aug_info}")
    
    def _load_samples(self):
        """扫描所有标注文件"""
        samples = []
        split_dir = self.data_prefix / self.split
        
        if not split_dir.exists():
            print(f"警告：{split_dir} 不存在！")
            return samples
        
        # 遍历所有患者文件夹
        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            
            label_dir = patient_dir / 'label'
            registered_dir = patient_dir / 'registered'
            
            # 检查必要的文件夹是否存在
            if not label_dir.exists() or not registered_dir.exists():
                continue
            
            # 遍历所有JSON标注文件
            for json_file in label_dir.glob('*.json'):
                try:
                    # 读取JSON标注
                    with open(json_file, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    
                    # 解析标注
                    if 'shapes' not in annotation or len(annotation['shapes']) == 0:
                        continue
                    
                    # 从JSON中的imagePath提取切片编号
                    # imagePath示例: "..\\CT\\CT_slice_019.jpg"
                    image_path = annotation['imagePath']
                    slice_num = self._extract_slice_number(image_path)
                    
                    if slice_num is None:
                        continue
                    
                    # 构建实际的DCM文件路径
                    ct_path = registered_dir / 'CT' / f'slice_{slice_num:04d}.dcm'
                    mri_path = registered_dir / 'MRI' / f'slice_{slice_num:04d}.dcm'
                    
                    # 检查文件是否存在
                    if not ct_path.exists() or not mri_path.exists():
                        continue
                    
                    # 提取边界框
                    bboxes = []
                    for shape in annotation['shapes']:
                        if shape['shape_type'] == 'rectangle':
                            points = shape['points']
                            x1, y1 = points[0]
                            x2, y2 = points[1]
                            
                            # 确保坐标顺序正确
                            xmin = min(x1, x2)
                            ymin = min(y1, y2)
                            xmax = max(x1, x2)
                            ymax = max(y1, y2)
                            
                            bboxes.append([xmin, ymin, xmax, ymax])
                    
                    if len(bboxes) == 0:
                        continue
                    
                    samples.append({
                        'patient_id': patient_dir.name,
                        'slice_num': slice_num,
                        'ct_path': str(ct_path),
                        'mri_path': str(mri_path),
                        'bboxes': bboxes
                    })
                
                except Exception as e:
                    print(f"处理 {json_file} 时出错: {e}")
                    continue
        
        return samples
    
    def _extract_slice_number(self, image_path):
        """
        从JSON中的imagePath提取切片编号
        示例: "..\\CT\\CT_slice_019.jpg" -> 19
        """
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 3:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return None
    
    def _load_dicom(self, dcm_path):
        """加载DICOM文件并转换为numpy数组"""
        try:
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array.astype(np.float32)
            
            # 归一化到[0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            return image
        except Exception as e:
            print(f"加载DICOM文件失败 {dcm_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载CT和MRI图像
        ct_image = self._load_dicom(sample['ct_path'])
        mri_image = self._load_dicom(sample['mri_path'])
        
        if ct_image is None or mri_image is None:
            print(f"警告：无法加载图像 {sample['patient_id']}")
            return None
        
        # 记录原始尺寸（用于边界框归一化）
        orig_h, orig_w = ct_image.shape
        
        # Resize到目标尺寸
        ct_tensor = torch.from_numpy(ct_image).unsqueeze(0).unsqueeze(0)
        mri_tensor = torch.from_numpy(mri_image).unsqueeze(0).unsqueeze(0)
        
        ct_image = F.interpolate(ct_tensor, size=self.target_size, mode='bilinear', align_corners=False)
        mri_image = F.interpolate(mri_tensor, size=self.target_size, mode='bilinear', align_corners=False)
        
        ct_image = ct_image.squeeze().numpy()
        mri_image = mri_image.squeeze().numpy()
        
        # 堆叠为双通道图像 (2, H, W)
        image = np.stack([ct_image, mri_image], axis=0)
        
        # 边界框 - 保留所有边界框
        bboxes = np.array(sample['bboxes'], dtype=np.float32)  # (N, 4)
        
        # 归一化边界框坐标到[0, 1]（使用原始图像尺寸）
        bboxes[:, 0] /= orig_w  # xmin
        bboxes[:, 1] /= orig_h  # ymin
        bboxes[:, 2] /= orig_w  # xmax
        bboxes[:, 3] /= orig_h  # ymax
        
        # 转换为tensor
        image = torch.from_numpy(image).float()
        bboxes = torch.from_numpy(bboxes).float()
        
        # 应用数据增强（仅训练集）
        if self.use_augmentation:
            image, bboxes = self.augmentation(image, bboxes)
        
        return {
            'image': image,  # (2, H, W)
            'bboxes': bboxes,  # (N, 4) [xmin, ymin, xmax, ymax] 归一化到[0,1]
            'patient_id': sample['patient_id'],
            'slice_num': sample['slice_num']
        }


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试数据集
    print("=" * 60)
    print("测试训练集")
    print("=" * 60)
    train_dataset = BrainInfarctDataset(data_prefix="./splits", split='train')
    
    print("\n" + "=" * 60)
    print("测试测试集")
    print("=" * 60)
    test_dataset = BrainInfarctDataset(data_prefix="./splits", split='test')
    
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    print(f"总计: {len(train_dataset) + len(test_dataset)} 个样本")
    
    # 可视化第一个样本
    if len(train_dataset) > 0:
        print("\n" + "=" * 60)
        print("可视化第一个训练样本")
        print("=" * 60)
        
        sample = train_dataset[0]
        if sample is not None:
            image = sample['image']  # (2, H, W)
            bboxes = sample['bboxes']  # (N, 4)
        
            print(f"患者ID: {sample['patient_id']}")
            print(f"切片编号: {sample['slice_num']}")
            print(f"图像形状: {image.shape}")
            print(f"边界框数量: {len(bboxes)}")
            print(f"边界框坐标: {bboxes}")
            
            # 绘制图像和边界框
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 获取图像尺寸
            h, w = image.shape[1], image.shape[2]
            
            # CT图像
            axes[0].imshow(image[0], cmap='gray')
            axes[0].set_title('CT')
            for bbox in bboxes:
                # 边界框是归一化的，需要转回像素坐标
                xmin, ymin, xmax, ymax = bbox
                xmin_px = xmin * w
                ymin_px = ymin * h
                xmax_px = xmax * w
                ymax_px = ymax * h
                rect = plt.Rectangle((xmin_px, ymin_px), xmax_px-xmin_px, ymax_px-ymin_px,
                                    fill=False, edgecolor='red', linewidth=2)
                axes[0].add_patch(rect)
            
            # MRI图像
            axes[1].imshow(image[1], cmap='gray')
            axes[1].set_title('MRI')
            for bbox in bboxes:
                # 边界框是归一化的，需要转回像素坐标
                xmin, ymin, xmax, ymax = bbox
                xmin_px = xmin * w
                ymin_px = ymin * h
                xmax_px = xmax * w
                ymax_px = ymax * h
                rect = plt.Rectangle((xmin_px, ymin_px), xmax_px-xmin_px, ymax_px-ymin_px,
                                    fill=False, edgecolor='red', linewidth=2)
                axes[1].add_patch(rect)
            
            plt.tight_layout()
            plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
            print("\n可视化结果已保存到 dataset_sample.png")
            plt.close()


def detection_collate_fn(batch):
    """
    自定义collate函数，处理不同数量的边界框
    
    Args:
        batch: list of dict, 每个dict包含 'image', 'bboxes', 'patient_id', 'slice_num'
    
    Returns:
        dict with:
            - images: [B, 2, H, W]
            - bboxes: list of [N_i, 4], 每个样本的边界框数量可能不同
            - patient_ids: list of str
            - slice_nums: list of int
    """
    images = []
    bboxes = []
    patient_ids = []
    slice_nums = []
    
    for sample in batch:
        if sample is not None:
            images.append(sample['image'])
            bboxes.append(sample['bboxes'])
            patient_ids.append(sample['patient_id'])
            slice_nums.append(sample['slice_num'])
    
    # 堆叠图像
    images = torch.stack(images, dim=0)  # [B, 2, H, W]
    
    return {
        'image': images,
        'bboxes': bboxes,  # list of tensors
        'patient_id': patient_ids,
        'slice_num': slice_nums
    }
