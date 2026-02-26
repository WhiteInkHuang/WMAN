"""
检测任务训练脚本
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from Config import parse_args
from datetime import datetime
from pathlib import Path
import time
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import random

# 导入检测相关模块
from dataset import BrainInfarctDataset, detection_collate_fn  # 导入自定义collate函数
from model.Detection2D import Detection2DCNN, AttentionDetection2D  # 改用2D模型
from losses_detection_2d import BBoxLoss, GIoULoss, CIoULoss, compute_detection_metrics
from hybrid_loss import HybridDetectionLoss, AdaptiveDetectionLoss  # 使用2D损失函数


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_dataset(dataset, test_size=0.2):
    """划分训练集和测试集"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(np.floor(test_size * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[test_size:], indices[:test_size]
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset


def get_model(model_name, in_channels=2):
    """根据名称创建模型"""
    if model_name == 'Detection2DCNN':
        return Detection2DCNN(in_channels=in_channels, init_features=32)  # 增加到32
    elif model_name == 'AttentionDetection2D':
        return AttentionDetection2D(in_channels=in_channels, init_features=32)  # 增加到32
    else:
        raise ValueError(f"未知的模型名称: {model_name}")


def get_loss_function(loss_type='bbox'):
    """创建损失函数"""
    if loss_type == 'bbox':
        return BBoxLoss(use_iou_loss=True, iou_weight=2.0, l1_weight=0.5)
    elif loss_type == 'giou':
        return GIoULoss()
    elif loss_type == 'ciou':
        return CIoULoss()
    elif loss_type == 'hybrid':
        return HybridDetectionLoss(ciou_weight=3.0, l1_weight=1.0, aspect_weight=0.5)
    elif loss_type == 'adaptive':
        return AdaptiveDetectionLoss()
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


def get_scheduler(optimizer, scheduler_type='cosine', epochs=500):
    """创建学习率调度器"""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          factor=0.5, patience=10, verbose=True)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        return None


def k_fold_cross_validation_with_test(device, dataset, epochs, test_size=0.2, k_fold=5,
                                    batch_size=1, workers=2, 
                                    model_dir="./checkpoints/", model_name='AttentionDetection2D',
                                    loss_type='bbox', scheduler_type='cosine', learning_rate=1e-3):
    """K折交叉验证训练（检测任务）"""
    start = time.time()
    
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"使用 {num_gpus} 个GPU进行训练!")
    
    # 划分训练集和测试集
    train_dataset, test_dataset = split_dataset(dataset, test_size=test_size)
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    
    # 准备测试集DataLoader - 使用自定义collate函数
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=detection_collate_fn)
    
    # K折交叉验证
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)
    
    # 保存所有fold的结果
    all_results = []
    
    for fold, (train_index, valid_index) in enumerate(kf.split(train_dataset)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{k_fold}")
        print(f"{'='*60}")
        
        k_train_fold = Subset(train_dataset, train_index)
        k_valid_fold = Subset(train_dataset, valid_index)
        print(f"训练大小: {len(k_train_fold)}, 验证大小: {len(k_valid_fold)}")
        
        # DataLoaders - 使用自定义collate函数
        train_dataloader = DataLoader(k_train_fold, batch_size=batch_size, 
                                     shuffle=True, pin_memory=True, num_workers=workers,
                                     collate_fn=detection_collate_fn)
        valid_dataloader = DataLoader(k_valid_fold, batch_size=batch_size, 
                                     shuffle=False, pin_memory=True, num_workers=workers,
                                     collate_fn=detection_collate_fn)
        
        # 模型初始化
        print(f"创建模型: {model_name}")
        model = get_model(model_name, in_channels=2)
        
        # 多GPU训练（可选，注释掉以使用单GPU）
        # if num_gpus > 1:
        #     model = nn.DataParallel(model)
        model = model.to(device)
        
        # 损失函数
        print(f"损失函数: {loss_type}")
        loss_fn = get_loss_function(loss_type)
        if hasattr(loss_fn, 'to'):
            loss_fn = loss_fn.to(device)
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = get_scheduler(optimizer, scheduler_type, epochs)
        if scheduler is not None:
            print(f"学习率调度: {scheduler_type}")
        
        best_iou = 0.0
        best_model_path = os.path.join(model_dir, f"{model_name}_fold{fold}.pth")
        
        for epoch in range(1, epochs + 1):
            # 更新自适应损失的epoch
            if loss_type == 'adaptive' and hasattr(loss_fn, 'set_epoch'):
                loss_fn.set_epoch(epoch)
            
            # 训练阶段
            model.train()
            epoch_train_loss = 0.0
            train_iou = 0.0
            train_giou = 0.0
            
            train_iterator = tqdm(train_dataloader, desc=f"训练 Epoch {epoch}/{epochs}", unit="batch")
            for batch_idx, batch in enumerate(train_iterator):
                # DCM数据集返回字典格式
                if batch is None:
                    continue
                images = batch['image'].to(device, non_blocking=True)
                # bboxes是list，需要逐个移到GPU
                bboxes = [bbox.to(device, non_blocking=True) for bbox in batch['bboxes']]
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # 计算损失
                loss = loss_fn(outputs, bboxes)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
                # 计算指标
                with torch.no_grad():
                    metrics = compute_detection_metrics(outputs, bboxes)
                    train_iou += metrics['iou']
                    train_giou += metrics['giou']
                
                # 更新进度条
                train_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{metrics["iou"]:.4f}'
                })
            
            # 计算训练指标
            epoch_train_loss /= len(train_dataloader)
            train_iou /= len(train_dataloader)
            train_giou /= len(train_dataloader)
            
            # 验证阶段
            model.eval()
            epoch_valid_loss = 0.0
            valid_iou = 0.0
            valid_giou = 0.0
            
            with torch.no_grad():
                valid_iterator = tqdm(valid_dataloader, desc=f"验证 Epoch {epoch}/{epochs}", unit="batch")
                for batch_idx, batch in enumerate(valid_iterator):
                    if batch is None:
                        continue
                    images = batch['image'].to(device, non_blocking=True)
                    # bboxes是list，需要逐个移到GPU
                    bboxes = [bbox.to(device, non_blocking=True) for bbox in batch['bboxes']]
                    outputs = model(images)
                    
                    # 计算损失
                    loss = loss_fn(outputs, bboxes)
                    epoch_valid_loss += loss.item()
                    
                    # 计算指标
                    metrics = compute_detection_metrics(outputs, bboxes)
                    valid_iou += metrics['iou']
                    valid_giou += metrics['giou']
                    
                    # 更新进度条
                    valid_iterator.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'iou': f'{metrics["iou"]:.4f}'
                    })
            
            # 计算验证指标
            epoch_valid_loss /= len(valid_dataloader)
            valid_iou /= len(valid_dataloader)
            valid_giou /= len(valid_dataloader)
            
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(valid_iou)
                else:
                    scheduler.step()
            
            # 打印结果
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train - Loss: {epoch_train_loss:.4f}, IoU: {train_iou:.4f}, GIoU: {train_giou:.4f}")
            print(f"  Valid - Loss: {epoch_valid_loss:.4f}, IoU: {valid_iou:.4f}, GIoU: {valid_giou:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if valid_iou > best_iou:
                best_iou = valid_iou
                # 保存模型
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ 保存最佳模型，IoU: {best_iou:.4f}")
        
        # 加载最佳模型进行测试
        print(f"\n加载最佳模型进行测试...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        # 测试阶段
        test_loss = 0.0
        test_iou = 0.0
        test_giou = 0.0
        
        with torch.no_grad():
            test_iterator = tqdm(test_dataloader, desc="测试", unit="batch")
            for batch_idx, batch in enumerate(test_iterator):
                if batch is None:
                    continue
                images = batch['image'].to(device, non_blocking=True)
                # bboxes是list，需要逐个移到GPU
                bboxes = [bbox.to(device, non_blocking=True) for bbox in batch['bboxes']]
                outputs = model(images)
                
                loss = loss_fn(outputs, bboxes)
                test_loss += loss.item()
                
                metrics = compute_detection_metrics(outputs, bboxes)
                test_iou += metrics['iou']
                test_giou += metrics['giou']
        
        # 计算测试指标
        test_loss /= len(test_dataloader)
        test_iou /= len(test_dataloader)
        test_giou /= len(test_dataloader)
        
        print(f"\n{'='*60}")
        print(f"Fold {fold} 测试结果:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test IoU: {test_iou:.4f}")
        print(f"  Test GIoU: {test_giou:.4f}")
        print(f"{'='*60}")
        
        # 保存结果
        fold_result = {
            'fold': fold,
            'best_valid_iou': best_iou,
            'test_iou': test_iou,
            'test_giou': test_giou,
            'test_loss': test_loss
        }
        all_results.append(fold_result)
        
        # 保存到文件
        with open(os.path.join(model_dir, "results_detection.txt"), "a") as f:
            f.write(f"Fold {fold} - Best Valid IoU: {best_iou:.4f}, "
                   f"Test IoU: {test_iou:.4f}, Test GIoU: {test_giou:.4f}\n")
    
    # 计算平均结果
    avg_valid_iou = np.mean([r['best_valid_iou'] for r in all_results])
    avg_test_iou = np.mean([r['test_iou'] for r in all_results])
    avg_test_giou = np.mean([r['test_giou'] for r in all_results])
    
    std_test_iou = np.std([r['test_iou'] for r in all_results])
    std_test_giou = np.std([r['test_giou'] for r in all_results])
    
    print(f"\n{'='*60}")
    print(f"最终结果 ({k_fold}折交叉验证)")
    print(f"{'='*60}")
    print(f"平均验证 IoU: {avg_valid_iou:.4f}")
    print(f"平均测试 IoU: {avg_test_iou:.4f} ± {std_test_iou:.4f}")
    print(f"平均测试 GIoU: {avg_test_giou:.4f} ± {std_test_giou:.4f}")
    print(f"{'='*60}")
    
    # 保存最终结果
    with open(os.path.join(model_dir, "results_detection.txt"), "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"最终结果 ({k_fold}折交叉验证)\n")
        f.write(f"{'='*60}\n")
        f.write(f"平均验证 IoU: {avg_valid_iou:.4f}\n")
        f.write(f"平均测试 IoU: {avg_test_iou:.4f} ± {std_test_iou:.4f}\n")
        f.write(f"平均测试 GIoU: {avg_test_giou:.4f} ± {std_test_giou:.4f}\n")
    
    total_time = time.time() - start
    print(f"\n总训练时间: {total_time // 60:.0f}m {total_time % 60:.0f}s")


if __name__ == '__main__':
    setup_seed(42)
    current_time = "{0:%Y%m%d_%H_%M}".format(datetime.now())
    args = parse_args()
    
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集初始化（DCM格式）
    # 注意：DCM是2D切片，所以target_size只需要(H, W)
    target_size = (512, 512)  # 2D图像尺寸
    
    # 加载训练集和测试集
    train_dataset = BrainInfarctDataset(
        data_prefix=args.data_dir,  # 应该是 "./splits"
        split='train',
        target_size=target_size
    )
    
    test_dataset = BrainInfarctDataset(
        data_prefix=args.data_dir,
        split='test',
        target_size=target_size
    )
    
    # 合并为一个数据集用于K折交叉验证
    # 注意：这里需要修改，因为BrainInfarctDataset返回的是字典格式
    print(f"训练集: {len(train_dataset)} 个样本（DCM格式）")
    print(f"测试集: {len(test_dataset)} 个样本（DCM格式）")
    
    # 使用训练集进行K折交叉验证
    dataset = train_dataset
    
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据路径!")
        sys.exit(1)
    
    # 创建模型目录
    model_dir = args.checkpoint_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"检测任务训练配置（DCM格式）")
    print(f"{'='*60}")
    print(f"数据格式: DCM (DICOM 2D切片)")
    print(f"模态组合: CT + MRI")
    print(f"归一化: Min-Max归一化")
    print(f"图像尺寸: {target_size}")
    print(f"模型: {args.model_name}")
    print(f"损失函数: BBox Loss (L1 + IoU)")
    print(f"评估指标: IoU, GIoU")
    print(f"学习率调度: cosine")
    print(f"初始学习率: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"K-Fold: {args.k_split_value}")
    print(f"{'='*60}\n")
    
    # 运行K折交叉验证
    k_fold_cross_validation_with_test(
        device=device,
        dataset=dataset,
        epochs=args.epochs,
        k_fold=args.k_split_value,
        batch_size=args.batch_size,
        workers=args.num_workers,
        model_dir=model_dir,
        model_name=args.model_name,
        loss_type='hybrid',  # 使用混合损失，结合CIoU + Focal L1 + Aspect Ratio
        scheduler_type='cosine',
        learning_rate=args.learning_rate
    )
