"""
2D检测模型（适配单切片数据）
输出边界框而不是分割mask
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Detection2DCNN(nn.Module):
    """
    2D CNN检测模型
    输入: [B, 2, H, W]
    输出: [B, 4] - 边界框 [x_min, y_min, x_max, y_max]
    """
    def __init__(self, in_channels=2, init_features=16):
        super(Detection2DCNN, self).__init__()
        
        features = init_features
        
        # 编码器
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈层
        self.bottleneck = self._block(features * 8, features * 16)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层（回归边界框）
        self.fc1 = nn.Linear(features * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # 输出4个值
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()  # 确保输出在[0,1]
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 全局池化
        pooled = self.global_pool(bottleneck)
        pooled = pooled.view(pooled.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # 输出格式: [cx, cy, w, h]
        x = self.sigmoid(x)
        
        # 转换为 [xmin, ymin, xmax, ymax]
        cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        
        # 裁剪到[0, 1]
        xmin = torch.clamp(xmin, 0, 1)
        ymin = torch.clamp(ymin, 0, 1)
        xmax = torch.clamp(xmax, 0, 1)
        ymax = torch.clamp(ymax, 0, 1)
        
        # 堆叠输出
        output = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        
        return output
    
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


class AttentionDetection2D(nn.Module):
    """
    带注意力机制的2D检测模型
    """
    def __init__(self, in_channels=2, init_features=16):
        super(AttentionDetection2D, self).__init__()
        
        features = init_features
        
        # 编码器（带CBAM）
        self.encoder1 = self._block_with_cbam(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block_with_cbam(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block_with_cbam(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block_with_cbam(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈层
        self.bottleneck = self._block_with_cbam(features * 8, features * 16)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(features * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 全局池化
        pooled = self.global_pool(bottleneck)
        pooled = pooled.view(pooled.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # 输出格式: [cx, cy, w, h]
        x = self.sigmoid(x)
        
        # 转换为 [xmin, ymin, xmax, ymax]
        cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        
        # 裁剪到[0, 1]
        xmin = torch.clamp(xmin, 0, 1)
        ymin = torch.clamp(ymin, 0, 1)
        xmax = torch.clamp(xmax, 0, 1)
        ymax = torch.clamp(ymax, 0, 1)
        
        # 堆叠输出
        output = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        
        return output
    
    def _block_with_cbam(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            CBAM2D(features)
        )


class CBAM2D(nn.Module):
    """CBAM注意力模块（2D版本）"""
    def __init__(self, channels, reduction=16):
        super(CBAM2D, self).__init__()
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # 空间注意力
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid_spatial = nn.Sigmoid()
    
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv_spatial(spatial_att)
        spatial_att = self.sigmoid_spatial(spatial_att)
        x = x * spatial_att
        
        return x


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试2D检测模型")
    print("=" * 60)
    
    # 创建模型
    model = Detection2DCNN(in_channels=2, init_features=16)
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 2, 512, 512)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")  # [2, 4]
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"\n第一个样本的边界框: {output[0]}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    print("\n✓ 2D检测模型测试通过!")
    
    # 测试注意力模型
    print("\n" + "=" * 60)
    print("测试注意力2D检测模型")
    print("=" * 60)
    
    model_att = AttentionDetection2D(in_channels=2, init_features=16)
    
    with torch.no_grad():
        output_att = model_att(x)
    
    print(f"输出形状: {output_att.shape}")
    print(f"输出范围: [{output_att.min():.4f}, {output_att.max():.4f}]")
    
    total_params_att = sum(p.numel() for p in model_att.parameters())
    print(f"总参数量: {total_params_att:,}")
    
    print("\n✓ 注意力2D检测模型测试通过!")
