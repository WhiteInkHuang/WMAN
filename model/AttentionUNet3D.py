"""
Attention U-Net 3D 实现
结合注意力机制的 U-Net，用于医学图像分割
论文: Attention U-Net: Learning Where to Look for the Pancreas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock3D(nn.Module):
    """
    注意力门控模块
    用于在跳跃连接中突出显示相关特征
    """
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: 门控信号的通道数（来自解码器）
        F_l: 输入特征的通道数（来自编码器）
        F_int: 中间层的通道数
        """
        super(AttentionBlock3D, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        g: 门控信号（来自解码器的上采样特征）
        x: 输入特征（来自编码器的跳跃连接）
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class CBAM3D(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 3D版本
    结合通道注意力和空间注意力
    论文: CBAM: Convolutional Block Attention Module
    """
    def __init__(self, channels, reduction=16):
        super(CBAM3D, self).__init__()
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # 空间注意力
        self.conv_spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(1)
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


class ConvBlock3D(nn.Module):
    """3D卷积块"""
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super(ConvBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM3D(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.use_cbam:
            x = self.cbam(x)
        
        return x


class AttentionUNet3D(nn.Module):
    """
    Attention U-Net 3D
    结合注意力门控和CBAM的改进U-Net
    """
    def __init__(self, in_channels=2, out_channels=1, init_features=16, use_cbam=True):
        super(AttentionUNet3D, self).__init__()
        
        features = init_features
        
        # 编码器
        self.encoder1 = ConvBlock3D(in_channels, features, use_cbam=use_cbam)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock3D(features, features * 2, use_cbam=use_cbam)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock3D(features * 2, features * 4, use_cbam=use_cbam)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = ConvBlock3D(features * 4, features * 8, use_cbam=use_cbam)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 瓶颈层
        self.bottleneck = ConvBlock3D(features * 8, features * 16, use_cbam=use_cbam)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock3D(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = ConvBlock3D(features * 16, features * 8, use_cbam=use_cbam)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = ConvBlock3D(features * 8, features * 4, use_cbam=use_cbam)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = ConvBlock3D(features * 4, features * 2, use_cbam=use_cbam)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = ConvBlock3D(features * 2, features, use_cbam=use_cbam)
        
        # 输出层
        self.conv_out = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 解码器 + 注意力门控
        dec4 = self.upconv4(bottleneck)
        if dec4.shape[-3:] != enc4.shape[-3:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[-3:], mode='trilinear', align_corners=True)
        enc4_att = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat((dec4, enc4_att), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.shape[-3:] != enc3.shape[-3:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[-3:], mode='trilinear', align_corners=True)
        enc3_att = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat((dec3, enc3_att), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape[-3:] != enc2.shape[-3:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[-3:], mode='trilinear', align_corners=True)
        enc2_att = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat((dec2, enc2_att), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape[-3:] != enc1.shape[-3:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[-3:], mode='trilinear', align_corners=True)
        enc1_att = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat((dec1, enc1_att), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 输出
        out = self.conv_out(dec1)
        out = self.sigmoid(out)
        
        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Attention U-Net 3D")
    print("=" * 60)
    
    # 创建模型
    model = AttentionUNet3D(in_channels=2, out_channels=1, init_features=16, use_cbam=True)
    
    # 测试输入
    batch_size = 1
    x = torch.randn(batch_size, 2, 32, 512, 512)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n✓ Attention U-Net 3D 测试通过!")
