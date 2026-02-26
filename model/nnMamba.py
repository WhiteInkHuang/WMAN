import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,  # 通道数
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, D, H, W] → 先换维度到 [B, D*H*W, C]
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, d*h*w, c)
        x = self.mamba(self.norm(x))
        # 再换回 [B, C, D, H, W]
        x = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return x

class nnMambaUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels)
        )
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*2)
        )
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*4)
        )
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*8)
        )
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*16, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*16)
        )

        self.up4 = nn.ConvTranspose3d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv3d(base_channels*16, base_channels*8, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*8)
        )
        self.up3 = nn.ConvTranspose3d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*4)
        )
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels*2)
        )
        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            MambaBlock3D(base_channels)
        )
        self.final = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with interpolation if needed
        d4 = self.up4(b)
        if d4.shape[-3:] != e4.shape[-3:]:
            d4 = F.interpolate(d4, size=e4.shape[-3:], mode='trilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape[-3:] != e3.shape[-3:]:
            d3 = F.interpolate(d3, size=e3.shape[-3:], mode='trilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = F.interpolate(d2, size=e2.shape[-3:], mode='trilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = F.interpolate(d1, size=e1.shape[-3:], mode='trilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))