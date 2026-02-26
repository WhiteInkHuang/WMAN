# model/nnUNet3D.py
import torch
import torch.nn as nn
from model.nnunet.generic_UNet import Generic_UNet

def build_nnUNet3D(in_channels=1, out_channels=1):
    """
    构建精简版原版nnUNet 3D结构
    支持输入尺寸 (B, 1, 29-32, 512, 512)
    """
    net = Generic_UNet(
        input_channels=in_channels,
        base_num_features=32,
        num_classes=out_channels,
        num_pool=5,  # 总共下采样5次
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={'p': 0, 'inplace': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
        deep_supervision=False,
        dropout_in_localization=False,
        final_nonlin=lambda x: torch.sigmoid(x),
        weightInitializer=None,
        pool_op_kernel_sizes=[
            (1, 2, 2),  # 第一次下采样深度方向不变
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2)
        ],
        conv_kernel_sizes=[
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3)
        ]
    )
    return net
