# model/nnunet/generic_UNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedConvLayers(nn.Module):
    def __init__(self, input_features, output_features, num_convs, conv_op, conv_kwargs,
                 norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op=None, dropout_op_kwargs=None):
        super(StackedConvLayers, self).__init__()
        layers = []
        current_input_features = input_features
        for _ in range(num_convs):
            layers.append(conv_op(current_input_features, output_features, **conv_kwargs))
            if dropout_op is not None and dropout_op_kwargs is not None and dropout_op_kwargs['p'] > 0:
                layers.append(dropout_op(**dropout_op_kwargs))
            layers.append(norm_op(output_features, **norm_op_kwargs))
            layers.append(nonlin(**nonlin_kwargs))
            current_input_features = output_features
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generic_UNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool,
                 num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=None, weightInitializer=None,
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None):
        super(Generic_UNet, self).__init__()

        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2, 2)] * num_pool

        self.final_nonlin = final_nonlin
        self.conv_op = conv_op
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        self.tu = nn.ModuleList()
        self.seg_outputs = nn.ModuleList()

        self.sigmoid = nn.Sigmoid()

        # 下采样部分
        input_features = input_channels
        output_features = base_num_features
        for p in range(num_pool):
            self.down_path.append(
                StackedConvLayers(input_features, output_features, num_conv_per_stage, conv_op,
                                  {'kernel_size': conv_kernel_sizes[p], 'padding': tuple(k // 2 for k in conv_kernel_sizes[p])},
                                  norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs)
            )
            input_features = output_features
            output_features = int(output_features * feat_map_mul_on_downscale)

        # 最底层
        self.bottleneck = StackedConvLayers(input_features, output_features, num_conv_per_stage, conv_op,
                                            {'kernel_size': conv_kernel_sizes[num_pool], 'padding': tuple(k // 2 for k in conv_kernel_sizes[num_pool])},
                                            norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs)

        # 上采样部分
        for p in range(num_pool):
            self.tu.append(nn.ConvTranspose3d(output_features, output_features // 2,
                                              pool_op_kernel_sizes[-(p + 1)], stride=pool_op_kernel_sizes[-(p + 1)]))
            output_features = output_features // 2
            self.up_path.append(
                StackedConvLayers(output_features * 2, output_features, num_conv_per_stage, conv_op,
                                  {'kernel_size': conv_kernel_sizes[num_pool - (p + 1)], 'padding': tuple(k // 2 for k in conv_kernel_sizes[num_pool - (p + 1)])},
                                  norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs)
            )

        # 最终分割输出
        self.final_conv = nn.Conv3d(output_features, num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.down_path:
            x = down(x)
            skips.append(x)
            x = F.max_pool3d(x, kernel_size=(1, 2, 2) if len(skips) == 1 else (2, 2, 2))  # 第一次只在H,W方向下采样

        x = self.bottleneck(x)

        for i in range(len(self.up_path)):
            x = self.tu[i](x)
            skip = skips[-(i + 1)]
            if x.shape != skip.shape:
                x = self._pad(x, skip)
            x = torch.cat((skip, x), dim=1)
            x = self.up_path[i](x)

        x = self.final_conv(x)
        # if self.final_nonlin is not None:
        #     x = self.final_nonlin(x)
        x = self.sigmoid(x)
        return x

    @staticmethod
    def _pad(x, target):
        diffZ = target.size(2) - x.size(2)
        diffY = target.size(3) - x.size(3)
        diffX = target.size(4) - x.size(4)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        return x
