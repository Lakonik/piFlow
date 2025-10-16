from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer

from mmgen.models.builder import MODULES, build_module
from mmgen.models.architectures.ddpm.modules import (
    MultiHeadAttention, DenoisingResBlock, DenoisingDownsample, DenoisingUpsample)


@MODULES.register_module()
class MultiHeadAttentionMod(MultiHeadAttention):

    def __init__(self,
                 in_channels,
                 num_heads=1,
                 groups=1,
                 norm_cfg=dict(type='GN', num_groups=32)):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.groups = groups
        _, self.norm = build_norm_layer(norm_cfg, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1, groups=groups)
        self.proj = nn.Conv1d(in_channels, in_channels, 1, groups=groups)

        if hasattr(F, 'scaled_dot_product_attention'):
            self.attn_proc = self.efficient_attn
        else:
            self.attn_proc = self.QKVAttention

        self.init_weights()

    @staticmethod
    def efficient_attn(qkv):
        q, k, v = torch.chunk(qkv, 3, dim=1)
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x):
        """Forward function for multi head attention.
        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Feature map after attention.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        spatial_numel = x.size(-1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(
            b, self.groups, -1, spatial_numel
        ).transpose(1, 2).reshape(b * self.num_heads, -1, self.groups * spatial_numel)
        h = self.attn_proc(qkv)
        h = h.reshape(
            b, -1, self.groups, spatial_numel
        ).transpose(1, 2).reshape(b, -1, spatial_numel)
        h = self.proj(h)
        return (h + x).reshape(b, c, *spatial)


@MODULES.register_module()
class DenoisingResBlockMod(DenoisingResBlock):
    def __init__(self,
                 in_channels,
                 embedding_channels,
                 use_scale_shift_norm,
                 dropout,
                 groups=1,
                 out_channels=None,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1):
        super(DenoisingResBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels

        _norm_cfg = deepcopy(norm_cfg)

        _, norm_1 = build_norm_layer(_norm_cfg, in_channels)
        conv_1 = [
            norm_1,
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups)
        ]
        self.conv_1 = nn.Sequential(*conv_1)

        norm_with_embedding_cfg = dict(
            in_channels=out_channels,
            embedding_channels=embedding_channels,
            use_scale_shift=use_scale_shift_norm,
            norm_cfg=_norm_cfg)
        self.norm_with_embedding = build_module(
            dict(type='NormWithEmbedding'),
            default_args=norm_with_embedding_cfg)

        conv_2 = [
            build_activation_layer(act_cfg),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=groups)
        ] if dropout > 0 else [
            build_activation_layer(act_cfg),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=groups)
        ]
        self.conv_2 = nn.Sequential(*conv_2)

        assert shortcut_kernel_size in [
            1, 3
        ], ('Only support `1` and `3` for `shortcut_kernel_size`, but '
            f'receive {shortcut_kernel_size}.')

        self.learnable_shortcut = out_channels != in_channels

        if self.learnable_shortcut:
            shortcut_padding = 1 if shortcut_kernel_size == 3 else 0
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                shortcut_kernel_size,
                padding=shortcut_padding,
                groups=groups)
        self.init_weights()


@MODULES.register_module()
class DenoisingDownsampleMod(DenoisingDownsample):
    def __init__(self, in_channels, groups=1, with_conv=True):
        super(DenoisingDownsample, self).__init__()
        if with_conv:
            self.downsample = nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=groups)
        else:
            self.downsample = nn.AvgPool2d(stride=2)


@MODULES.register_module()
class DenoisingUpsampleMod(DenoisingUpsample):
    def __init__(self, in_channels, groups=1, with_conv=True):
        super(DenoisingUpsample, self).__init__()
        if with_conv:
            self.with_conv = True
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=groups)
