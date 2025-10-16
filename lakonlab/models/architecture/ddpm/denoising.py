from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule

from mmgen.models.architectures.ddpm.modules import TimeEmbedding, EmbedSequential
from mmgen.models.architectures.ddpm.denoising import DenoisingUnet
from mmgen.models.builder import MODULES, build_module


@MODULES.register_module()
class DenoisingUnetMod(DenoisingUnet):

    def __init__(self,
                 image_size,
                 in_channels=3,
                 concat_cond_channels=0,
                 base_channels=128,
                 resblocks_per_downsample=3,
                 num_timesteps=1000,
                 use_rescale_timesteps=True,
                 dropout=0,
                 embedding_channels=-1,
                 num_classes=0,
                 channels_cfg=None,
                 groups=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1,
                 use_scale_shift_norm=False,
                 num_heads=4,
                 time_embedding_mode='sin',
                 time_embedding_cfg=None,
                 resblock_cfg=dict(type='DenoisingResBlockMod'),
                 attention_cfg=dict(type='MultiHeadAttentionMod'),
                 downsample_conv=True,
                 upsample_conv=True,
                 downsample_cfg=dict(type='DenoisingDownsampleMod'),
                 upsample_cfg=dict(type='DenoisingUpsampleMod'),
                 attention_res=[16, 8],
                 pretrained=None):
        super(DenoisingUnet, self).__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.use_rescale_timesteps = use_rescale_timesteps

        out_channels = in_channels
        self.out_channels = out_channels
        self.concat_cond_channels = concat_cond_channels

        # check type of image_size
        if isinstance(image_size, list) or isinstance(image_size, tuple):
            assert len(image_size) == 2, 'The length of `image_size` should be 2.'
        elif isinstance(image_size, int):
            image_size = [image_size, image_size]
        else:
            raise TypeError('Only support `int` and `list[int]` for `image_size`.')
        self.image_size = image_size

        if isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channels_cfg`, '
                             f'receive {type(channels_cfg)}')

        embedding_channels = base_channels * 4 \
            if embedding_channels == -1 else embedding_channels
        self.time_embedding = TimeEmbedding(
            base_channels,
            embedding_channels=embedding_channels,
            embedding_mode=time_embedding_mode,
            embedding_cfg=time_embedding_cfg,
            act_cfg=act_cfg)

        if self.num_classes != 0:
            self.label_embedding = nn.Embedding(self.num_classes,
                                                embedding_channels)

        self.resblock_cfg = deepcopy(resblock_cfg)
        self.resblock_cfg.setdefault('dropout', dropout)
        self.resblock_cfg.setdefault('groups', groups)
        self.resblock_cfg.setdefault('norm_cfg', norm_cfg)
        self.resblock_cfg.setdefault('act_cfg', act_cfg)
        self.resblock_cfg.setdefault('embedding_channels', embedding_channels)
        self.resblock_cfg.setdefault('use_scale_shift_norm',
                                     use_scale_shift_norm)
        self.resblock_cfg.setdefault('shortcut_kernel_size',
                                     shortcut_kernel_size)

        # get scales of ResBlock to apply attention
        attention_scale = [min(image_size) // int(res) for res in attention_res]
        self.attention_cfg = deepcopy(attention_cfg)
        self.attention_cfg.setdefault('num_heads', num_heads)
        self.attention_cfg.setdefault('groups', groups)
        self.attention_cfg.setdefault('norm_cfg', norm_cfg)

        self.downsample_cfg = deepcopy(downsample_cfg)
        self.downsample_cfg.setdefault('groups', groups)
        self.downsample_cfg.setdefault('with_conv', downsample_conv)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.upsample_cfg.setdefault('groups', groups)
        self.upsample_cfg.setdefault('with_conv', upsample_conv)

        # init the channel scale factor
        scale = 1
        self.in_blocks = nn.ModuleList([
            EmbedSequential(
                nn.Conv2d(in_channels + concat_cond_channels, base_channels, 3, 1, padding=1, groups=groups))
        ])
        self.in_channels_list = [base_channels]

        # construct the encoder part of Unet
        for level, factor in enumerate(self.channel_factor_list):
            in_channels_ = base_channels if level == 0 \
                else base_channels * self.channel_factor_list[level - 1]
            out_channels_ = base_channels * factor

            for _ in range(resblocks_per_downsample):
                layers = [
                    build_module(self.resblock_cfg, {
                        'in_channels': in_channels_,
                        'out_channels': out_channels_
                    })
                ]
                in_channels_ = out_channels_

                if scale in attention_scale:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))

                self.in_channels_list.append(in_channels_)
                self.in_blocks.append(EmbedSequential(*layers))

            if level != len(self.channel_factor_list) - 1:
                self.in_blocks.append(
                    EmbedSequential(
                        build_module(self.downsample_cfg,
                                     {'in_channels': in_channels_})))
                self.in_channels_list.append(in_channels_)
                scale *= 2

        # construct the bottom part of Unet
        self.mid_blocks = EmbedSequential(
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
            build_module(self.attention_cfg, {'in_channels': in_channels_}),
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
        )

        # construct the decoder part of Unet
        in_channels_list = deepcopy(self.in_channels_list)
        self.out_blocks = nn.ModuleList()
        for level, factor in enumerate(self.channel_factor_list[::-1]):
            for idx in range(resblocks_per_downsample + 1):
                layers = [
                    build_module(
                        self.resblock_cfg, {
                            'in_channels':
                            in_channels_ + in_channels_list.pop(),
                            'out_channels': base_channels * factor
                        })
                ]
                in_channels_ = base_channels * factor
                if scale in attention_scale:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))
                if (level != len(self.channel_factor_list) - 1
                        and idx == resblocks_per_downsample):
                    layers.append(
                        build_module(self.upsample_cfg,
                                     {'in_channels': in_channels_}))
                    scale //= 2
                self.out_blocks.append(EmbedSequential(*layers))

        self.out = ConvModule(
            in_channels=in_channels_,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

        self.init_weights(pretrained)

    def forward(self, x_t, t, label=None, concat_cond=None, return_noise=False):
        if self.use_rescale_timesteps:
            t = t.float() * (1000.0 / self.num_timesteps)
        with torch.autocast(
                device_type='cuda',
                enabled=True,
                dtype=x_t.dtype):
            embedding = self.time_embedding(t)

        if label is not None:
            assert hasattr(self, 'label_embedding')
            embedding = self.label_embedding(label) + embedding

        h, hs = x_t, []
        if self.concat_cond_channels > 0:
            h = torch.cat([h, concat_cond], dim=1)
        # forward downsample blocks
        for block in self.in_blocks:
            h = block(h, embedding)
            hs.append(h)

        # forward middle blocks
        h = self.mid_blocks(h, embedding)

        # forward upsample blocks
        for block in self.out_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), embedding)
        outputs = self.out(h)

        return outputs
