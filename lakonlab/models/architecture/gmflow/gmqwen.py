import numpy as np
import torch
import torch.nn as nn

from typing import Any, Dict, Optional, Tuple, List
from accelerate import init_empty_weights
from diffusers.models import ModelMixin
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel, QwenEmbedRope, QwenImageTransformerBlock, QwenTimestepProjEmbeddings)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, RMSNorm
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from peft import LoraConfig
from mmcv.cnn import constant_init, xavier_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_checkpoint, load_full_state_dict
from ..utils import flex_freeze
from .gm_output import GMFlowModelOutput


class _GMQwenImageTransformer2DModel(QwenImageTransformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            logweights_channels=1,
            in_channels: int = 64,
            out_channels: Optional[int] = None,
            num_layers: int = 60,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 3584,
            axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)):
        super(QwenImageTransformer2DModel, self).__init__()

        self.num_gaussians = num_gaussians
        self.logweights_channels = logweights_channels

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_means = nn.Linear(self.inner_dim, self.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(self.inner_dim, self.num_gaussians * self.logweights_channels)
        self.constant_logstd = constant_logstd

        if self.constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = self.inner_dim
            logstd_layers = []
            for _ in range(gm_num_logstd_layers - 1):
                logstd_layers.extend([
                    nn.SiLU(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.proj_out_logstds = nn.Sequential(
                *logstd_layers,
                nn.SiLU(),
                nn.Linear(in_dim, 1))

        self.gradient_checkpointing = False

    def init_weights(self):
        # Output layers
        constant_init(self.proj_out_means.to_empty(device='cpu'), val=0)
        rand_noise = torch.randn((self.num_gaussians * self.out_channels // self.logweights_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[:, None].expand(-1, self.logweights_channels).flatten())
        constant_init(self.proj_out_logweights.to_empty(device='cpu'), val=0)
        if self.constant_logstd is None:
            # logstd layers
            for m in self.proj_out_logstds:
                if isinstance(m, nn.Linear):
                    xavier_init(m.to_empty(device='cpu'), distribution='uniform')
            constant_init(self.proj_out_logstds[-1], val=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_shapes: Optional[List[Tuple[int, int, int]]] = None,
            txt_seq_lens: Optional[List[int]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            assert attention_kwargs is None or attention_kwargs.get('scale', None) is None

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        out_means = self.proj_out_means(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels).log_softmax(dim=-2)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(temb.detach()).reshape(bs, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1), float(self.constant_logstd))

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMQwenImageTransformer2DModel(_GMQwenImageTransformer2DModel):

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        assert self.patch_size * self.patch_size == self.logweights_channels

        self.init_weights(pretrained, pretrained_adapter)

        if autocast_dtype is not None:
            assert torch_dtype == 'float32'
        self.autocast_dtype = autocast_dtype

        self.use_lora = use_lora
        self.lora_target_modules = lora_target_modules
        self.lora_rank = lora_rank
        if self.use_lora:
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights='gaussian',
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
            )
            self.add_adapter(transformer_lora_config)

        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

        if checkpointing:
            self.enable_gradient_checkpointing()

    def init_weights(self, pretrained=None, pretrained_adapter=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            if 'proj_out.weight' in state_dict and state_dict['proj_out.weight'].size(0) == self.out_channels:
                state_dict['proj_out_means.weight'] = state_dict['proj_out.weight'][None].expand(
                    self.num_gaussians, -1, -1).reshape(self.num_gaussians * self.out_channels, -1)
                del state_dict['proj_out.weight']
            if 'proj_out.bias' in state_dict and state_dict['proj_out.bias'].size(0) == self.out_channels:
                state_dict['proj_out_means.bias'] = state_dict['proj_out.bias'][None].expand(
                    self.num_gaussians, -1).reshape(self.num_gaussians * self.out_channels)
                p2 = self.patch_size * self.patch_size
                rand_noise = torch.randn(
                    (self.num_gaussians * self.out_channels // p2),
                    dtype=state_dict['proj_out_means.bias'].dtype,
                    device=state_dict['proj_out_means.bias'].device) * 0.05
                state_dict['proj_out_means.bias'] += rand_noise[:, None].expand(-1, p2).flatten()
                del state_dict['proj_out.bias']
            if (self.constant_logstd is None
                    and 'proj_out_means.weight' in state_dict
                    and 'proj_out_means.bias' in state_dict):
                self.proj_out_logstds[-1].bias.data = torch.full_like(
                    self.proj_out_logstds[-1].bias.data, np.log(0.05))  # reduce the initial logstd
            if pretrained_adapter is not None:
                adapter_state_dict = _load_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                lora_state_dict = dict()
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k] = v
                    else:
                        state_dict[k] = v
                load_full_state_dict(self, state_dict, logger=logger, assign=True)
                if len(lora_state_dict) > 0:
                    self.load_lora_adapter(lora_state_dict, prefix=None)
                    self.fuse_lora()
                    self.unload_lora()
            else:
                load_full_state_dict(self, state_dict, logger=logger, assign=True)

    def patchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        return latents

    def unpatchify(self, gm):
        if self.patch_size > 1:
            bs, k, c, h, w = gm['means'].size()
            gm['means'] = gm['means'].reshape(
                bs, k, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
            gm['logweights'] = gm['logweights'].reshape(
                bs, k, 1, self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, 1, h * self.patch_size, w * self.patch_size)
        return gm

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(bs, c, h * w).permute(0, 2, 1)
        img_shapes = [[(1, h, w)]]
        if encoder_hidden_states_mask is not None:
            txt_seq_lens = encoder_hidden_states_mask.sum(dim=1)
            max_txt_seq_len = txt_seq_lens.max()
            encoder_hidden_states = encoder_hidden_states[:, :max_txt_seq_len]
            encoder_hidden_states_mask = encoder_hidden_states_mask[:, :max_txt_seq_len]
            txt_seq_lens = txt_seq_lens.tolist()
        else:
            txt_seq_lens = None

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                **kwargs)

        output['means'] = output['means'].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.out_channels, h, w)
        output['logweights'] = output['logweights'].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.logweights_channels, h, w)
        output['logstds'] = output['logstds'].unsqueeze(-1)  # (bs, 1, 1, 1, 1)
        return self.unpatchify(output)
