# Copyright (c) 2025 Hansheng Chen

import torch

from copy import deepcopy
from accelerate import init_empty_weights
from mmgen.models.builder import MODELS, build_module
from mmgen.utils import get_root_logger

from .base import BaseModel
from lakonlab.utils import tie_or_copy_untrained_params, materialize_meta_states, tie_untrained_submodules


@MODELS.register_module()
class Diffusion2D(BaseModel):

    def __init__(self,
                 diffusion=dict(type='GMFlow'),
                 diffusion_use_ema=False,
                 tie_ema=True,
                 pretrained=None,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        diffusion.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            if inference_only:
                self.diffusion_ema = self.diffusion
            else:
                diffusion_ema = deepcopy(diffusion)
                if isinstance(diffusion_ema.get('denoising', None), dict):
                    diffusion_ema['denoising'].pop('pretrained', None)
                with init_empty_weights():
                    self.diffusion_ema = build_module(diffusion_ema)  # deepcopy doesn't work due to the monkey patch lora
                if tie_ema:
                    tie_untrained_submodules(self.diffusion_ema, self.diffusion)
                else:
                    tie_or_copy_untrained_params(self.diffusion_ema, self.diffusion, copy=True)
                materialize_meta_states(self.diffusion_ema)

        self.pretrained = pretrained

        if self.pretrained is not None:
            self.load_checkpoint(self.pretrained, map_location='cpu', strict=False, logger=get_root_logger())

        self.train_cfg = dict() if train_cfg is None else deepcopy(train_cfg)
        self.test_cfg = dict() if test_cfg is None else deepcopy(test_cfg)

    def train_step_single(self, data, loss_scaler=None, running_status=None):
        bs = data['x'].size(0)

        loss, log_vars = self.diffusion(
            data['x'].reshape(bs, 2, 1, 1),
            return_loss=True)

        loss.backward() if loss_scaler is None else loss_scaler.scale(loss).backward()

        return log_vars, bs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        bs = data['x'].size(0)
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            if 'noise' in data:
                noise = data['noise'].reshape(bs, 2, 1, 1)
            else:
                noise = torch.randn((bs, 2, 1, 1), device=data['x'].device)
            x_out = diffusion(
                noise=noise,
                test_cfg_override=test_cfg_override)

            return dict(num_samples=bs, pred_x=x_out)
