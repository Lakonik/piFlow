# Copyright (c) 2025 Hansheng Chen

import torch

from abc import abstractmethod
from copy import deepcopy
from accelerate import init_empty_weights
from mmgen.models.builder import build_module
from mmgen.utils import get_root_logger

from .base import BaseModel
from lakonlab.utils import clone_params, rgetattr, tie_untrained_submodules


class BaseDiffusion(BaseModel):
    """Base class providing the common training interface for diffusion models. Optionally supports:
    - Teacher model for distillation training
    - EMA version of the diffusion model
    - Multi-step diffusion training
    - Image/video patching for patch-wise GMFlow
    """

    def __init__(self,
                 diffusion=dict(type='GaussianFlow'),
                 diffusion_use_ema=False,
                 tie_ema=True,
                 teacher=None,
                 tie_teacher=False,
                 patch_size=1,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        # order matters: teacher must be built before diffusion for FSDP tying
        if teacher is not None and not inference_only:
            teacher.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.teacher = build_module(teacher)
        else:
            self.teacher = None

        diffusion.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.diffusion = build_module(diffusion)
        if self.teacher is not None and tie_teacher:
            tie_untrained_submodules(self.diffusion, self.teacher, tie_tgt_lora_base_layer=True)

        self.patch_size = patch_size

        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            if inference_only:
                self.diffusion_ema = self.diffusion
            else:
                diffusion_ema = deepcopy(diffusion)
                if isinstance(diffusion_ema.get('denoising', None), dict):
                    diffusion_ema['denoising'].pop('pretrained', None)
                with init_empty_weights():
                    self.diffusion_ema = build_module(diffusion_ema)
                if tie_ema:
                    tie_untrained_submodules(self.diffusion_ema, self.diffusion)
                clone_params(self.diffusion_ema, self.diffusion)

        self.train_cfg = dict() if train_cfg is None else deepcopy(train_cfg)
        self.test_cfg = dict() if test_cfg is None else deepcopy(test_cfg)

    def patchify(self, x):
        if isinstance(self.patch_size, int) and self.patch_size == 1:
            return x
        if x.dim() == 4:
            if isinstance(self.patch_size, int):
                ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 2
                ph, pw = self.patch_size
            bs, c, h, w = x.size()
            x = x.reshape(
                bs, c, h // ph, ph, w // pw, pw
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * ph * pw, h // ph, w // pw)
        elif x.dim() == 5:
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 3
                pt, ph, pw = self.patch_size
            bs, c, t, h, w = x.size()
            x = x.reshape(
                bs, c, t // pt, pt, h // ph, ph, w // pw, pw
            ).permute(
                0, 1, 3, 5, 7, 2, 4, 6
            ).reshape(
                bs, c * pt * ph * pw, t // pt, h // ph, w // pw)
        else:
            raise ValueError(f'Unsupported input dimension {x.dim()}. Expected 4 or 5 dimensions.')
        return x

    def unpatchify(self, x):
        if isinstance(self.patch_size, int) and self.patch_size == 1:
            return x
        if x.dim() == 4:
            if isinstance(self.patch_size, int):
                ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 2
                ph, pw = self.patch_size
            bs, c, h, w = x.size()
            x = x.reshape(
                bs, c // (ph * pw), ph, pw, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c // (ph * pw), h * ph, w * pw)
        elif x.dim() == 5:
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                assert len(self.patch_size) == 3
                pt, ph, pw = self.patch_size
            bs, c, t, h, w = x.size()
            x = x.reshape(
                bs, c // (pt * ph * pw), pt, ph, pw, t, h, w
            ).permute(
                0, 1, 5, 2, 6, 3, 7, 4
            ).reshape(
                bs, c // (pt * ph * pw), t * pt, h * ph, w * pw)
        else:
            raise ValueError(f'Unsupported input dimension {x.dim()}. Expected 4 or 5 dimensions.')
        return x

    @abstractmethod
    def _prepare_train_step_args(self, data, running_status=None):
        """
        Prepare the arguments for the training step.

        Args:
            data (dict): The input data for the training step.
            running_status (dict): The running status for the training step.

        Returns:
            tuple: A tuple containing the batch size, diffusion arguments, and diffusion keyword arguments.
        """

    def train_step_single(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_step_args(data, running_status)
        is_multistep = rgetattr(self.diffusion, 'is_multistep', False)

        if is_multistep:
            step_states, log_vars = self.diffusion(*diffusion_args, return_step_states=True, **diffusion_kwargs)
            loss = 0
            step_id = 0
            while not step_states['terminate']:
                step_loss, step_log_vars, step_states = self.diffusion(
                    *diffusion_args, return_loss=True, step_states=step_states, **diffusion_kwargs)
                if step_states['detachable']:
                    step_loss.backward() if loss_scaler is None else loss_scaler.scale(step_loss).backward()
                    step_loss.detach_()
                loss = loss + step_loss
                for k, v in step_log_vars.items():
                    if k in log_vars:
                        log_vars[k] += v
                    else:
                        log_vars[k] = v
                step_id += 1

        else:
            loss, log_vars = self.diffusion(*diffusion_args, return_loss=True, **diffusion_kwargs)

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward() if loss_scaler is None else loss_scaler.scale(loss).backward()

        return log_vars, bs

    @abstractmethod
    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        """Perform a validation step.

        Args:
            data (dict): The input data for the validation step.
            test_cfg_override (dict): Override configuration for the test.

        Returns:
            dict: A dictionary containing the number of samples and predicted outputs.
        """
