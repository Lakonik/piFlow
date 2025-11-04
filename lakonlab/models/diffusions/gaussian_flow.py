# Copyright (c) 2025 Hansheng Chen

import sys
import inspect
import torch
import torch.nn as nn
import mmcv
import diffusers

from copy import deepcopy
from mmcv.runner.fp16_utils import force_fp32
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module

from . import schedulers


@torch.jit.script
def guidance_jit(pos_mean, neg_mean, guidance_scale: float, orthogonal: bool = False):
    bias = (pos_mean - neg_mean) * (guidance_scale - 1)
    if orthogonal:
        dim = list(range(1, pos_mean.dim()))
        bias = bias - (bias * pos_mean).mean(
            dim=dim, keepdim=True
        ) / (pos_mean * pos_mean).mean(dim=dim, keepdim=True).clamp(min=1e-6) * pos_mean
    return bias


@MODULES.register_module()
class GaussianFlow(nn.Module):

    def __init__(self,
                 denoising=None,
                 flow_loss=None,
                 num_timesteps=1000,
                 timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0),
                 flip_model_timesteps=False,
                 denoising_mean_mode='U',
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        # build denoising module in this function
        self.num_timesteps = num_timesteps
        self.denoising = build_module(denoising) if isinstance(denoising, dict) else denoising
        self.denoising_mean_mode = denoising_mean_mode

        self.flip_model_timesteps = flip_model_timesteps
        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

        # build sampler
        self.timestep_sampler = build_module(
            timestep_sampler,
            default_args=dict(num_timesteps=num_timesteps))
        self.flow_loss = build_module(flow_loss) if flow_loss is not None else None

    def forward_transition(
            self, x_t_src, t_src=None, t_tgt=None, sigma_src=None, sigma_tgt=None, eps=1e-6):
        if sigma_src is None:
            if not isinstance(t_src, torch.Tensor):
                t_src = torch.tensor(t_src, device=x_t_src.device)
            t_src = t_src.reshape(*t_src.size(), *((x_t_src.dim() - t_src.dim()) * [1]))
            sigma_src = t_src / self.num_timesteps

        if sigma_tgt is None:
            if not isinstance(t_tgt, torch.Tensor):
                t_tgt = torch.tensor(t_tgt, device=x_t_src.device)
            t_tgt = t_tgt.reshape(*t_tgt.size(), *((x_t_src.dim() - t_tgt.dim()) * [1]))
            sigma_tgt = t_tgt / self.num_timesteps

        alpha_src = 1 - sigma_src
        alpha_tgt = 1 - sigma_tgt

        scale_trans = alpha_tgt / alpha_src.clamp(min=eps)
        var_trans = sigma_tgt ** 2 - (scale_trans * sigma_src) ** 2
        return dict(mean=x_t_src * scale_trans, var=var_trans), scale_trans

    def sample_forward_transition(self, x_t_src, noise, t_src=None, t_tgt=None, sigma_src=None, sigma_tgt=None):
        trans_g = self.forward_transition(
            x_t_src, t_src=t_src, t_tgt=t_tgt, sigma_src=sigma_src, sigma_tgt=sigma_tgt)[0]
        return trans_g['mean'] + noise * trans_g['var'].sqrt()

    def sample_forward_diffusion(self, x_0, t, noise):
        if t.dim() == 0:
            t = t.expand(x_0.size(0))
        std = t.reshape(*t.size(), *((x_0.dim() - t.dim()) * [1])) / self.num_timesteps
        mean = 1 - std
        return x_0 * mean + noise * std, mean, std

    def pred(self, x_t=None, t=None, **kwargs):
        ori_dtype = x_t.dtype
        if hasattr(self.denoising, 'dtype'):
            denoising_dtype = self.denoising.dtype
        else:
            denoising_dtype = next(self.denoising.parameters()).dtype
        x_t = x_t.to(denoising_dtype)
        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        if self.flip_model_timesteps:
            t = self.num_timesteps - t
        output = self.denoising(x_t, t, **kwargs)
        if isinstance(output, dict):
            output = {k: v.to(ori_dtype) for k, v in output.items()}
        else:
            output = output.to(ori_dtype)
        return output

    @force_fp32()
    def loss(self, denoising_output, x_0, noise, t, pred_mask=None):
        if self.denoising_mean_mode.upper() == 'U':
            if isinstance(denoising_output, dict):
                loss_kwargs = denoising_output
            elif isinstance(denoising_output, torch.Tensor):
                loss_kwargs = dict(u_t_pred=denoising_output)
            else:
                raise AttributeError('Unknown denoising output type '
                                     f'[{type(denoising_output)}].')
            loss_kwargs.update(u_t=noise - x_0)
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')
        loss_kwargs.update(
            x_0=x_0,
            noise=noise,
            timesteps=t,
            weight=pred_mask.float() if pred_mask is not None else None)

        return self.flow_loss(loss_kwargs)

    def forward_train(self, x_0, **kwargs):
        device = get_module_device(self)

        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()  # h * w or t * h * w

        t = self.timestep_sampler(num_batches, seq_len=seq_len, device=device)

        noise = torch.randn_like(x_0)
        x_t, _, _ = self.sample_forward_diffusion(x_0, t, noise)

        denoising_output = self.pred(x_t, t, **kwargs)
        loss = self.loss(denoising_output, x_0, noise, t)
        log_vars = self.flow_loss.log_vars
        log_vars.update(loss_diffusion=float(loss))

        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=1.0,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t.size(0)
        ori_dtype = x_t.dtype
        x_t = x_t.float()

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        sampler = cfg.get('sampler', 'FlowEulerODE')
        sampler_class = getattr(diffusers.schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            sampler_class = getattr(schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            raise AttributeError(f'Cannot find sampler [{sampler}].')

        sampler_kwargs = cfg.get('sampler_kwargs', {})
        signatures = inspect.signature(sampler_class).parameters.keys()
        for key in ['shift', 'use_dynamic_shifting', 'base_seq_len', 'max_seq_len', 'base_logshift', 'max_logshift']:
            if key in signatures and key not in sampler_kwargs:
                sampler_kwargs[key] = cfg.get(key, getattr(self.timestep_sampler, key))
        if 'flow_shift' in signatures and 'use_flow_sigmas' in signatures:
            sampler_kwargs['prediction_type'] = 'flow_prediction'
            sampler_kwargs['use_flow_sigmas'] = True
            if 'flow_shift' not in sampler_kwargs:
                sampler_kwargs['flow_shift'] = cfg.get('shift', self.timestep_sampler.shift)
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)

        num_timesteps = cfg.get('num_timesteps', self.num_timesteps)
        guidance_interval = cfg.get('guidance_interval', [0, self.num_timesteps])
        orthogonal_guidance = cfg.get('orthogonal_guidance', False)
        use_guidance = guidance_scale > 1.0

        set_timesteps_signatures = inspect.signature(sampler.set_timesteps).parameters.keys()
        if 'seq_len' in set_timesteps_signatures:
            seq_len = x_t.shape[2:].numel()  # h * w or t * h * w
            sampler.set_timesteps(num_timesteps, seq_len=seq_len, device=x_t.device)
        else:
            sampler.set_timesteps(num_timesteps, device=x_t.device)

        timesteps = sampler.timesteps

        if show_pbar:
            pbar = mmcv.ProgressBar(len(timesteps))

        for t in timesteps:
            x_t_input = x_t
            _kwargs = kwargs
            if use_guidance:
                guidance_active = guidance_interval[0] <= t <= guidance_interval[1]
                if guidance_active:
                    x_t_input = torch.cat([x_t_input, x_t_input], dim=0)
                else:
                    _kwargs = {
                        k: v[num_batches:] if isinstance(v, torch.Tensor) and v.size(0) == 2 * num_batches else v
                        for k, v in kwargs.items()}

            denoising_output = self.pred(x_t_input, t, **_kwargs)

            if use_guidance and guidance_active:
                mean_neg, mean_pos = denoising_output.chunk(2, dim=0)
                bias = guidance_jit(mean_pos, mean_neg, guidance_scale, orthogonal_guidance)
                denoising_output = mean_pos + bias

            x_t = sampler.step(denoising_output, t, x_t, return_dict=False)[0]
            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t.to(ori_dtype)

    def forward_u(self, x_t=None, t=None, guidance_scale=1.0, test_cfg_override=dict(), **kwargs):
        ori_dtype = x_t.dtype
        x_t = x_t.float()
        num_batches = x_t.size(0)

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        orthogonal_guidance = cfg.get('orthogonal_guidance', False)
        guidance_interval = cfg.get('guidance_interval', [0, self.num_timesteps])

        use_guidance = guidance_scale > 1.0

        x_t_input = x_t
        t_input = t
        if use_guidance:
            x_t_input = torch.cat([x_t_input, x_t_input], dim=0)
            t_input = torch.cat([t_input, t_input], dim=0)

        denoising_output = self.pred(x_t_input, t_input, **kwargs)

        if use_guidance:
            mean_neg, mean_pos = denoising_output.chunk(2, dim=0)
            bias = guidance_jit(mean_pos, mean_neg, guidance_scale, orthogonal_guidance)
            if guidance_interval[0] > 0 or guidance_interval[1] < self.num_timesteps:
                guidance_active = ((t >= guidance_interval[0]) & (t <= guidance_interval[1])).reshape(
                    [num_batches] + [1] * (bias.dim() - 1))
                bias = bias.masked_fill(~guidance_active, 0.0)
            denoising_output = mean_pos + bias

        return denoising_output.to(ori_dtype)

    def forward(
            self,
            x_0=None,
            return_loss=False,
            return_u=False,
            return_denoising_output=False,
            **kwargs):
        if return_loss:
            return self.forward_train(x_0, **kwargs)
        elif return_u:
            return self.forward_u(**kwargs)
        elif return_denoising_output:
            return self.pred(**kwargs)
        else:
            return self.forward_test(x_0, **kwargs)
