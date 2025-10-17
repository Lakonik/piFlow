# Copyright (c) 2025 Hansheng Chen

import sys
import torch
import mmcv

from copy import deepcopy
from functools import partial
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES

from . import GaussianFlow
from .piflow_policies import POLICY_CLASSES, GMFlowPolicy
from lakonlab.utils import module_eval


class PiFlowImitationBase(GaussianFlow):

    def __init__(self, *args, policy_type='GMFlow', policy_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert policy_type in POLICY_CLASSES, \
            f'Invalid policy: {policy_type}. Supported policies are {list(POLICY_CLASSES.keys())}.'
        self.policy_type = policy_type
        self.policy_class = partial(
            POLICY_CLASSES[policy_type], **policy_kwargs
        ) if policy_kwargs else POLICY_CLASSES[policy_type]

    def policy_rollout(
            self,
            x_t_start: torch.Tensor,  # (B, C, *, H, W)
            sigma_t_start: torch.Tensor,  # (B, 1, *, 1, 1)
            raw_t_start: torch.Tensor,  # (B, )
            raw_t_end: torch.Tensor,  # (B, )
            total_substeps: int,
            policy,
            seq_len=None):
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()
        raw_t_start = raw_t_start.reshape(num_batches, *((ndim - 1) * [1]))
        raw_t_end = raw_t_end.reshape(num_batches, *((ndim - 1) * [1]))

        delta_raw_t = raw_t_start - raw_t_end
        num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
        substep_size = delta_raw_t / num_substeps
        max_num_substeps = num_substeps.max()

        raw_t = raw_t_start
        sigma_t = sigma_t_start
        x_t = x_t_start

        for substep_id in range(max_num_substeps.item()):
            u = policy.u(x_t, sigma_t)
            
            raw_t_minus = (raw_t - substep_size).clamp(min=0)
            sigma_t_minus = self.timestep_sampler.warp_t(raw_t_minus, seq_len=seq_len)
            x_t_minus = x_t + u * (sigma_t_minus - sigma_t)
            
            active_mask = num_substeps > substep_id
            x_t = torch.where(active_mask, x_t_minus, x_t)
            sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
            raw_t = torch.where(active_mask, raw_t_minus, raw_t)

        x_t_end = x_t
        sigma_t_end = sigma_t
        t_end = sigma_t_end.flatten() * self.num_timesteps
        return x_t_end, sigma_t_end, t_end

    def policy_average_u(
            self,
            x_t_start: torch.Tensor,  # (B, C, *, H, W)
            sigma_t_start: torch.Tensor,  # (B, 1, *, 1, 1)
            raw_t_start: torch.Tensor,  # (B, )
            raw_t_end: torch.Tensor,  # (B, )
            total_substeps: int,
            policy,
            seq_len=None,
            eps=1e-4):
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()
        is_small_length = torch.round((raw_t_start - raw_t_end) * total_substeps) < 2
        pred_mean_u = pred_local_u = None
        if not is_small_length.all():  # mean velocity over the rollout length
            x_t_end, sigma_t_end, _ = self.policy_rollout(
                x_t_start, sigma_t_start, raw_t_start, raw_t_end, total_substeps,
                policy, seq_len=seq_len)
            pred_mean_u = (x_t_start - x_t_end) / (sigma_t_start - sigma_t_end).clamp(min=eps)
        if is_small_length.any():  # numerically stable local velocity
            pred_local_u = policy.u(x_t_start, sigma_t_start)
        if pred_mean_u is None:
            pred_u = pred_local_u
        elif pred_local_u is None:
            pred_u = pred_mean_u
        else:
            pred_u = torch.where(
                is_small_length.reshape(num_batches, *((ndim - 1) * [1])), pred_local_u, pred_mean_u)
        return pred_u

    @staticmethod
    def get_shape_info(x):
        x_t_dst_shape = x.size()
        bs = x_t_dst_shape[0]
        ndim = len(x_t_dst_shape)
        seq_len = x.shape[2:].numel()
        return ndim, bs, seq_len

    def piid_segment(
            self, teacher, x_t_src, raw_t_src, sigma_t_src, t_src, teacher_ratio, segment_size,
            kwargs, teacher_kwargs, get_x_t_dst=False):
        eps = self.train_cfg.get('eps', 1e-4)
        total_substeps = self.train_cfg.get('total_substeps', self.num_timesteps)
        num_intermediate_states = self.train_cfg.get('num_intermediate_states', 2)
        window_substeps = self.train_cfg.get('window_substeps', 0)

        device = x_t_src.device
        ndim, bs, seq_len = self.get_shape_info(x_t_src)
        if not isinstance(segment_size, torch.Tensor):
            segment_size = torch.tensor(
                [segment_size], dtype=torch.float32, device=device)

        # window size ∆τ ≈ window_substeps / total_substeps
        num_substeps = (segment_size * total_substeps).round().to(torch.long).clamp(min=1)
        substep_size = segment_size / num_substeps
        window_size = torch.minimum(window_substeps * substep_size, segment_size)

        raw_t_dst = raw_t_src - segment_size

        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        policy = self.policy_class(
            denoising_output, x_t_src, sigma_t_src, eps=eps)

        policy_detached = policy.detach()
        if isinstance(policy_detached, GMFlowPolicy):
            gm_dropout = self.train_cfg.get('gm_dropout', 0.0)
            policy_detached.dropout_(gm_dropout)

        # time sampling for scheduled trajectory mixing
        assert not self.timestep_sampler.logit_normal_enable
        student_intervals = torch.rand(
            (bs, num_intermediate_states), device=device
        ) * ((1 - teacher_ratio) * (segment_size - window_size).unsqueeze(-1))
        student_intervals = torch.sort(student_intervals, dim=-1)[0]
        student_intervals = torch.diff(student_intervals, dim=-1, prepend=torch.zeros((bs, 1), device=device))

        teacher_intervals = torch.rand((bs, num_intermediate_states - 1), device=device)
        teacher_intervals = torch.sort(teacher_intervals, dim=-1)[0]
        teacher_intervals = torch.diff(
            teacher_intervals, dim=-1,
            prepend=torch.zeros((bs, 1), device=device),
            append=torch.ones(
                (bs, 1), device=device)
        ) * (teacher_ratio * (segment_size - window_size).unsqueeze(-1))

        x_t = x_t_src
        raw_t = raw_t_src
        sigma_t = sigma_t_src

        all_pred_u = []
        all_tgt_u = []
        all_timesteps = []

        for teacher_step_id in range(num_intermediate_states):
            raw_t1 = (raw_t - student_intervals[:, teacher_step_id]).clamp(min=0)
            raw_t2 = (raw_t1 - teacher_intervals[:, teacher_step_id]).clamp(min=0)

            with torch.no_grad(), module_eval(teacher):
                x_t1, sigma_t1, t1 = self.policy_rollout(
                    x_t, sigma_t, raw_t, raw_t1, total_substeps,
                    policy_detached, seq_len=seq_len)
                tgt_u = teacher(return_u=True, x_t=x_t1, t=t1, **teacher_kwargs)
                all_tgt_u.append(tgt_u)
                all_timesteps.append(t1)

            pred_u = self.policy_average_u(
                x_t1, sigma_t1, raw_t1, raw_t2 - window_size, total_substeps,
                policy, seq_len=seq_len, eps=eps)
            all_pred_u.append(pred_u)

            sigma_t2 = self.timestep_sampler.warp_t(raw_t2, seq_len=seq_len).reshape(bs, *((ndim - 1) * [1]))
            x_t = x_t1 + tgt_u * (sigma_t2 - sigma_t1)
            raw_t = raw_t2
            sigma_t = sigma_t2

        loss_kwargs = dict(
            u_t_pred=torch.cat(all_pred_u, dim=0),
            u_t=torch.cat(all_tgt_u, dim=0),
            timesteps=torch.cat(all_timesteps, dim=0)
        )
        loss = self.flow_loss(loss_kwargs)

        if get_x_t_dst:
            with torch.no_grad():
                x_t_dst, _, _ = self.policy_rollout(
                    x_t, sigma_t, raw_t, raw_t_dst, total_substeps,
                    policy_detached, seq_len=seq_len)
        else:
            x_t_dst = None

        return loss, x_t_dst, raw_t_dst, policy

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t_src = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()  # h * w or t * h * w
        ori_dtype = x_t_src.dtype
        device = x_t_src.device
        x_t_src = x_t_src.float()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        total_substeps = cfg.get('total_substeps', self.num_timesteps)
        eps = cfg.get('eps', 1e-4)
        nfe = cfg['nfe']
        final_step_size_scale = max(cfg.get('final_step_size_scale', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + final_step_size_scale)

        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        if show_pbar:
            pbar = mmcv.ProgressBar(self.distill_steps)

        # ========== Main sampling loop ==========
        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            if is_final_step:
                segment_size = base_segment_size * final_step_size_scale
            else:
                segment_size = base_segment_size

            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(x_t_src, t_src, **kwargs)
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, eps=eps)
            if isinstance(policy, GMFlowPolicy) and not is_final_step:
                temperature = cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            x_t_dst, sigma_t_dst, t_dst = self.policy_rollout(
                x_t_src, sigma_t_src, raw_t_src, raw_t_dst, total_substeps,
                policy, seq_len=seq_len)

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t_src.to(ori_dtype)


@MODULES.register_module()
class PiFlowImitation(PiFlowImitationBase):

    def sample_t(self, num_batches, ndim, seq_len=None, device=None):
        eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']

        final_step_size_scale = max(self.train_cfg.get('final_step_size_scale', 1.0), eps)
        one_minus_final_scale = 1 - final_step_size_scale
        base_segment_size = 1 / (nfe - one_minus_final_scale)
        final_step_size = final_step_size_scale * base_segment_size

        raw_t = self.timestep_sampler(
            num_batches, warp_t=False, scale_t=False, device=device).clamp(min=eps)
        raw_t_src_idx = torch.ceil(
            raw_t / base_segment_size + one_minus_final_scale).clamp(min=1)
        if isinstance(nfe, torch.Tensor):
            raw_t_src_idx = torch.minimum(raw_t_src_idx, nfe)
        else:
            raw_t_src_idx = raw_t_src_idx.clamp(max=nfe)
        raw_t_src = ((raw_t_src_idx - one_minus_final_scale) * base_segment_size).clamp(min=eps, max=1)
        is_final_step = raw_t_src_idx == 1
        segment_size = torch.where(
            is_final_step, final_step_size, base_segment_size)

        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps
        return raw_t_src, sigma_t_src, t_src, segment_size

    def forward_train(self, x_0, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()  # h * w or t * h * w
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        raw_t_src, sigma_t_src, t_src, segment_size = self.sample_t(
            num_batches, ndim, seq_len=seq_len, device=device)
        noise = torch.randn_like(x_0)
        x_t_src, _, _ = self.sample_forward_diffusion(x_0, t_src, noise)

        loss_diffusion, _, _, _ = self.piid_segment(
            teacher, x_t_src, raw_t_src, sigma_t_src, t_src, teacher_ratio, segment_size,
            kwargs, teacher_kwargs)

        loss = loss_diffusion
        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(loss_diffusion=float(loss_diffusion))

        return loss, log_vars


@MODULES.register_module()
class PiFlowImitationDataFree(PiFlowImitationBase):

    is_multistep = True

    def forward_initialize(
            self, x_0, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        device = get_module_device(self)
        num_batches = x_0.size(0)  # x_0 is a dummy input

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        x_t_src = torch.randn_like(x_0)
        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        step_states = dict(
            step_id=0,
            terminate=False,
            detachable=True,
            teacher_ratio=teacher_ratio,
            x_t_src=x_t_src,
            raw_t_src=raw_t_src,
        )

        return step_states, log_vars

    def forward_train(
            self, x_0, step_states=None, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        step_id = step_states['step_id']
        teacher_ratio = step_states['teacher_ratio']
        x_t_src = step_states['x_t_src']
        raw_t_src = step_states['raw_t_src']

        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']
        final_step_size_scale = max(self.train_cfg.get('final_step_size_scale', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + final_step_size_scale)
        is_final_step = step_id == nfe - 1
        if is_final_step:
            segment_size = base_segment_size * final_step_size_scale
        else:
            segment_size = base_segment_size

        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        step_loss_diffusion, x_t_dst, raw_t_dst, _ = self.piid_segment(
            teacher, x_t_src, raw_t_src, sigma_t_src, t_src, teacher_ratio, segment_size,
            kwargs, teacher_kwargs, get_x_t_dst=True)

        if step_id < nfe - 1:
            step_states.update(
                step_id=step_id + 1,
                x_t_src=x_t_dst,
                raw_t_src=raw_t_dst)
        else:
            step_states.update(terminate=True)

        loss_diffusion = step_loss_diffusion * segment_size  # Weighing by segment size
        loss = loss_diffusion
        log_vars = {k: v * segment_size for k, v in self.flow_loss.log_vars.items()}
        log_vars.update({
            'loss_diffusion': float(loss_diffusion),
            f'loss_diffusion_step{step_id}': float(step_loss_diffusion)
        })

        return loss, log_vars, step_states

    def forward(self, x_0=None, return_step_states=False, **kwargs):
        if return_step_states:
            return self.forward_initialize(x_0=x_0, **kwargs)
        else:
            return super().forward(x_0=x_0, **kwargs)
