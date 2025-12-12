# Copyright (c) 2025 Hansheng Chen

import inspect
import torch

from copy import deepcopy
from mmgen.models.builder import MODELS, build_module

from .base_diffusion import BaseDiffusion
from lakonlab.utils import rgetattr


@MODELS.register_module()
class LatentDiffusionClassImage(BaseDiffusion):

    def __init__(self,
                 *args,
                 vae=dict(type='PretrainedVAE'),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = build_module(vae) if vae is not None else None

    def _prepare_train_minibatch_args(self, data, running_status=None):
        if 'latents' in data:
            latents = data['latents']
        elif 'images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding images to latents.'
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents = self.vae.encode((data['images'] * 2 - 1).to(vae_dtype)).float()
        else:
            raise ValueError('Either `latents` or `images` should be provided in the input data.')

        labels = data['labels']
        bs = labels.size(0)

        diffusion_args = (self.patchify(latents), )

        prob_class = self.train_cfg.get('prob_class', 1.0)
        if prob_class < 1.0:
            labels = torch.where(
                torch.rand_like(labels, dtype=torch.float32) < prob_class,
                labels, data['negative_labels'])

        diffusion_kwargs = dict(class_labels=labels)

        parameters = inspect.signature(rgetattr(self.diffusion, 'forward_train')).parameters

        if 'running_status' in parameters:
            diffusion_kwargs['running_status'] = running_status

        if 'teacher' in parameters and 'teacher_kwargs' in parameters and self.teacher is not None:
            teacher_guidance_scale = self.train_cfg.get('teacher_guidance_scale', None)
            teacher_use_guidance = (teacher_guidance_scale is not None
                                    and teacher_guidance_scale != 0.0 and teacher_guidance_scale != 1.0)
            if teacher_use_guidance:
                teacher_kwargs = dict(class_labels=torch.cat([data['negative_labels'], labels], dim=0))
                teacher_kwargs.update(guidance_scale=teacher_guidance_scale)
            else:
                teacher_kwargs = dict(class_labels=labels)

            diffusion_kwargs.update(
                teacher=self.teacher,
                teacher_kwargs=teacher_kwargs)

        return bs, diffusion_args, diffusion_kwargs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        bs = len(data['labels'])
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            class_labels = data['labels']

            if guidance_scale != 0.0 and guidance_scale != 1.0:
                class_labels = torch.cat([data['negative_labels'], class_labels], dim=0)

            if 'noise' in data:
                noise = data['noise']
            else:
                latent_size = cfg['latent_size']
                noise = torch.randn((bs, *latent_size), device=data['labels'].device)
            noise = self.patchify(noise)
            latents_out = diffusion(
                noise=noise,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                test_cfg_override=test_cfg_override)
            latents_out = self.unpatchify(latents_out)

            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents_out = latents_out.to(vae_dtype)

            out_images = (self.vae.decode(latents_out).float() / 2 + 0.5).clamp(min=0, max=1)

            return dict(num_samples=bs, pred_imgs=out_images)
