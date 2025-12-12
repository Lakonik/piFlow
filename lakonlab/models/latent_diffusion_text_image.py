# Copyright (c) 2025 Hansheng Chen

import torch
import inspect

from copy import deepcopy
from mmgen.models.builder import MODELS, build_module

from .base_diffusion import BaseDiffusion
from lakonlab.utils import rgetattr


@MODELS.register_module()
class LatentDiffusionTextImage(BaseDiffusion):

    def __init__(self,
                 *args,
                 vae=None,
                 text_encoder=None,
                 use_condition_latents=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = build_module(vae) if vae is not None else None
        self.text_encoder = build_module(text_encoder) if text_encoder is not None else None
        self.use_condition_latents = use_condition_latents

    def _prepare_train_minibatch_diffusion_args(self, data):
        if 'prompt_embed_kwargs' in data:
            prompt_embed_kwargs = data['prompt_embed_kwargs']
        elif 'prompt_kwargs' in data:
            assert self.text_encoder is not None, 'Text encoder must be provided for encoding text to embeddings.'
            prompt_embed_kwargs = self.text_encoder(**data['prompt_kwargs'])
        else:
            raise ValueError('Either `prompt_embed_kwargs` or `prompt_kwargs` should be provided in the input data.')

        if self.use_condition_latents and ('condition_latents' in data or 'condition_images' in data):
            if 'condition_latents' in data:
                condition_latents = data['condition_latents']
            else:
                assert self.vae is not None, 'VAE must be provided for encoding images to latents.'
                if hasattr(self.vae, 'dtype'):
                    vae_dtype = self.vae.dtype
                else:
                    vae_dtype = next(self.vae.parameters()).dtype
                kwargs = dict()
                if 'sample_mode' in inspect.signature(rgetattr(self.vae, 'encode')).parameters:
                    kwargs.update(sample_mode='argmax')
                condition_latents = self.vae.encode(
                    (data['condition_images'] * 2 - 1).to(vae_dtype), **kwargs).float()
            prompt_embed_kwargs['condition_latents'] = self.patchify(condition_latents)

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

        v = next(iter(prompt_embed_kwargs.values()))
        bs = v.size(0)
        device = v.device

        diffusion_args = (self.patchify(latents), )
        diffusion_kwargs = prompt_embed_kwargs.copy()

        distilled_guidance_scale = self.train_cfg.get('distilled_guidance_scale', None)
        if distilled_guidance_scale is not None:
            distilled_guidance_scale = torch.full(
                (bs,), distilled_guidance_scale, dtype=torch.float32, device=device)
            diffusion_kwargs.update(guidance=distilled_guidance_scale)

        return diffusion_args, diffusion_kwargs, prompt_embed_kwargs, bs, device

    def _prepare_train_minibatch_teacher_args(self, data, prompt_embed_kwargs, bs, device):
        teacher_guidance_scale = self.train_cfg.get('teacher_guidance_scale', None)
        teacher_use_guidance = (teacher_guidance_scale is not None
                                and teacher_guidance_scale != 0.0 and teacher_guidance_scale != 1.0)
        if teacher_use_guidance:
            if 'negative_prompt_embed_kwargs' in data:
                negative_prompt_embed_kwargs = data['negative_prompt_embed_kwargs']
            elif 'negative_prompt_kwargs' in data:
                negative_prompt_embed_kwargs = self.text_encoder(**data['negative_prompt_kwargs'])
            else:
                raise ValueError(
                    'Either `negative_prompt_embed_kwargs` or `negative_prompt_kwargs` should be provided in the '
                    'input data for classifier-free guidance.')
            if 'condition_latents' in prompt_embed_kwargs:
                negative_prompt_embed_kwargs['condition_latents'] = prompt_embed_kwargs['condition_latents']
            teacher_kwargs = {
                k: torch.cat([negative_prompt_embed_kwargs[k], v], dim=0)
                for k, v in prompt_embed_kwargs.items()}
            teacher_kwargs.update(guidance_scale=teacher_guidance_scale)
        else:
            teacher_kwargs = prompt_embed_kwargs.copy()

        teacher_distilled_guidance_scale = self.train_cfg.get('teacher_distilled_guidance_scale', None)
        if teacher_distilled_guidance_scale is not None:
            teacher_distilled_guidance_scale = torch.full(
                (bs * 2,) if teacher_use_guidance else (bs,),
                teacher_distilled_guidance_scale, dtype=torch.float32, device=device)
            teacher_kwargs.update(guidance=teacher_distilled_guidance_scale)

        return teacher_kwargs

    def _prepare_train_minibatch_args(self, data, running_status=None):
        diffusion_args, diffusion_kwargs, prompt_embed_kwargs, bs, device = \
            self._prepare_train_minibatch_diffusion_args(data)
        parameters = inspect.signature(rgetattr(self.diffusion, 'forward_train')).parameters
        if 'running_status' in parameters:
            diffusion_kwargs['running_status'] = running_status

        if 'teacher' in parameters and 'teacher_kwargs' in parameters and self.teacher is not None:
            teacher_kwargs = self._prepare_train_minibatch_teacher_args(
                data, prompt_embed_kwargs, bs, device)

            diffusion_kwargs.update(
                teacher=self.teacher,
                teacher_kwargs=teacher_kwargs)

        return bs, diffusion_args, diffusion_kwargs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        if 'prompt_embed_kwargs' in data:
            prompt_embed_kwargs = data['prompt_embed_kwargs']
        elif 'prompt_kwargs' in data:
            assert self.text_encoder is not None, 'Text encoder must be provided for encoding text to embeddings.'
            prompt_embed_kwargs = self.text_encoder(**data['prompt_kwargs'])
        else:
            raise ValueError('Either `prompt_embed_kwargs` or `prompt_kwargs` should be provided in the input data.')

        if self.use_condition_latents and ('condition_latents' in data or 'condition_images' in data):
            if 'condition_latents' in data:
                condition_latents = data['condition_latents']
            else:
                assert self.vae is not None, 'VAE must be provided for encoding images to latents.'
                if hasattr(self.vae, 'dtype'):
                    vae_dtype = self.vae.dtype
                else:
                    vae_dtype = next(self.vae.parameters()).dtype
                kwargs = dict()
                if 'sample_mode' in inspect.signature(rgetattr(self.vae, 'encode')).parameters:
                    kwargs.update(sample_mode='argmax')
                condition_latents = self.vae.encode(
                    (data['condition_images'] * 2 - 1).to(vae_dtype), **kwargs).float()
            prompt_embed_kwargs['condition_latents'] = self.patchify(condition_latents)

        v = next(iter(prompt_embed_kwargs.values()))
        bs = v.size(0)
        device = v.device

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            use_guidance = guidance_scale != 0.0 and guidance_scale != 1.0
            if use_guidance:
                if 'negative_prompt_embed_kwargs' in data:
                    negative_prompt_embed_kwargs = data['negative_prompt_embed_kwargs']
                elif 'negative_prompt_kwargs' in data:
                    negative_prompt_embed_kwargs = self.text_encoder(**data['negative_prompt_kwargs'])
                else:
                    raise ValueError(
                        'Either `negative_prompt_embed_kwargs` or `negative_prompt_kwargs` should be provided in the '
                        'input data for classifier-free guidance.')
                if self.use_condition_latents:
                    negative_prompt_embed_kwargs['condition_latents'] = prompt_embed_kwargs['condition_latents']
                kwargs = {
                    k: torch.cat([negative_prompt_embed_kwargs[k], v], dim=0)
                    for k, v in prompt_embed_kwargs.items()}
            else:
                kwargs = prompt_embed_kwargs.copy()
            distilled_guidance_scale = cfg.get('distilled_guidance_scale', None)
            if distilled_guidance_scale is not None:
                distilled_guidance_scale = torch.full(
                    (bs * 2,) if use_guidance else (bs,),
                    distilled_guidance_scale, dtype=torch.float32, device=device)
                kwargs.update(guidance=distilled_guidance_scale)

            if 'noise' in data:
                noise = data['noise']
            else:
                latent_size = cfg['latent_size']
                noise = torch.randn((bs, *latent_size), device=device)
            noise = self.patchify(noise)
            latents_out = diffusion(
                noise=noise,
                guidance_scale=guidance_scale,
                test_cfg_override=test_cfg_override,
                **kwargs)
            latents_out = self.unpatchify(latents_out)

            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents_out = latents_out.to(vae_dtype)

            out_images = (self.vae.decode(latents_out).float() / 2 + 0.5).clamp(min=0, max=1)

            return dict(num_samples=bs, pred_imgs=out_images)
