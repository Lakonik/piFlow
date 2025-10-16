# Copyright (c) 2025 Hansheng Chen

import numpy as np
import torch

from typing import Dict, List, Optional, Union, Any, Callable
from functools import partial
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.utils import is_torch_xla_available
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    QwenImagePipeline, calculate_shift, retrieve_timesteps, QwenImagePipelineOutput)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from lakonlab.models.diffusions.piflow_policies import POLICY_CLASSES
from .piflow_loader import PiFlowLoaderMixin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def retrieve_raw_timesteps(
    num_inference_steps: int,
    total_substeps: int,
    final_step_size_scale: float
):
    r"""
    Retrieve the raw times and the number of substeps for each inference step.

    Args:
        num_inference_steps (`int`):
            Number of inference steps.
        total_substeps (`int`):
            Total number of substeps (e.g., 128).
        final_step_size_scale (`float`):
            Scale for the final step size (e.g., 0.5).

    Returns:
        `Tuple[List[float], List[int], int]`: A tuple where the first element is the raw timestep schedule, the second
        element is the number of substeps for each inference step, and the third element is the rounded total number of
        substeps.
    """
    base_segment_size = 1 / (num_inference_steps - 1 + final_step_size_scale)
    raw_timesteps = []
    num_inference_substeps = []
    _raw_t = 1.0
    for i in range(num_inference_steps):
        if i < num_inference_steps - 1:
            segment_size = base_segment_size
        else:
            segment_size = base_segment_size * final_step_size_scale
        _num_inference_substeps = max(round(segment_size * total_substeps), 1)
        num_inference_substeps.append(_num_inference_substeps)
        raw_timesteps.extend(np.linspace(
            _raw_t, _raw_t - segment_size, _num_inference_substeps, endpoint=False).clip(min=0.0).tolist())
        _raw_t = _raw_t - segment_size
    total_substeps = sum(num_inference_substeps)
    return raw_timesteps, num_inference_substeps, total_substeps


class PiQwenImagePipeline(QwenImagePipeline, PiFlowLoaderMixin):
    r"""
    The policy-based QwenImage pipeline for text-to-image generation.

    Reference: https://arxiv.org/abs/2510.14974

    Args:
        transformer ([`QwenImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`QwenTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        policy_type (`str`, *optional*, defaults to `"GMFlow"`):
            The type of flow policy to use. Currently supports `"GMFlow"` and `"DX"`.
        policy_kwargs (`Dict`, *optional*):
            Additional keyword arguments to pass to the policy class.
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: QwenImageTransformer2DModel,
        policy_type: str = 'GMFlow',
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            transformer,
        )
        assert policy_type in POLICY_CLASSES, f'Invalid policy: {policy_type}. Supported policies are {list(POLICY_CLASSES.keys())}.'
        self.policy_type = policy_type
        self.policy_class = partial(
            POLICY_CLASSES[policy_type], **policy_kwargs
        ) if policy_kwargs else POLICY_CLASSES[policy_type]

    def _unpack_gm(self, gm, height, width, num_channels_latents, patch_size=2, gm_patch_size=1):
        c = num_channels_latents * patch_size * patch_size
        h = (int(height) // (self.vae_scale_factor * patch_size))
        w = (int(width) // (self.vae_scale_factor * patch_size))
        bs = gm['means'].size(0)
        k = self.transformer.num_gaussians
        scale = patch_size // gm_patch_size
        gm['means'] = gm['means'].reshape(
            bs, h, w, k, c // (scale * scale), scale, scale
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, c // (scale * scale), h * scale, w * scale)
        gm['logweights'] = gm['logweights'].reshape(
            bs, h, w, k, 1, scale, scale
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, 1, h * scale, w * scale)
        gm['logstds'] = gm['logstds'].reshape(bs, 1, 1, 1, 1)
        return gm

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, patch_size=1, target_patch_size=2):
        scale = target_patch_size // patch_size
        latents = latents.view(
            batch_size,
            num_channels_latents * patch_size * patch_size,
            height // target_patch_size, scale, width // target_patch_size, scale)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (height // target_patch_size) * (width // target_patch_size),
            num_channels_latents * target_patch_size * target_patch_size)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor, patch_size=2, target_patch_size=1):
        batch_size, num_patches, channels = latents.shape
        scale = patch_size // target_patch_size

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = (int(height) // (vae_scale_factor * patch_size))
        width = (int(width) // (vae_scale_factor * patch_size))

        latents = latents.view(
            batch_size, height, width, channels // (scale * scale), scale, scale)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (scale * scale), height * scale, width * scale)

        return latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        total_substeps: int = 128,
        final_step_size_scale: float = 0.5,
        temperature: Union[float, str] = 'auto',
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            total_substeps (`int`, *optional*, defaults to 128):
                The total number of substeps for policy-based flow integration.
            final_step_size_scale (`float`, *optional*, defaults to 0.5):
                The scale for the final step size.
            temperature (`float` or `"auto"`, *optional*, defaults to `"auto"`):
                The tmperature parameter for the flow policy.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare prompt embeddings
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        # 5. Prepare timesteps
        raw_timesteps, num_inference_substeps, total_substeps = retrieve_raw_timesteps(
            num_inference_steps, total_substeps, final_step_size_scale)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=raw_timesteps,
            mu=mu,
        )
        assert len(timesteps) == total_substeps
        self._num_timesteps = total_substeps

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        timestep_id = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                if self.interrupt:
                    continue

                t_src = timesteps[timestep_id]
                sigma_t_src = t_src / self.scheduler.config.num_train_timesteps
                is_final_step = i == (num_inference_steps - 1)

                self._current_timestep = t_src

                with self.transformer.cache_context("cond"):
                    denoising_output = self.transformer(
                        hidden_states=latents.to(dtype=self.transformer.dtype),
                        timestep=t_src.expand(latents.shape[0]) / 1000,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                    )

                # unpack and create policy
                latents = self._unpack_latents(
                    latents, height, width, self.vae_scale_factor, target_patch_size=1)
                if self.policy_type == 'GMFlow':
                    denoising_output = self._unpack_gm(
                        denoising_output, height, width, num_channels_latents, gm_patch_size=1)
                    denoising_output = {k: v.to(torch.float32) for k, v in denoising_output.items()}
                    policy = self.policy_class(
                        denoising_output, latents, sigma_t_src)
                    if not is_final_step:
                        if temperature == 'auto':
                            temperature = min(max(0.1 * (num_inference_steps - 1), 0), 1)
                        else:
                            assert isinstance(temperature, (float, int))
                        policy.temperature_(temperature)
                elif self.policy_type == 'DX':
                    denoising_output = denoising_output[0]
                    denoising_output = self._unpack_latents(
                        denoising_output, height, width, self.vae_scale_factor, target_patch_size=1)
                    denoising_output = denoising_output.reshape(latents.size(0), -1, *latents.shape[1:])
                    denoising_output = denoising_output.to(torch.float32)
                    policy = self.policy_class(
                        denoising_output, latents, sigma_t_src)
                else:
                    raise ValueError(f'Unknown policy type: {self.policy_type}.')

                # compute the previous noisy sample x_t -> x_t-1
                for _ in range(num_inference_substeps[i]):
                    t = timesteps[timestep_id]
                    sigma_t = t / self.scheduler.config.num_train_timesteps
                    u = policy.u(latents, sigma_t)
                    latents = self.scheduler.step(u, t, latents, return_dict=False)[0]
                    timestep_id += 1

                # repack
                latents = self._pack_latents(
                    latents, latents.size(0), num_channels_latents,
                    2 * (int(height) // (self.vae_scale_factor * 2)),
                    2 * (int(width) // (self.vae_scale_factor * 2)),
                    patch_size=1)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t_src, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)[:, :, None]
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_std + latents_mean
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
