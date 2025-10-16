# Copyright (c) 2025 Hansheng Chen

import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowSDESchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowSDEScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            h: Union[float, str] = 1.0,
            shift: float = 1.0,
            use_dynamic_shifting=False,
            base_seq_len=256,
            max_seq_len=4096,
            base_logshift=0.5,
            max_logshift=1.15,
            terminal_sigma=None):
        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_train_timesteps, dtype=np.float32, endpoint=False))
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def get_shift(self, seq_len=None):
        if self.config.use_dynamic_shifting and seq_len is not None:
            m = (self.config.max_logshift - self.config.base_logshift
                 ) / (self.config.max_seq_len - self.config.base_seq_len)
            logshift = (seq_len - self.config.base_seq_len) * m + self.config.base_logshift
            if isinstance(logshift, torch.Tensor):
                shift = torch.exp(logshift)
            else:
                shift = np.exp(logshift)
        else:
            shift = self.config.shift
        return shift

    def stretch_to_terminal(self, sigma):
        one_minus_sigma = 1 - sigma
        stretched_sigma = 1 - (one_minus_sigma * (1 - self.config.terminal_sigma) / one_minus_sigma[-1])
        return stretched_sigma

    def set_timesteps(self, num_inference_steps: int, seq_len=None, device=None):
        self.num_inference_steps = num_inference_steps

        sigmas = torch.from_numpy(np.linspace(
            1, 0, num_inference_steps, dtype=np.float32, endpoint=False))
        shift = self.get_shift(seq_len=seq_len)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.terminal_sigma is not None:
            sigmas = self.stretch_to_terminal(sigmas)

        self.timesteps = (sigmas * self.config.num_train_timesteps).to(device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            prediction_type='u',
            eps=1e-6) -> Union[FlowSDESchedulerOutput, Tuple]:
        assert prediction_type in ['u', 'x0']

        if isinstance(timestep, int) \
                or isinstance(timestep, torch.IntTensor) \
                or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    'Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to'
                    ' `EulerDiscreteScheduler.step()` is not supported. Make sure to pass'
                    ' one of the `scheduler.timesteps` as a timestep.'
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        ori_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        alpha = 1 - sigma
        alpha_to = 1 - sigma_to

        if prediction_type == 'u':
            x0 = sample - sigma * model_output
            epsilon = sample + alpha * model_output
        else:
            x0 = model_output
            epsilon = (sample - alpha * x0) / sigma.clamp(min=eps)
        noise = randn_tensor(
            model_output.shape, dtype=torch.float32, device=model_output.device, generator=generator)

        if self.config.h == 'inf':
            m = torch.zeros_like(sigma)
        elif self.config.h == 0.0:
            m = torch.ones_like(sigma)
        else:
            assert self.config.h > 0.0
            h2 = self.config.h * self.config.h
            m = (sigma_to * alpha / (sigma * alpha_to).clamp(min=eps)) ** h2

        prev_sample = alpha_to * x0 + sigma_to * (m * epsilon + (1 - m.square()).clamp(min=0).sqrt() * noise)

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowSDESchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
