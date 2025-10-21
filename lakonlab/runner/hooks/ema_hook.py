# Copyright (c) 2025 Hansheng Chen

import mmcv
import torch

from copy import deepcopy
try:
    from torch.distributed.fsdp import FSDPModule
except:
    FSDPModule = None
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS
from mmgen.core import ExponentialMovingAverageHook
from lakonlab.utils import rgetattr, rhasattr


def get_ori_key(key):
    ori_key = key.split('.')
    if ori_key[0].endswith('_ema'):
        ori_key[0] = ori_key[0][:-4]
    elif ori_key[0].endswith('_ema2'):
        ori_key[0] = ori_key[0][:-5]
    else:
        raise ValueError(
            f'Invalid module key {key}, it should be in the format of '
            '<module_name>_ema or <module_name>_ema2, but got {ori_key[0]}')
    ori_key = '.'.join(ori_key)
    return ori_key


@HOOKS.register_module()
class ExponentialMovingAverageHookMod(ExponentialMovingAverageHook):

    _registered_momentum_updaters = ['rampup', 'fixed', 'karras']

    def __init__(self,
                 module_keys,
                 trainable_only=True,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0,
                 momentum_policy='fixed',
                 momentum_cfg=None):
        super(ExponentialMovingAverageHook, self).__init__()
        self.trainable_only = trainable_only
        # check args
        assert interp_mode in self._registered_interp_funcs, (
            'Supported '
            f'interpolation functions are {self._registered_interp_funcs}, '
            f'but got {interp_mode}')

        assert momentum_policy in self._registered_momentum_updaters, (
            'Supported momentum policy are'
            f'{self._registered_momentum_updaters},'
            f' but got {momentum_policy}')

        assert isinstance(module_keys, str) or mmcv.is_tuple_of(
            module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            module_name = k.split('.')[0]
            assert module_name.endswith('_ema') or module_name.endswith('_ema2')
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = getattr(self, interp_mode)

        self.momentum_cfg = dict() if momentum_cfg is None else deepcopy(
            momentum_cfg)
        self.momentum_policy = momentum_policy
        if momentum_policy != 'fixed':
            assert hasattr(
                self, momentum_policy
            ), f'Currently, we do not support {self.momentum_policy} for EMA.'
            self.momentum_updater = getattr(self, momentum_policy)

    def karras(self, runner, gamma=7.0, max_momentum=1.0):
        t = max(runner.iter + 1 - self.start_iter, 1)
        ema_beta = min((1 - 1 / t) ** (gamma + 1), max_momentum)
        return dict(momentum=ema_beta)

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        with torch.no_grad():
            model = runner.model.module if is_module_wrapper(
                runner.model) else runner.model

            # update momentum
            _interp_cfg = deepcopy(self.interp_cfg)
            if self.momentum_policy != 'fixed':
                _updated_args = self.momentum_updater(runner, **self.momentum_cfg)
                _interp_cfg.update(_updated_args)

            for key in self.module_keys:
                net = rgetattr(model, get_ori_key(key))
                ema = rgetattr(model, key)
                if FSDPModule is not None and isinstance(net, FSDPModule):  # Root parameters in EMA are unsharded after inference
                    net_is_sharded = net._get_fsdp_state()._fsdp_param_group.is_sharded
                    ema_is_sharded = ema._get_fsdp_state()._fsdp_param_group.is_sharded
                    if net_is_sharded and not ema_is_sharded:
                        ema.reshard()

                for p_net, p_ema in zip(net.parameters(), ema.parameters()):
                    if self.trainable_only and not p_net.requires_grad:
                        continue
                    if runner.iter < self.start_iter:
                        p_ema.data.copy_(p_net.data)
                    else:
                        p_ema.data.copy_(self.interp_func(
                            p_net, p_ema, trainable=p_net.requires_grad, **_interp_cfg))

                for b_net, b_ema in zip(net.buffers(), ema.buffers()):
                    b_ema.data.copy_(b_net.data)

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not rhasattr(model, k):
                raise RuntimeError(
                    f'Cannot find {k} network for EMA hook.')
