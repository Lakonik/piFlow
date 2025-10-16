# Copyright (c) 2025 Hansheng Chen

import torch
import torch.nn as nn
from lakonlab.utils import rgetattr


def autocast_patch(module, dtype=None, enabled=True):

    def make_new_forward(old_forward, dtype, enabled):
        def new_forward(*args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled):
                result = old_forward(*args, **kwargs)
            return result

        return new_forward

    module.forward = make_new_forward(module.forward, dtype, enabled)


def flex_freeze(module, exclude_keys=None, exclude_fp32=True, exclude_autocast_dtype='float32'):
    module.requires_grad_(False)

    if exclude_keys is not None and len(exclude_keys) > 0:
        exclude_modules = []
        for name, _ in module.named_modules():
            for exclude_key in exclude_keys:
                if exclude_key.startswith('self.'):  # use full name matching
                    if exclude_key[5:] == name:
                        exclude_modules.append(name)
                        break
                elif exclude_key in name:  # use partial name matching
                    exclude_modules.append(name)
                    break

        for attr in exclude_modules:
            rgetattr(module, attr).requires_grad_(True)

        if exclude_fp32:
            for attr in exclude_modules:
                m = rgetattr(module, attr)
                assert isinstance(m, nn.Module)
                m.to(torch.float32)
                autocast_patch(m, dtype=getattr(torch, exclude_autocast_dtype))
