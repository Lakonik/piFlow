import torch
import mmcv

from . import MMDistributedDataParallel, DistributedDataParallelWrapper, FSDPWrapper, FSDP2Wrapper


def apply_module_wrapper(model, module_wrapper, cfg):
    if module_wrapper is None:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    elif module_wrapper.lower() == 'ddp':
        mmcv.print_log('Use DDP Wrapper.', 'mmgen')
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    elif module_wrapper.lower() == 'fsdp':
        mmcv.print_log('Use FSDP Wrapper.', 'mmgen')
        fsdp_kwargs = cfg.get('fsdp_kwargs', {})
        model = FSDPWrapper(
            model,
            device_id=torch.cuda.current_device(),
            **fsdp_kwargs)
    elif module_wrapper.lower() == 'fsdp2':
        mmcv.print_log('Use FSDP2 Wrapper.', 'mmgen')
        fsdp_kwargs = cfg.get('fsdp_kwargs', {})
        model = FSDP2Wrapper(
            model,
            **fsdp_kwargs)
    else:
        raise ValueError(f'Unsupported module wrapper: {module_wrapper}.')
    return model
