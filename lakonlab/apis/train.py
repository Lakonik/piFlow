# Modified from https://github.com/open-mmlab/mmgeneration

import warnings
import re
from copy import deepcopy

from mmcv.parallel import MMDataParallel
from mmcv.runner import HOOKS, IterBasedRunner, OptimizerHook, build_runner
from mmcv.utils import build_from_cfg

from mmgen.datasets import build_dataset
from mmgen.utils import get_root_logger

from lakonlab.parallel import apply_module_wrapper
from lakonlab.runner.optimizer import build_optimizers
from lakonlab.runner.checkpoint import exists_ckpt
from lakonlab.datasets import build_dataloader


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        seed=cfg.seed)

    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items()
        if k not in [
            'train', 'train_dataloader', 'val_dataloader', 'test_dataloader'
        ] and not re.fullmatch(r'(val|test)\d*', k)
    })

    # The specific datalaoder settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    if cfg.get('apex_amp', None):
        raise NotImplementedError('Apex AMP is no longer supported.')

    # put model on gpus
    if distributed:
        module_wrapper = cfg.get('module_wrapper', None)
        model = apply_module_wrapper(model, module_wrapper, cfg)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build optimizer
    if cfg.optimizer:
        optimizer = build_optimizers(model, cfg.optimizer)
    # In GANs, we allow building optimizer in GAN model.
    else:
        optimizer = None

    # allow users to define the runner
    if cfg.get('runner', None):
        runner = build_runner(
            cfg.runner,
            dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                use_apex_amp=False,
                meta=meta))
    else:
        runner = IterBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
        # set if use dynamic ddp in training
        # is_dynamic_ddp=cfg.get('is_dynamic_ddp', False))
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)

    # In GANs, we can directly optimize parameter in `train_step` function.
    if cfg.get('optimizer_cfg', None) is None:
        optimizer_config = None
    elif fp16_cfg is not None:
        raise NotImplementedError('Fp16 has not been supported.')
        # optimizer_config = Fp16OptimizerHook(
        #     **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # default to use OptimizerHook
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # # update `out_dir` in  ckpt hook
    # if cfg.checkpoint_config is not None:
    #     cfg.checkpoint_config['out_dir'] = os.path.join(
    #         cfg.work_dir, cfg.checkpoint_config.get('out_dir', 'ckpt'))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # # DistSamplerSeedHook should be used with EpochBasedRunner
    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())

    # In general, we do NOT adopt standard evaluation hook in GAN training.
    # Thus, if you want a eval hook, you need further define the key of
    # 'evaluation' in the config.
    # register eval hooks
    if validate and cfg.get('evaluation', None) is not None:
        assert isinstance(cfg.evaluation, list)
        for eval_cfg_ in cfg.evaluation:
            val_dataset = build_dataset(cfg.data[eval_cfg_.data])
            val_loader_cfg = {
                **loader_cfg, 'shuffle': False,
                **cfg.data.get('val_dataloader', {})
            }
            val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
            eval_cfg = deepcopy(eval_cfg_)
            priority = eval_cfg.pop('priority', 'LOW')
            eval_cfg.update(dict(dist=distributed, dataloader=val_dataloader))
            eval_hook = build_from_cfg(eval_cfg, HOOKS)
            runner.register_hook(eval_hook, priority=priority)

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    ckpt_kwargs = dict()
    if distributed and module_wrapper.lower() in ['fsdp', 'fsdp2']:
        ckpt_kwargs.update(map_location='cpu')
    if exists_ckpt(cfg.resume_from):
        runner.resume(cfg.resume_from, **ckpt_kwargs)
        for data_loader in data_loaders:
            data_loader.sampler.set_epoch(runner.epoch)
            data_loader.sampler.set_iter(runner.iter)
    elif exists_ckpt(cfg.load_from):
        runner.load_checkpoint(cfg.load_from, **ckpt_kwargs)

    runner.run(data_loaders, cfg.workflow, cfg.total_iters)
