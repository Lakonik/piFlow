import gc
import copy
import warnings
import time
import os.path as osp
import threading
import torch
import mmcv
from typing import Any, Dict
from torch.optim import Optimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions
from mmcv.runner import RUNNERS, get_dist_info, get_host_info
from mmgen.core.runners.dynamic_iterbased_runner import DynamicIterBasedRunner, IterLoader

from .checkpoint import get_checkpoint, load_checkpoint, write_checkpoint_to_file
from lakonlab.utils import rgetattr, gc_context


_last_save_thread: threading.Thread | None = None
_last_save_lock = threading.Lock()        # protects _last_save_thread


@RUNNERS.register_module()
class DynamicIterBasedRunnerMod(DynamicIterBasedRunner):
    
    def __init__(self,
                 *args,
                 ckpt_trainable_only=False,
                 ckpt_fp16=False,
                 ckpt_fp16_ema=False,
                 ckpt_bf16_optim=False,
                 gc_interval=-1,
                 **kwargs):
        super(DynamicIterBasedRunnerMod, self).__init__(*args, **kwargs)
        self.ckpt_trainable_only = ckpt_trainable_only
        self.ckpt_fp16 = ckpt_fp16
        self.ckpt_fp16_ema = ckpt_fp16_ema
        self.ckpt_bf16_optim = ckpt_bf16_optim
        self.gc_interval = gc_interval
        self.manual_gc = isinstance(gc_interval, int) and gc_interval > 0

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x, self) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                with gc_context(enable=not self.manual_gc):
                    self._inner_iter = 0
                    mode, iters = flow
                    if not isinstance(mode, str) or not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run a workflow'.
                            format(mode))
                    iter_runner = getattr(self, mode)
                    for _ in range(iters):
                        if mode == 'train' and self.iter >= self._max_iters:
                            break
                        if self.manual_gc and self._inner_iter % self.gc_interval == 0:
                            gc.collect()
                        iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True,
                        after_save_hook=None,
                        asynchronous=False):
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        _loss_scaler = self.loss_scaler if self.with_fp16_grad_scaler else None
        checkpoint = get_checkpoint(
            self.model,
            optimizer=optimizer,
            loss_scaler=_loss_scaler,
            meta=meta,
            trainable_only=self.ckpt_trainable_only,
            fp16=self.ckpt_fp16,
            fp16_ema=self.ckpt_fp16_ema,
            bf16_optim=self.ckpt_bf16_optim)

        rank, _ = get_dist_info()
        if rank == 0:
            global _last_save_thread
            with _last_save_lock:
                if _last_save_thread is not None and _last_save_thread.is_alive():
                    print('Waiting for the previous write to finish...')
                    _last_save_thread.join()  # wait for the previous write

                if asynchronous:
                    _last_save_thread = threading.Thread(
                        target=write_checkpoint_to_file,
                        args=(copy.deepcopy(checkpoint), filepath, create_symlink, after_save_hook),
                        daemon=True)
                    _last_save_thread.start()

                else:
                    write_checkpoint_to_file(
                        checkpoint,
                        filepath,
                        create_symlink=create_symlink,
                        after_save_hook=after_save_hook)

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,
            revise_keys=revise_keys)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               resume_loss_scaler=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']

        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                set_optimizer_state_dict(
                    model=self.model,
                    optimizers=self.optimizer,
                    optim_state_dict=checkpoint['optimizer'],
                    options=StateDictOptions(
                        full_state_dict=isinstance(self.model, FSDP),
                        broadcast_from_rank0=False))
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    m = rgetattr(self.model, k)
                    set_optimizer_state_dict(
                        model=m,
                        optimizers=self.optimizer[k],
                        optim_state_dict=checkpoint['optimizer'][k],
                        options=StateDictOptions(
                            full_state_dict=isinstance(m, FSDP),
                            broadcast_from_rank0=False))
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'loss_scaler' in checkpoint and resume_loss_scaler and hasattr(self, 'loss_scaler'):
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')
