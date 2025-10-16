import copy
from mmcv.runner.hooks import CheckpointHook as _CheckpointHook
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import allreduce_params, get_dist_info
from lakonlab.parallel import FSDPWrapper


@HOOKS.register_module(force=True)
class CheckpointHook(_CheckpointHook):

    def __init__(self,
                 interval: int = -1,
                 must_save_interval: int = 1000000000,
                 **kwargs):
        super().__init__(interval=interval, **kwargs)
        self.must_save_interval = must_save_interval

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.every_n_epochs(runner, self.interval) \
                or self.every_n_epochs(runner, self.must_save_interval) \
                or (self.save_last and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner, asynchronous=not self.is_last_epoch(runner))

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.every_n_iters(runner, self.interval) \
                or self.every_n_iters(runner, self.must_save_interval) \
                or (self.save_last and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner, asynchronous=not self.is_last_iter(runner))

    def _save_checkpoint(self, runner, asynchronous=False):
        rank, _ = get_dist_info()
        # in FSDP we need to call save_checkpoint on all ranks
        if rank == 0 or isinstance(runner.model, FSDPWrapper):
            if self.max_keep_ckpts > 0:
                # remove other checkpoints
                cur_progress = copy.deepcopy(runner.epoch) if self.by_epoch else copy.deepcopy(runner.iter)

                def after_save_hook():
                    if self.by_epoch:
                        name = 'epoch_{}.pth'
                        current_ckpt = cur_progress + 1
                    else:
                        name = 'iter_{}.pth'
                        current_ckpt = cur_progress + 1
                    redundant_ckpts = [i for i in range(
                        current_ckpt - self.max_keep_ckpts * self.interval, 0,
                        -self.interval) if i % self.must_save_interval != 0]
                    filename_tmpl = self.args.get('filename_tmpl', name)
                    for _step in redundant_ckpts:
                        ckpt_path = self.file_client.join_path(
                            self.out_dir, filename_tmpl.format(_step))
                        if self.file_client.isfile(ckpt_path):
                            self.file_client.remove(ckpt_path)
                        else:
                            break

            else:
                after_save_hook = None

            runner.save_checkpoint(
                self.out_dir,
                save_optimizer=self.save_optimizer,
                after_save_hook=after_save_hook,
                asynchronous=asynchronous,
                **self.args)

        if rank == 0:
            if runner.meta is not None:
                if self.by_epoch:
                    cur_ckpt_filename = self.args.get(
                        'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
                else:
                    cur_ckpt_filename = self.args.get(
                        'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
                runner.meta.setdefault('hook_msgs', dict())
                runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                    self.out_dir, cur_ckpt_filename)
