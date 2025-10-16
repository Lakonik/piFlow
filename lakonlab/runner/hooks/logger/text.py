import torch
import torch.distributed as dist

from mmcv.runner import HOOKS
from mmcv.runner import TextLoggerHook as _TextLoggerHook

from lakonlab.parallel import FSDPWrapper


@HOOKS.register_module(force=True)
class TextLoggerHook(_TextLoggerHook):

    def _get_max_memory(self, runner) -> int:
        if isinstance(runner.model, FSDPWrapper):
            return 0
        else:
            device = getattr(runner.model, 'output_device', None)
            mem = torch.cuda.max_memory_allocated(device=device)
            mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                                  dtype=torch.int,
                                  device=device)
            if runner.world_size > 1:
                dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
            return mem_mb.item()
