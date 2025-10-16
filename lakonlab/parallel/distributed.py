from typing import Any

from mmcv.parallel.distributed import MMDistributedDataParallel as _MMDistributedDataParallel


class MMDistributedDataParallel(_MMDistributedDataParallel):

    def _run_ddp_forward(self, *inputs, **kwargs) -> Any:
        if hasattr(self, '_use_replicated_tensor_module') and self._use_replicated_tensor_module:
            module_to_run = self._replicated_tensor_module
        else:
            module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)
