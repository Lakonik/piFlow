from mmcv.parallel import MODULE_WRAPPERS
from mmgen.core.ddp_wrapper import DistributedDataParallelWrapper as _DistributedDataParallelWrapper
from lakonlab.parallel import MMDistributedDataParallel


@MODULE_WRAPPERS.register_module('mmgen.DDPWrapper', force=True)
class DistributedDataParallelWrapper(_DistributedDataParallelWrapper):

    def to_ddp(self, device_ids, dim, broadcast_buffers,
               find_unused_parameters, **kwargs):
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                module = module.cuda()
            elif name.endswith('_ema') or name.endswith('_ema2'):
                module = module.cuda()
            else:
                module = MMDistributedDataParallel(
                    module.cuda(),
                    device_ids=device_ids,
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)
            self.module._modules[name] = module
