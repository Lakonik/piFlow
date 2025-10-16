import warnings
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, digit_version
from torch.utils.data import DataLoader

from mmgen.datasets.builder import worker_init_fn
from .samplers import DistributedSampler


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     persistent_workers=False,
                     sampler=None,
                     **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        assert sampler is None, 'sampler is not supported in distributed mode'
        sampler = DistributedSampler(
            dataset,
            world_size,
            rank,
            shuffle=shuffle,
            samples_per_gpu=samples_per_gpu,
            seed=seed)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (digit_version(TORCH_VERSION) >= digit_version('1.7.0')
            and TORCH_VERSION != 'parrots'):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader
