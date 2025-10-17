import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

import os
import argparse
import multiprocessing as mp
import re
import warnings

import pickle
import zstandard as zstd
import gzip
import orjson
import torch
import torch.nn as nn
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from io import BytesIO
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmcv.fileio import FileClient

from mmgen.apis import set_random_seed
from mmgen.datasets import build_dataset
from mmgen.models import build_module
from mmgen.utils import get_root_logger
from lakonlab.datasets import build_dataloader
from lakonlab.parallel import apply_module_wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cache the image latents and text embeddings for the ImagePrompt dataset. Usage: torchrun '
                    '--nnodes=1 --nproc_per_node=<NUM_GPUS> tools/cache_image_prompt_data.py <PATH_TO_CONFIG> '
                    '--text-encoder <PATH_TO_TEXT_ENCODER_CONFIG> --launcher pytorch --diff_seed')
    parser.add_argument('config', help='file path to the config containing the dataset and vae')
    parser.add_argument('--text-encoder', type=str, help='file path to the config of text encoder')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed testing)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed testing)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--max-size',
        type=int,
        help='max dataset size')
    parser.add_argument(
        '--batch-size', type=int, default=4, help='batch size per GPU')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def to_scaled_fp8(t: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn, eps=1e-6):
    amax = t.abs().max()
    fp8_max = torch.finfo(dtype).max
    scale = (amax.float() / fp8_max).clamp(min=eps)
    t_scaled = (t.float() / scale).to(dtype)
    return t_scaled, scale


class Preprocessor(nn.Module):

    def __init__(self, text_encoder, vae=None):
        super().__init__()
        self.text_encoder = build_module(text_encoder)
        self.vae = build_module(vae) if vae is not None else None

    def val_step(self, data, **kwargs):
        prompt_embed_kwargs = self.text_encoder(**data['prompt_kwargs'])
        output = dict(
            data_ids=data['ids'],
            prompts=data['name'],
            prompt_embed_kwargs={k: v for k, v in prompt_embed_kwargs.items()})
        if 'images' in data and self.vae is not None:
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents = self.vae.encode((data['images'] * 2 - 1).to(vae_dtype)).float()
            output.update(latents=latents)
        return output


def save_cache(data, cache_path):
    file_client = FileClient.infer_client(uri=cache_path)
    bytesio = BytesIO()
    compressor = zstd.ZstdCompressor(level=3)
    with compressor.stream_writer(bytesio, closefd=False) as f:
        pickle.dump(data, f)
    file_client.put(bytesio.getvalue(), cache_path)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.text_encoder is not None:
        cfg.model.text_encoder = Config.fromfile(args.text_encoder).model.text_encoder
    assert hasattr(cfg.model, 'text_encoder')

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
        world_size = 1
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE', 1)
    num_cpus = mp.cpu_count() // int(local_world_size)
    max_save_cache_workers = max(num_cpus - 2, 1)

    logger = get_root_logger(log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    model = Preprocessor(
        cfg.model.text_encoder,
        getattr(cfg.model, 'vae', None)
    ).requires_grad_(False).eval()

    if distributed:
        module_wrapper = cfg.get('module_wrapper', 'ddp')
        model = apply_module_wrapper(model, module_wrapper, cfg)
    else:
        model = MMDataParallel(model, device_ids=[0])

    if args.data is None:
        data_dict = {k: v for k, v in cfg.data.items() if k == 'train' or re.fullmatch(r'(val|test)\d*', k)}
    else:
        data_dict = {k: cfg.data[k] for k in args.data}

    for dataset_name, dataset in data_dict.items():
        data_root = dataset['data_root']
        cache_dir_path = os.path.join(data_root, dataset['cache_dir'])
        cache_datalist_path = dataset['cache_datalist_path']
        if FileClient.infer_client(uri=cache_datalist_path).isfile(cache_datalist_path):
            continue  # already cached

        if rank == 0:
            os.makedirs(cache_dir_path, exist_ok=True)
            os.makedirs(os.path.dirname(cache_datalist_path), exist_ok=True)
        if world_size > 1:
            dist.barrier()

        # disable slicing and repeat
        for key in ['start_ind', 'end_ind', 'repeat']:
            dataset.pop(key, None)
        if args.max_size is not None:
            dataset.start_ind = -args.max_size

        # build the dataloader
        dataset = build_dataset(dataset)
        bucket_ids = getattr(dataset, 'bucket_ids', None)

        # The default loader config
        loader_cfg = dict(
            num_gpus=len(cfg.gpu_ids),
            shuffle=False)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items()
            if k not in [
                'train', 'train_dataloader', 'val_dataloader', 'test_dataloader'
            ] and not re.fullmatch(r'(val|test)\d*', k)
        })

        # specific config for test loader
        batch_size = args.batch_size
        test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}
        test_loader_cfg.update(
            samples_per_gpu=batch_size,
            workers_per_gpu=1,
            prefetch_factor=batch_size * 2)

        dataloader = build_dataloader(dataset, **test_loader_cfg)

        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}, '
                        f'use_rank_shift: {args.diff_seed}')
            set_random_seed(
                args.seed,
                deterministic=args.deterministic,
                use_rank_shift=args.diff_seed)

        batch_size = dataloader.batch_size
        total_batch_size = batch_size * world_size

        max_num = len(dataloader.dataset)

        mp_ctx = mp.get_context("spawn")
        proc_pool = ProcessPoolExecutor(
            max_workers=min(max_save_cache_workers, batch_size * 2),
            mp_context=mp_ctx)
        cap = max_save_cache_workers * 2
        pending = set()

        if rank == 0:
            pbar = tqdm(total=max_num, desc=f'Caching {dataset_name} data')

        for _, data in enumerate(dataloader):
            outputs_dict = model.val_step(data)
            data_ids = outputs_dict['data_ids']
            prompts = outputs_dict['prompts']

            for batch_id, data_id in enumerate(data_ids):
                cached_data = dict(
                    prompt=prompts[batch_id],
                    prompt_embed_kwargs=dict())
                for k, v in outputs_dict['prompt_embed_kwargs'].items():
                    if k == 'encoder_hidden_states':
                        encoder_hidden_states_fp8, encoder_hidden_states_scale = to_scaled_fp8(v[batch_id])
                        cached_data['prompt_embed_kwargs']['encoder_hidden_states'] = encoder_hidden_states_fp8.cpu()
                        cached_data['prompt_embed_kwargs']['encoder_hidden_states_scale'] = encoder_hidden_states_scale.cpu()
                    else:
                        cached_data['prompt_embed_kwargs'][k] = v[batch_id].cpu()
                if 'latents' in outputs_dict:
                    latents_fp8, latents_scale = to_scaled_fp8(outputs_dict['latents'][batch_id])
                    cached_data['latents'] = latents_fp8.cpu()
                    cached_data['latents_scale'] = latents_scale.cpu()
                cached_data_name = f'{data_id:012d}'

                if len(pending) >= cap:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()

                pending.add(proc_pool.submit(
                    save_cache,
                    cached_data,
                    os.path.join(cache_dir_path, cached_data_name + '.zst')))

            if rank == 0:
                pbar.update(total_batch_size)

        proc_pool.shutdown(wait=True)

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            bytesio = BytesIO()
            if cache_datalist_path.endswith('.jsonl.gz') or cache_datalist_path.endswith('.jsonl'):
                datalist = []
                if bucket_ids is not None:
                    for data_id, bucket_idx in enumerate(bucket_ids):
                        datalist.append(orjson.dumps(
                            dict(
                                filename=f'{data_id:012d}',
                                bucket_id=int(bucket_idx))
                        ).decode('utf-8') + '\n')
                else:
                    for data_id in range(len(dataset)):
                        datalist.append(orjson.dumps(
                            dict(
                                filename=f'{data_id:012d}')
                        ).decode('utf-8') + '\n')
                datalist = ''.join(datalist).encode('utf-8')
                if cache_datalist_path.endswith('.jsonl.gz'):
                    with gzip.GzipFile(fileobj=bytesio, mode='wb') as f:
                        f.write(datalist)
                else:
                    bytesio.write(datalist)
            elif cache_datalist_path.endswith('.json'):
                datalist = [f'{data_id:012d}' for data_id in range(len(dataset))]
                bytesio.write(orjson.dumps(datalist))
            else:
                raise ValueError('Datalist file must be .jsonl, .jsonl.gz or .json')

            FileClient.infer_client(uri=cache_datalist_path).put(
                bytesio.getvalue(), cache_datalist_path)
            logger.info(f'Wrote datalist to {cache_datalist_path}')

    return


if __name__ == '__main__':
    main()
