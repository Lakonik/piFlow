import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import itertools
import shutil
import argparse
import torch
import torch.distributed as dist
import mmcv

from mmcv.runner import get_dist_info, init_dist
from mmgen.apis import set_random_seed

from lakonlab.datasets import ImageNet, build_dataloader
from lakonlab.models import PretrainedVAEEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cache the image latents for the ImageNet dataset using SD-VAE.')
    parser.add_argument('--data-root', type=str, default='data/imagenet/train')
    parser.add_argument('--datalist-path', type=str, default='data/imagenet/train.txt')
    parser.add_argument('--out-data-root', type=str, default='data/imagenet/train_cache')
    parser.add_argument('--out-datalist-path', type=str, default='data/imagenet/train_cache.txt')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per GPU')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()

    init_dist('pytorch')
    rank, ws = get_dist_info()

    if args.seed is not None:
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=True)

    dataset = ImageNet(
        data_root=args.data_root,
        datalist_path=args.datalist_path,
        image_size=args.image_size)

    dataloader = build_dataloader(
        dataset, args.batch_size, 8,
        persistent_workers=True, prefetch_factor=args.batch_size / 4, dist=True, shuffle=False)

    encoder = PretrainedVAEEncoder(
        from_pretrained='stabilityai/sd-vae-ft-ema', torch_dtype=args.dtype).eval().cuda()

    if rank == 0:
        if os.path.exists(args.out_data_root):
            for file in os.listdir(args.out_data_root):
                shutil.rmtree(os.path.join(args.out_data_root, file))
        if os.path.exists(args.out_datalist_path):
            os.remove(args.out_datalist_path)
        os.makedirs(args.out_data_root, exist_ok=True)
        os.makedirs(os.path.dirname(args.out_datalist_path), exist_ok=True)
    dist.barrier()

    torch.set_grad_enabled(False)

    if rank == 0:
        pbar = mmcv.ProgressBar(len(dataset))

    for data in dataloader:
        images = data['images'].to(dtype=getattr(torch, args.dtype)).cuda()
        labels = data['labels']
        paths = data['paths']

        latents = encoder(images * 2 - 1)

        for latent, label, path in zip(latents, labels, list(itertools.chain.from_iterable(paths.data))):
            out_path = os.path.join(
                os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.pth')
            os.makedirs(os.path.join(args.out_data_root, os.path.dirname(out_path)), exist_ok=True)
            torch_data = dict(x=latent.cpu(), y=label.cpu())
            torch.save(torch_data, os.path.join(args.out_data_root, out_path))

        if rank == 0:
            pbar.update(args.batch_size * ws)

    dist.barrier()

    if rank == 0:
        with open(args.out_datalist_path, 'w') as f:
            for label, path in zip(dataset.all_labels, dataset.all_paths):
                out_path = os.path.join(
                    os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.pth')
                assert os.path.exists(os.path.join(args.out_data_root, out_path))
                f.write(f'{out_path} {label:d}\n')
        print('Done!')
