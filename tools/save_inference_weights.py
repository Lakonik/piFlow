import os
import argparse
import torch
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Save the inference weights of a checkpoint')
    parser.add_argument('path', help='path to the checkpoint')
    parser.add_argument('--out-path', help='path to save the inference weights')
    parser.add_argument('--dtype', default='float16', help='dtype of inference weights')
    parser.add_argument('--ema-only', action='store_true', help='save only the EMA weights')
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    if args.out_path is not None:
        out_path = args.out_path
    else:
        root, ext = os.path.splitext(path)
        out_path = root + '_inference' + ext
    dtype = args.dtype
    a = torch.load(path, map_location='cpu')
    if 'optimizer' in a:
        del a['optimizer']
    if args.ema_only:
        ema_keys = [key for key in a['state_dict'].keys() if '_ema' in key]
        exclude_keys = [key.replace('_ema', '') for key in ema_keys]
    else:
        exclude_keys = []
    out_dict = OrderedDict()
    for key, value in a['state_dict'].items():
        if key not in exclude_keys:
            out_dict[key] = value.to(getattr(torch, dtype))
    a['state_dict'] = out_dict
    torch.save(a, out_path)


if __name__ == '__main__':
    main()
