import os
import sys
import argparse
import socket
from contextlib import closing


def parse_args():
    parser = argparse.ArgumentParser(description='Test and eval a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--ckpt', help='checkpoint file')
    parser.add_argument(
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--timer',
        action='store_true',
        help='whether to enable timers')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='whether to skip evaluation when visualization HTML already exists')
    parser.add_argument(
        '--reuse-viz',
        action='store_true',
        help='whether to bypass generation and reuse existing visualization images for evaluation')
    args = parser.parse_args()
    return args


def args_to_str(args):
    argv = [args.config]
    if args.ckpt is not None:
        argv += ['--ckpt', args.ckpt]
    if args.seed is not None:
        argv += ['--seed', str(args.seed)]
    if args.deterministic:
        argv.append('--deterministic')
    if args.data is not None:
        argv += ['--data'] + args.data
    if args.timer:
        argv.append('--timer')
    if args.skip_existing:
        argv.append('--skip-existing')
    if args.reuse_viz:
        argv.append('--reuse-viz')
    return argv


def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) == 1:
        import tools.test
        sys.argv = [''] + args_to_str(args)
        tools.test.main()
    else:
        from torch.distributed import launch
        for port in range(29500, 65536):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                res = sock.connect_ex(('localhost', port))
                if res != 0:
                    break
        sys.argv = ['',
                    '--nproc_per_node={}'.format(len(gpu_ids)),
                    '--master_port={}'.format(port),
                    './tools/test.py'
                    ] + args_to_str(args) + ['--launcher', 'pytorch', '--diff_seed']
        launch.main()


if __name__ == '__main__':
    main()
