# Copyright (c) 2025 Hansheng Chen

import os
import sys
import logging
import pickle
import warnings
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import mmcv
import hashlib

from copy import deepcopy
from contextlib import contextmanager, redirect_stdout, nullcontext
from scipy import linalg
from scipy.stats import entropy
from torchvision import models
from mmcv.runner import get_dist_info, load_checkpoint
from mmgen.utils import get_root_logger
from mmgen.core.registry import METRICS
from mmgen.core.evaluation.metrics import (
    Metric, TERO_INCEPTION_URL, _load_inception_torch, MMGEN_CACHE_DIR)
from mmgen.core.evaluation.metrics import FID as _FID
from mmgen.core.evaluation.metrics import PR as _PR
from open_clip import get_tokenizer, create_model
from lakonlab.utils.io_utils import download_from_huggingface, download_from_url


# Global caches for model loading
_inception_cache = {}
_hpsv2_cache = {}
_clip_cache = {}


def _argv_ctx(argv):
    class _Argv:

        def __enter__(self):
            self._old = sys.argv
            sys.argv = argv

        def __exit__(self, exc_type, exc, tb):
            sys.argv = self._old

    return _Argv()


def _redirect_stdout(to_buf):
    return redirect_stdout(to_buf) if to_buf is not None else nullcontext()


@contextmanager
def _quarantine_openclip_logging():
    """
    Guard against open_clip (and friends) mutating global logging.
    Snapshots root handlers/level, runs the block, then removes any
    NEW handlers and restores the level. Also disables propagation
    for the open_clip logger so logs don’t bubble to root.
    """
    root = logging.getLogger()
    before_handlers = tuple(root.handlers)   # snapshot by identity
    before_ids = {id(h) for h in before_handlers}
    before_level = root.level

    try:
        yield
    finally:
        # Remove only handlers that were added during the block
        for h in list(root.handlers):
            if id(h) not in before_ids:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        root.setLevel(before_level)

        # Clamp open_clip logger so it won’t re-emit to root
        oc = logging.getLogger("open_clip")
        oc.propagate = False
        oc.handlers.clear()


def _load_inception_from_path(inception_path, map_location=None):
    mmcv.print_log(
        'Try to load Tero\'s Inception Model from '
        f'\'{inception_path}\'.', 'mmgen')
    try:
        model = torch.jit.load(inception_path, map_location=map_location)
        mmcv.print_log('Load Tero\'s Inception Model successfully.', 'mmgen')
    except Exception as e:
        model = None
        mmcv.print_log(
            'Load Tero\'s Inception Model failed. '
            f'\'{e}\' occurs.', 'mmgen')
    return model


def _load_inception_from_url(inception_url, map_location=None):
    """
    Fix multi-node downloading issue in MMGen.
    """
    inception_url = inception_url if inception_url else TERO_INCEPTION_URL
    mmcv.print_log(f'Try to download Inception Model from {inception_url}...',
                   'mmgen')
    try:
        path = download_from_url(inception_url, dest_dir=MMGEN_CACHE_DIR)
        mmcv.print_log('Download Finished.')
        return _load_inception_from_path(path, map_location=map_location)
    except Exception as e:
        mmcv.print_log(f'Download Failed. {e} occurs.')
        return None


def load_inception(inception_args, metric, map_location=None):
    """
    Fix multi-node downloading issue in MMGen.
    """
    if not isinstance(inception_args, dict):
        raise TypeError('Receive invalid \'inception_args\': '
                        f'\'{inception_args}\'')

    # Create cache key from arguments
    cache_key = hashlib.md5(str(sorted(inception_args.items())).encode()).hexdigest()
    cache_key += f"_{metric}"
    
    # Check if model is already cached
    if cache_key in _inception_cache:
        return _inception_cache[cache_key]

    _inception_args = deepcopy(inception_args)
    inceptoin_type = _inception_args.pop('type', None)

    if torch.__version__ < '1.6.0':
        mmcv.print_log(
            'Current Pytorch Version not support script module, load '
            'Inception Model from torch model zoo. If you want to use '
            'Tero\' script model, please update your Pytorch higher '
            f'than \'1.6\' (now is {torch.__version__})', 'mmgen')
        result = _load_inception_torch(_inception_args, metric), 'pytorch'
        _inception_cache[cache_key] = result
        return result

    # load pytorch version is specific
    if inceptoin_type != 'StyleGAN':
        result = _load_inception_torch(_inception_args, metric), 'pytorch'
        _inception_cache[cache_key] = result
        return result

    # try to load Tero's version
    path = _inception_args.get('inception_path', TERO_INCEPTION_URL)

    # try to parse `path` as web url and download
    if 'http' not in path:
        model = _load_inception_from_path(path, map_location=map_location)
        if isinstance(model, torch.nn.Module):
            result = model, 'StyleGAN'
            _inception_cache[cache_key] = result
            return result

    # try to parse `path` as path on disk
    model = _load_inception_from_url(path, map_location=map_location)
    if isinstance(model, torch.nn.Module):
        result = model, 'StyleGAN'
        _inception_cache[cache_key] = result
        return result

    raise RuntimeError('Cannot Load Inception Model, please check the input '
                       f'`inception_args`: {inception_args}')


def load_hpsv2(hps_version, device='cpu', precision='fp16'):
    assert hps_version in ['v2', 'v2.1']
    
    # Create cache key from arguments
    cache_key = f"{hps_version}_{device}_{precision}"
    
    # Check if model is already cached
    if cache_key in _hpsv2_cache:
        return _hpsv2_cache[cache_key]

    with _quarantine_openclip_logging():
        model = create_model(
            'ViT-H-14-quickgelu',
            precision=precision,
            device=device,
            output_dict=True)
        model.requires_grad_(False)
        tokenizer = get_tokenizer('ViT-H-14')
    load_checkpoint(
        model,
        f'huggingface://xswu/HPSv2/HPS_{hps_version}_compressed.pt',
        map_location='cpu', strict=True)

    result = model, tokenizer
    _hpsv2_cache[cache_key] = result
    return result


def load_openclip(
        model_name='ViT-L-14-336-quickgelu',
        pretrained='openai',
        device='cpu',
        precision='fp16'):
    cache_key = f'{model_name}_{pretrained}_{device}_{precision}'
    if cache_key in _clip_cache:
        return _clip_cache[cache_key]

    with _quarantine_openclip_logging():
        model = create_model(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            output_dict=True)
        model.requires_grad_(False)
        tokenizer = get_tokenizer(model_name)
    _clip_cache[cache_key] = (model, tokenizer)
    return _clip_cache[cache_key]


def compute_pr_distances(row_features,
                         col_features,
                         col_batch_size=10000):
    dist_batches = []
    for col_batch in col_features.split(col_batch_size):
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        dist_batches.append(dist_batch.cpu())
    return torch.cat(dist_batches, dim=1)


@METRICS.register_module(force=True)
class PR(_PR):

    def __init__(
            self,
            num_images=None,
            image_shape=None,
            feats_pkl=None,
            k=3,
            bgr2rgb=True,
            vgg16_script=None,
            inception_args=None,
            row_batch_size=10000,
            col_batch_size=10000):
        super(_PR, self).__init__(num_images, image_shape)

        self.feats_pkl = feats_pkl

        self.vgg16 = self.inception_net = None
        self.device = 'cpu'

        if vgg16_script is not None:
            mmcv.print_log('loading vgg16 for improved precision and recall...',
                           'mmgen')
            if os.path.isfile(vgg16_script):
                self.vgg16 = torch.jit.load('work_dirs/cache/vgg16.pt', map_location=self.device).eval()
                self.use_tero_scirpt = True
            else:
                mmcv.print_log(
                    'Cannot load Tero\'s script module. Use official '
                    'vgg16 instead', 'mmgen')
                self.vgg16 = models.vgg16(pretrained=True).eval()
                self.use_tero_scirpt = False
        elif inception_args is not None:
            self.inception_net, self.inception_style = load_inception(
                inception_args, 'FID')
        else:
            raise ValueError('Please provide either vgg16_script or inception_args')

        self.k = k
        self.bgr2rgb = bgr2rgb
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size

    def prepare(self):
        self.features_of_reals = []
        self.features_of_fakes = []
        if self.feats_pkl is not None:
            assert mmcv.is_filepath(self.feats_pkl)
            with open(self.feats_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.features_of_reals = [torch.from_numpy(feat) for feat in reference['features_of_reals']]
                self.num_real_feeded = reference['num_real_feeded']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.feats_pkl}',
                    'mmgen')

    def extract_features(self, batch):
        if self.vgg16 is not None:
            if self.use_tero_scirpt:
                batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                feat = self.vgg16(batch, return_features=True)
            else:
                batch = F.interpolate(batch, size=(224, 224))
                before_fc = self.vgg16.features(batch)
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feat = self.vgg16.classifier[:4](before_fc)
        else:
            if self.inception_style == 'StyleGAN':
                batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                feat = self.inception_net(batch, return_features=True)
            else:
                feat = self.inception_net(batch)[0].view(batch.shape[0], -1)
        return feat

    @torch.no_grad()
    def feed_op(self, batch, mode):
        batch = batch.to(self.device)
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]

        feat = self.extract_features(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(feat) for _ in range(ws)]
            dist.all_gather(placeholder, feat)
            feat = torch.stack(placeholder, dim=1).reshape(feat.size(0) * ws, *feat.shape[1:])

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            if mode == 'reals':
                self.features_of_reals.append(feat)
            elif mode == 'fakes':
                self.features_of_fakes.append(feat)
            else:
                raise ValueError(f'{mode} is not a implemented feed mode.')

    def feed(self, batch, mode):
        if self.num_images is not None:
            return super().feed(batch, mode)
        else:
            self.feed_op(batch, mode)

    @torch.no_grad()
    def summary(self):
        gen_features = torch.cat(self.features_of_fakes)
        real_features = torch.cat(self.features_of_reals).to(device=gen_features.device)
        if self.num_images is not None:
            assert gen_features.shape[0] >= self.num_images
            gen_features = gen_features[:self.num_images]
            if self.feats_pkl is None:  # real feats not pre-calculated
                assert real_features.shape[0] >= self.num_images
                real_features = real_features[:self.num_images]

        self._result_dict = {}

        for name, manifold, probes in [
            ('precision', real_features, gen_features),
            ('recall', gen_features, real_features)
        ]:
            kth = []
            for manifold_batch in manifold.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=manifold_batch,
                    col_features=manifold,
                    col_batch_size=self.col_batch_size)
                kth.append(
                    distance.to(torch.float32).kthvalue(self.k + 1).values.to(torch.float16))
            kth = torch.cat(kth)
            pred = []
            for probes_batch in probes.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=probes_batch,
                    col_features=manifold,
                    col_batch_size=self.col_batch_size)
                pred.append((distance <= kth).any(dim=1))
            self._result_dict[name] = float(torch.cat(pred).to(torch.float32).mean())

        precision = self._result_dict['precision']
        recall = self._result_dict['recall']
        self._result_str = f'precision: {precision}, recall:{recall}'
        return self._result_dict

    def clear_fake_data(self):
        self.features_of_fakes = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()
        if clear_reals:
            self.features_of_reals = []
            self.num_real_feeded = 0

    def load_to_gpu(self):
        """Move models to GPU."""
        if torch.cuda.is_available():
            if self.vgg16 is not None:
                self.vgg16 = self.vgg16.cuda()
            elif self.inception_net is not None:
                self.inception_net.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        """Move models to CPU."""
        if self.vgg16 is not None:
            self.vgg16 = self.vgg16.cpu()
        elif self.inception_net is not None:
            self.inception_net.cpu()
        self.device = 'cpu'


@METRICS.register_module(force=True)
class FID(_FID):

    def __init__(self,
                 num_images=None,
                 image_shape=None,
                 inception_pkl=None,
                 bgr2rgb=True,
                 inception_args=dict(normalize_input=False)):
        super().__init__(
            num_images,
            image_shape=image_shape,
            inception_pkl=inception_pkl,
            bgr2rgb=bgr2rgb,
            inception_args=inception_args)

    def prepare(self):
        if self.inception_pkl is not None:
            assert mmcv.is_filepath(self.inception_pkl)
            if self.inception_pkl.startswith('huggingface://'):
                self.inception_pkl = download_from_huggingface(self.inception_pkl)
            elif self.inception_pkl.startswith(('http://', 'https://')):
                self.inception_pkl = download_from_url(self.inception_pkl)
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.inception_pkl}',
                    'mmgen')
            self.num_real_feeded = self.num_images

    @torch.no_grad()
    def summary(self):
        # calculate reference inception stat
        if self.real_mean is None:
            feats = torch.cat(self.real_feats, dim=0)
            if self.num_images is not None:
                assert feats.shape[0] >= self.num_images
                feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        # calculate fake inception stat
        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert fake_feats.shape[0] >= self.num_images
            fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        # calculate distance between real and fake statistics
        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean, self.real_cov)

        # results for print/table
        self._result_str = (f'{fid:.4f} ({mean:.5f}/{cov:.5f})')
        # results for log_buffer
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov)

        return fid, mean, cov

    def feed(self, batch, mode):
        if self.num_images is not None:
            return super().feed(batch, mode)
        else:
            self.feed_op(batch, mode)


@METRICS.register_module()
class FIDKID(FID):
    name = 'FIDKID'

    def __init__(self,
                 num_images=None,
                 num_subsets=100,
                 max_subset_size=1000,
                 **kwargs):
        super().__init__(num_images=num_images, **kwargs)
        self.num_subsets = num_subsets
        self.max_subset_size = max_subset_size
        self.real_feats_np = None

    def prepare(self):
        if self.inception_pkl is not None:
            assert mmcv.is_filepath(self.inception_pkl)
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['feats_np']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.inception_pkl}',
                    'mmgen')
            self.num_real_feeded = self.num_images

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def summary(self):
        if self.real_feats_np is None:
            feats = torch.cat(self.real_feats, dim=0)
            if self.num_images is not None:
                assert feats.shape[0] >= self.num_images
                feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_feats_np = feats_np
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert fake_feats.shape[0] >= self.num_images
            fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.num_subsets,
                             self.max_subset_size) * 1000

        self._result_str = f'{fid:.4f} ({mean:.5f}/{cov:.5f}), {kid:.4f}'
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov, kid=kid)

        return fid, mean, cov, kid


@METRICS.register_module()
class InceptionMetrics(Metric):
    name = 'InceptionMetrics'

    def __init__(self,
                 num_images=None,
                 reference_pkl=None,
                 bgr2rgb=False,
                 center_crop=False,  # SDXL-Lightning patch FID
                 resize=True,
                 inception_args=dict(
                    type='StyleGAN',
                    inception_path=TERO_INCEPTION_URL),
                 use_kid=False,
                 use_pr=True,
                 use_is=True,
                 kid_num_subsets=100,
                 kid_max_subset_size=1000,
                 pr_k=3,
                 pr_row_batch_size=10000,
                 pr_col_batch_size=10000,
                 is_splits=10,
                 prefix=''):
        super().__init__(num_images)
        self.reference_pkl = reference_pkl
        self.real_feats = []
        self.fake_feats = []
        self.preds = []
        self.real_mean = None
        self.real_cov = None
        self.bgr2rgb = bgr2rgb
        self.center_crop = center_crop
        self.resize = resize
        self.device = 'cpu'

        if self.center_crop and self.resize:
            warnings.warn('`center_crop` is set to True, `resize` will be ignored.')

        logger = get_root_logger()
        ori_level = logger.level
        logger.setLevel('ERROR')
        self.inception_net, self.inception_style = load_inception(
            inception_args, 'FID', map_location=self.device)
        logger.setLevel(ori_level)

        self.inception_net.eval()

        self.use_kid = use_kid
        self.use_pr = use_pr
        self.use_is = use_is
        self.kid_num_subsets = kid_num_subsets
        self.kid_max_subset_size = kid_max_subset_size
        self.real_feats_np = None

        self.pr_k = pr_k
        self.pr_row_batch_size = pr_row_batch_size
        self.pr_col_batch_size = pr_col_batch_size

        self.is_splits = is_splits
        self.prefix = prefix

    def prepare(self):
        self.real_feats = []
        self.real_feats_np = None
        self.fake_feats = []
        self.preds = []
        if self.reference_pkl is not None:
            assert mmcv.is_filepath(self.reference_pkl)
            if self.reference_pkl.startswith('huggingface://'):
                self.reference_pkl = download_from_huggingface(self.reference_pkl)
            elif self.reference_pkl.startswith(('http://', 'https://')):
                self.reference_pkl = download_from_url(self.reference_pkl)
            with open(self.reference_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['real_feats_np']
                self.real_feats = [torch.from_numpy(reference['real_feats_np'])]
                self.num_real_feeded = reference['num_real_feeded']

    @staticmethod
    def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return fid, mean_norm, trace

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    def extract_features(self, batch):
        if self.center_crop:
            crop_size = 299
            h, w = batch.shape[2], batch.shape[3]
            assert h >= crop_size and w >= crop_size
            h_offset = (h - crop_size) // 2
            w_offset = (w - crop_size) // 2
            batch = batch[:, :, h_offset:h_offset + crop_size, w_offset:w_offset + crop_size]
        elif self.resize:
            batch = F.interpolate(
                batch, size=(299, 299), mode='bicubic', align_corners=False, antialias=True).clamp(min=-1, max=1)
        assert self.inception_style == 'StyleGAN'
        batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        feat = self.inception_net(batch, return_features=True)
        pred = F.linear(feat, self.inception_net.output.weight).softmax(dim=1)
        return feat, pred

    @torch.no_grad()
    def feed_op(self, batch, mode):
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]
        batch = batch.to(self.device)

        feat, pred = self.extract_features(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(feat) for _ in range(ws)]
            dist.all_gather(placeholder, feat)
            feat = torch.stack(placeholder, dim=1).reshape(feat.size(0) * ws, *feat.shape[1:])
            if mode == 'fakes':
                placeholder = [torch.zeros_like(pred) for _ in range(ws)]
                dist.all_gather(placeholder, pred)
                pred = torch.stack(placeholder, dim=1).reshape(pred.size(0) * ws, *pred.shape[1:])

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            if mode == 'reals':
                self.real_feats.append(feat.cpu())
            elif mode == 'fakes':
                self.fake_feats.append(feat.cpu())
                self.preds.append(pred.cpu().numpy())
            else:
                raise ValueError(
                    f"The expected mode should be set to 'reals' or 'fakes,\
                    but got '{mode}'")

    def feed(self, batch, mode):
        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()
            if mode == 'reals':
                if self.num_real_feeded == self.num_real_need:
                    return 0

                if isinstance(batch, dict):
                    batch_size = len(list(batch.values())[0])
                    end = min(batch_size, self.num_real_need - self.num_real_feeded)
                    batch_to_feed = {k: v[:end] for k, v in batch.items()}
                else:
                    batch_size = batch.shape[0]
                    end = min(batch_size, self.num_real_need - self.num_real_feeded)
                    batch_to_feed = batch[:end]

                global_end = min(batch_size * ws,
                                 self.num_real_need - self.num_real_feeded)
                self.feed_op(batch_to_feed, mode)
                self.num_real_feeded += global_end
                return end

            elif mode == 'fakes':
                if self.num_fake_feeded == self.num_fake_need:
                    return 0

                if isinstance(batch, dict):
                    batch_size = len(list(batch.values())[0])
                    end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                    batch_to_feed = {k: v[:end] for k, v in batch.items()}
                else:
                    batch_size = batch.shape[0]
                    end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                    batch_to_feed = batch[:end]

                global_end = min(batch_size * ws,
                                 self.num_fake_need - self.num_fake_feeded)
                self.feed_op(batch_to_feed, mode)
                self.num_fake_feeded += global_end
                return end
            else:
                raise ValueError(
                    'The expected mode should be set to \'reals\' or \'fakes\','
                    f'but got \'{mode}\'')

    @torch.no_grad()
    def summary(self):
        real_feats = torch.cat(self.real_feats, dim=0)
        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert fake_feats.shape[0] >= self.num_images
            fake_feats = fake_feats[:self.num_images]
            if self.reference_pkl is None:  # real feats not pre-calculated
                assert real_feats.shape[0] >= self.num_images
                real_feats = real_feats[:self.num_images]

        if self.real_feats_np is None:
            real_feats_np = real_feats.numpy()
            self.real_feats_np = real_feats_np
            self.real_mean = np.mean(real_feats_np, 0)
            self.real_cov = np.cov(real_feats_np, rowvar=False)

        self._result_dict = dict()

        prefix = self.prefix + '_' if len(self.prefix) > 0 else ''

        # FID
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)
        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        self._result_dict.update({f'{prefix}fid': fid})
        _result_str = f'{prefix}FID: {fid:.4f} ({mean:.4f}/{cov:.4f})'

        # KID
        if self.use_kid:
            kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.kid_num_subsets,
                                 self.kid_max_subset_size) * 1000
            self._result_dict.update({f'{prefix}kid': kid})
            _result_str += f', {prefix}KID: {kid:.4f}'
        else:
            kid = None

        # PR
        if self.use_pr:
            for name, manifold, probes in [
                (f'{prefix}precision', real_feats, fake_feats),
                (f'{prefix}recall', fake_feats, real_feats)
            ]:
                kth = []
                for manifold_batch in manifold.split(self.pr_row_batch_size):
                    distance = compute_pr_distances(
                        row_features=manifold_batch,
                        col_features=manifold,
                        col_batch_size=self.pr_col_batch_size)
                    kth.append(
                        distance.to(torch.float32).kthvalue(self.pr_k + 1).values.to(torch.float16))
                kth = torch.cat(kth)
                pred = []
                for probes_batch in probes.split(self.pr_row_batch_size):
                    distance = compute_pr_distances(
                        row_features=probes_batch,
                        col_features=manifold,
                        col_batch_size=self.pr_col_batch_size)
                    pred.append((distance <= kth).any(dim=1))
                self._result_dict[name] = float(torch.cat(pred).to(torch.float32).mean())
            precision = self._result_dict[f'{prefix}precision']
            recall = self._result_dict[f'{prefix}recall']
            _result_str += f', {prefix}Precision: {precision:.5f}, {prefix}Recall:{recall:.5f}'
        else:
            precision = recall = None

        # IS
        if self.use_is:
            split_scores = []
            self.preds = np.concatenate(self.preds, axis=0)
            if self.num_images is not None:
                assert self.preds.shape[0] >= self.num_images
                self.preds = self.preds[:self.num_images]
            num_preds = self.preds.shape[0]
            for k in range(self.is_splits):
                part = self.preds[k * (num_preds // self.is_splits):(k + 1) * (num_preds // self.is_splits), :]
                py = np.mean(part, axis=0)
                scores = []
                for i in range(part.shape[0]):
                    pyx = part[i, :]
                    scores.append(entropy(pyx, py))
                split_scores.append(np.exp(np.mean(scores)))
            is_mean = np.mean(split_scores)
            self._result_dict.update({f'{prefix}is': is_mean})
            _result_str += f', {prefix}IS: {is_mean:.2f}'
        else:
            is_mean = None

        self._result_str = _result_str

        return fid, kid, precision, recall, is_mean

    def clear_fake_data(self):
        self.fake_feats = []
        self.preds = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()
        if clear_reals:
            self.real_feats = []
            self.real_feats_np = None
            self.num_real_feeded = 0

    def load_to_gpu(self):
        """Move models to GPU."""
        if torch.cuda.is_available():
            self.inception_net.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        """Move models to CPU."""
        self.inception_net.cpu()
        self.device = 'cpu'


@METRICS.register_module()
class ColorStats(Metric):
    name = 'ColorStats'

    def __init__(self,
                 num_images=None):
        super().__init__(num_images)

    def prepare(self):
        self.stats = []

    @staticmethod
    def srgb_to_linear(c):
        threshold = 0.04045
        below = c <= threshold
        out = torch.where(
            below, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return out

    @staticmethod
    def linear_to_srgb(c):
        threshold = 0.0031308
        below = c <= threshold
        out = torch.where(
            below, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)
        return out

    def rgb_to_grayscale_srgb(self, img_srgb):
        img_lin = self.srgb_to_linear(img_srgb)
        R_lin, G_lin, B_lin = img_lin.unbind(dim=1)
        Y_lin = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
        gray_srgb = self.linear_to_srgb(Y_lin)
        return gray_srgb

    @staticmethod
    def srgb_to_hsv_saturation(img_srgb):
        c_max = torch.amax(img_srgb, dim=1)
        c_min = torch.amin(img_srgb, dim=1)
        delta = c_max - c_min
        sat = delta / c_max.clamp(min=1e-5)
        return sat

    def compute_stats(self, batch):
        batch = (batch / 2 + 0.5).clamp(0, 1)
        gray = self.rgb_to_grayscale_srgb(batch).flatten(1)
        contrast, brightness = torch.std_mean(gray, dim=1)
        saturation = self.srgb_to_hsv_saturation(batch).flatten(1).mean(dim=1)
        return torch.stack([brightness, contrast, saturation], dim=-1)

    @torch.no_grad()
    def feed_op(self, batch, mode):
        stats = self.compute_stats(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(stats) for _ in range(ws)]
            dist.all_gather(placeholder, stats)
            stats = torch.stack(placeholder, dim=1).reshape(stats.size(0) * ws, *stats.shape[1:])

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.stats.append(stats.cpu())

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            if isinstance(batch, dict):
                batch_size = len(list(batch.values())[0])
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = {k: v[:end] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = batch[:end]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        stats = torch.cat(self.stats, dim=0)
        if self.num_images is not None:
            assert stats.shape[0] >= self.num_images
            stats = stats[:self.num_images]
        stats = stats.mean(dim=0)
        brightness, contrast, saturation = stats.tolist()
        self._result_dict = dict(
            brightness=brightness, contrast=contrast, saturation=saturation)
        self._result_str = f'Brightness: {brightness:.4f}, Contrast: {contrast:.4f}, Saturation: {saturation:.4f}'
        return brightness, contrast, saturation

    def clear_fake_data(self):
        self.stats = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()


@METRICS.register_module()
class HPSv2(Metric):
    name = 'HPSv2'
    requires_prompt = True

    def __init__(self,
                 num_images=None,
                 hps_version='v2.1'):
        super().__init__(num_images)
        self.hps_version = hps_version
        self.device = 'cpu'  # Initialize on CPU
        self.dtype = torch.float16
        self.model, self.tokenizer = load_hpsv2(hps_version, device=self.device, precision='fp16')
        self.model.eval()
        image_size = self.model.visual.image_size
        if isinstance(image_size, tuple):
            assert len(image_size) == 2 and image_size[0] == image_size[1]
            image_size = image_size[0]
        self.image_size = image_size
        self.image_mean = torch.tensor(self.model.visual.image_mean, device=self.device).view(3, 1, 1)
        self.image_std = torch.tensor(self.model.visual.image_std, device=self.device).view(3, 1, 1)

    def prepare(self):
        self.scores = []

    def resize(self, imgs):
        h, w = imgs.shape[2:]
        scale = self.image_size / float(max(h, w))
        if scale != 1.0:
            h = int(round(h * scale))
            w = int(round(w * scale))
            imgs = F.interpolate(imgs, size=(h, w), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        if h != w:
            pad_h = self.image_size - h
            pad_w = self.image_size - w
            imgs = F.pad(
                imgs, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)
        return imgs

    @torch.no_grad()
    def feed_op(self, batch, mode):
        imgs = batch['imgs']
        prompts = batch['prompts']

        imgs = (imgs.to(device=self.device, dtype=torch.float32) / 2 + 0.5).clamp(0, 1)
        imgs = ((self.resize(imgs) - self.image_mean) / self.image_std).to(dtype=self.dtype)
        prompts = self.tokenizer(prompts).to(device=self.device)

        outputs = self.model(imgs, prompts)
        image_features, text_features = outputs['image_features'], outputs['text_features']
        hps_scores = (image_features * text_features).sum(dim=-1)  # (bs, )

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.empty_like(hps_scores) for _ in range(ws)]
            dist.all_gather(placeholder, hps_scores)
            hps_scores = torch.stack(placeholder, dim=1).reshape(hps_scores.size(0) * ws)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.scores.append(hps_scores.float().cpu())

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            if isinstance(batch, dict):
                batch_size = len(list(batch.values())[0])
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = {k: v[:end] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = batch[:end]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        scores = torch.cat(self.scores, dim=0)
        if self.num_images is not None:
            assert scores.shape[0] >= self.num_images
            scores = scores[:self.num_images]
        mean_score = scores.mean().item()
        self._result_dict = dict(hpsv2=mean_score)
        self._result_str = f'HPSv2: {mean_score:.4f}'
        return mean_score

    def clear_fake_data(self):
        self.scores = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    def load_to_gpu(self):
        if torch.cuda.is_available():
            self.model.cuda()
            self.image_mean = self.image_mean.cuda()
            self.image_std = self.image_std.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        self.model.cpu()
        self.image_mean = self.image_mean.cpu()
        self.image_std = self.image_std.cpu()
        self.device = 'cpu'


@METRICS.register_module()
class CLIPSimilarity(Metric):
    """
    Average image–text CLIP cosine similarity (↑ better).
    Preprocess emulates OpenAI CLIP for ViT-L/14@336:
      - Resize so min(H, W) = 336 (bicubic, antialias), keep aspect ratio
      - Center crop to 336x336
      - Normalize with model.visual.image_mean/std
    Expects batch = {'imgs': (B,3,H,W) in [-1,1], 'prompts': List[str]}
    """
    name = 'CLIPSimilarity'
    requires_prompt = True

    def __init__(
        self,
        num_images=None,
        model_name='ViT-L-14-336-quickgelu',
        pretrained='openai',
        precision='fp16',   # 'fp16' | 'fp32' | 'bf16'
    ):
        super().__init__(num_images)
        self.model_name = model_name
        self.pretrained = pretrained
        self.precision = precision

        self.device = 'cpu'
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32
        }.get(precision, torch.float16)

        self.model, self.tokenizer = load_openclip(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
            precision=precision,
        )
        self.model.eval()

        # OpenAI ViT-L/14@336 uses square 336 input
        image_size = self.model.visual.image_size
        if isinstance(image_size, tuple):
            assert len(image_size) == 2 and image_size[0] == image_size[1]
            image_size = image_size[0]
        self.image_size = int(image_size)  # 336

        # Use the model's own stats for normalization
        self.image_mean = torch.tensor(self.model.visual.image_mean, device=self.device).view(3, 1, 1)
        self.image_std = torch.tensor(self.model.visual.image_std, device=self.device).view(3, 1, 1)

    def prepare(self):
        self.scores = []

    def _resize_min_side_then_center_crop(self, imgs):
        """
        imgs: (B,3,H,W) in [0,1], float32, on self.device
        1) Resize so min(H,W) == self.image_size, preserve AR (bicubic, antialias)
        2) Center-crop to (self.image_size, self.image_size)
        3) Normalize with model mean/std
        4) Cast to self.dtype
        """
        _, _, H, W = imgs.shape
        target = self.image_size

        # Scale factor so that the shorter side becomes 'target'
        short, long = (H, W) if H < W else (W, H)
        if short == 0:
            raise ValueError("Invalid image with zero dimension.")
        scale = target / float(short)

        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        if new_h != H or new_w != W:
            imgs = F.interpolate(
                imgs, size=(new_h, new_w),
                mode='bicubic', align_corners=False, antialias=True
            ).clamp(0, 1)

        # Center crop to target x target
        top = max(0, (new_h - target) // 2)
        left = max(0, (new_w - target) // 2)
        imgs = imgs[:, :, top:top + target, left:left + target]

        imgs = (imgs - self.image_mean) / self.image_std
        return imgs.to(dtype=self.dtype)

    @torch.no_grad()
    def feed_op(self, batch, mode):
        if mode == 'reals':
            return 0

        imgs = batch['imgs']
        prompts = batch['prompts']

        # [-1,1] -> [0,1]
        imgs = (imgs.to(device=self.device, dtype=torch.float32) / 2 + 0.5).clamp(0, 1)
        imgs = self._resize_min_side_then_center_crop(imgs)

        # Tokenize on device
        text = self.tokenizer(prompts).to(device=self.device)

        # Forward (create_model(..., output_dict=True)) => dict w/ features
        out = self.model(imgs, text)
        if isinstance(out, dict) and ('image_features' in out and 'text_features' in out):
            img_feat = out['image_features']
            txt_feat = out['text_features']
        else:
            img_feat = self.model.encode_image(imgs)
            txt_feat = self.model.encode_text(text)

        # Cosine similarity per pair
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        sim = (img_feat * txt_feat).sum(dim=-1).to(torch.float32)  # (B,)

        # DDP gather
        if dist.is_initialized():
            ws = dist.get_world_size()
            bucket = [torch.empty_like(sim) for _ in range(ws)]
            dist.all_gather(bucket, sim)
            sim = torch.stack(bucket, dim=1).reshape(sim.size(0) * ws)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.scores.append(sim.cpu())

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            if isinstance(batch, dict):
                batch_size = len(list(batch.values())[0])
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = {k: v[:end] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = batch[:end]

            global_end = min(batch_size * ws, self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        sims = torch.cat(self.scores, dim=0)
        if self.num_images is not None:
            assert sims.shape[0] >= self.num_images
            sims = sims[:self.num_images]
        mean_sim = sims.mean().item()

        self._result_dict = dict(clipsim=mean_sim)  # raw cosine in [-1,1]
        self._result_str = f'CLIPSim: {mean_sim:.4f}'
        return mean_sim

    def clear_fake_data(self):
        self.scores = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    def load_to_gpu(self):
        if torch.cuda.is_available():
            self.model.cuda()
            self.image_mean = self.image_mean.cuda()
            self.image_std = self.image_std.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        self.model.cpu()
        self.image_mean = self.image_mean.cpu()
        self.image_std = self.image_std.cpu()
        self.device = 'cpu'
