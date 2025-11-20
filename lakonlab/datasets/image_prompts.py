# Copyright (c) 2025 Hansheng Chen

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
import pickle
import gzip
import orjson
import mmcv
import torch.storage
torch.storage.UntypedStorage.dtype = torch.uint8  # hot patch for torch 2.6 deserialization

from io import BytesIO
from typing import Optional, Tuple, Union
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from mmcv.fileio import FileClient
from mmcv.parallel import DataContainer as DC
from mmgen.utils import get_root_logger
from mmgen.datasets.builder import DATASETS
from lakonlab.utils.io_utils import load_image


@DATASETS.register_module()
class ImagePrompt(Dataset):
    """Initialize an image/prompt dataset that reads either cached pickled records
    (zstd-compressed) or a HuggingFace prompt dataset (optionally paired with images).

    Args:
        data_root (str): Root path for IO, resolved via `mmcv.FileClient`.
        cache_dir (Optional[str]): Subdirectory of `data_root` containing `.zst`
            cache shards. Enables cache mode when provided and exists. Caches must
            contain pickled dicts with keys `"prompt"` and `"prompt_embed_kwargs"`,
            and optionally `"latents"` or `"latent_size"`.
        cache_datalist_path (Optional[str]): Optional datalist path for `cache_dir`.
            Supports `.jsonl`, `.jsonl.gz`, or `.json`. If not exists, files are
            discovered by listing the directory.
        ignore_cached_latents (bool): If True, ignores any cached latents and
            prioritizes loading images from `image_dir`. Defaults to False.
        prompt_dataset_kwargs (Optional[dict]): Keyword arguments forwarded to
            `datasets.load_dataset(...)`. Enables prompt-dataset mode when provided.
            If a `DatasetDict` is returned, a split (e.g., "train") is selected
            internally.
        image_dir (Optional[str]): Subdirectory of `data_root` with images to pair
            with prompts (used only in prompt-dataset mode).
        image_datalist_path (Optional[str]): Optional datalist for `image_dir`
            (same formats as above). When `bucketize=True`, JSONL entries must
            include `"size_idx"`.
        image_extension (Optional[str]): Image file extension used to compose
            paths when `image_dir` is set. Defaults to ".png".
        image_scale_factor (float): Scale factor applied to image spatial dimensions
            after loading. Defaults to 1.0 (no scaling).
        negative_prompt_embeds_path (Optional[str]): Path to a `torch.load`-able
            file containing keyward arguments forwarded to the diffusion model
            for negative prompt embeddings. Added to each sample when provided.
        negative_prompt_kwargs (Optional[dict]): Keyword arguments forwarded to
            the text encoder for negative prompts. Added to each sample when provided.
        pad_seq_len (int): If set, pads/truncates `"encoder_hidden_states"` and
            `"encoder_hidden_states_mask"` along the sequence dimension to this
            length. Defaults to None.
        latent_size (Optional[Tuple[int]]): Default latent shape `(C, H, W)` used
            when no cached latents exist and no image size is provided. Defaults to
            `(16, 128, 128)`.
        vae_scale_factor (Optional[Union[int, Tuple[int]]]): Downscale factor(s)
            applied to image dimensions when deriving latent sizes from image size.
            If an `int`, applies to each spatial dim; if a `tuple`, its length must
            match the provided image spatial size (e.g., `(H, W)` or `(T, H, W)` for
            video VAEs).
        repeat (int): Virtual repetition factor for each underlying sample.
            Affects `__len__` and index mapping.
        start_ind (Optional[int]): Start index (inclusive) into the underlying
            dataset. Defaults to 0.
        end_ind (int): End index (exclusive) into the underlying dataset. Defaults
            to dataset length.
        bucketize (bool): If True, enables bucketing in `DistributedSampler` so that
            each rank receives samples of the same size. Expects `"size_idx"` in JSONL
            datalists and collects bucket ids. Defaults to False.
        test_mode (bool): If True, return deterministic noise per sample instead of
            reading/allocating real latents or images.
    """

    PROMPT_KEY_MAPS = {
        'prompt_embeds': 'encoder_hidden_states',
        'prompt_embeds_scale': 'encoder_hidden_states_scale',
        'pooled_prompt_embeds': 'pooled_projections',
        'prompt_embeds_mask': 'encoder_hidden_states_mask'
    }

    def __init__(self,
                 data_root: str,
                 cache_dir: Optional[str] = None,
                 cache_datalist_path: Optional[str] = None,
                 ignore_cached_latents: bool = False,
                 prompt_dataset_kwargs: Optional[dict] = None,
                 image_dir: Optional[str] = None,
                 image_datalist_path: Optional[str] = None,
                 image_extension: Optional[str] = '.png',
                 image_scale_factor: float = 1.0,
                 negative_prompt_embeds_path: Optional[str] = None,
                 negative_prompt_kwargs: Optional[dict] = None,
                 pad_seq_len: int = None,
                 latent_size: Optional[Tuple[int]] = (16, 128, 128),
                 vae_scale_factor: Optional[Union[int, Tuple[int]]] = 8,
                 repeat: int = 1,
                 start_ind: Optional[int] = None,
                 end_ind: int = None,
                 bucketize: bool = False,
                 test_mode: bool = False):
        super().__init__()
        self.data_root = data_root
        self.file_client = FileClient.infer_client(uri=self.data_root)

        self.pad_seq_len = pad_seq_len

        self.cache_dir_path = self.cache_datalist_path = None
        self.prompt_dataset = self.image_dir_path = self.image_datalist_path = None
        self.image_extension = image_extension
        self.image_scale_factor = image_scale_factor
        self.ignore_cached_latents = ignore_cached_latents
        self.bucketize = bucketize
        bucket_ids = None

        if (cache_dir is not None
                and self.file_client.isdir(self.file_client.join_path(data_root, cache_dir))
                and cache_datalist_path is not None
                and FileClient.infer_client(uri=cache_datalist_path).isfile(cache_datalist_path)):
            self.cache_dir_path = self.file_client.join_path(data_root, cache_dir)
            self.cache_datalist, bucket_ids = self.parse_datalist(
                self.cache_dir_path, cache_datalist_path)
            dataset_len = len(self.cache_datalist)

        elif prompt_dataset_kwargs is not None:
            self.prompt_dataset = load_dataset(**prompt_dataset_kwargs)
            if isinstance(self.prompt_dataset, DatasetDict):
                split = 'train' if 'train' in self.prompt_dataset else list(self.prompt_dataset.keys())[0]
                self.prompt_dataset = self.prompt_dataset[split]
            assert isinstance(self.prompt_dataset, HFDataset), \
                f"Expected HF Dataset/DatasetDict, got {type(self.prompt_dataset)}."
            dataset_len = len(self.prompt_dataset)

        else:
            raise ValueError('Either `cache_dir` or `prompt_dataset_kwargs` must be provided.')

        if image_dir is not None and self.file_client.isdir(self.file_client.join_path(data_root, image_dir)):
            self.image_dir_path = self.file_client.join_path(data_root, image_dir)
            self.image_datalist, bucket_ids = self.parse_datalist(
                self.image_dir_path, image_datalist_path, datalist_must_exist=True)
            assert dataset_len == len(self.image_datalist)

        if bucket_ids is None and self.bucketize:
            assert self.prompt_dataset is not None
            bucket_ids = self.get_bucket_ids_from_prompt_dataset()

        self.negative_prompt_embed_kwargs = None
        if negative_prompt_embeds_path is not None:
            negative_prompt_embeds_bytesio = BytesIO(
                FileClient.infer_client(uri=negative_prompt_embeds_path).get(negative_prompt_embeds_path))
            self.negative_prompt_embed_kwargs = self.parse_prompt_embeds(
                torch.load(negative_prompt_embeds_bytesio, map_location='cpu'))
        self.negative_prompt_kwargs = negative_prompt_kwargs

        self.latent_size = latent_size
        self.vae_scale_factor = vae_scale_factor

        self.repeat = repeat
        if start_ind is not None:
            start_ind = max(min(start_ind, dataset_len - 1), -dataset_len) % dataset_len
        else:
            start_ind = 0
        if end_ind is not None:
            end_ind = max(min(end_ind - 1, dataset_len - 1), -dataset_len) % dataset_len + 1
        else:
            end_ind = dataset_len
        assert start_ind < end_ind, f'Invalid start_ind and end_ind.'
        self.start_ind = start_ind
        self.end_ind = end_ind

        if self.bucketize:
            assert bucket_ids is not None and len(bucket_ids) == dataset_len
            self.bucket_ids = [bucket_ids[self._map_idx(i)] for i in range(len(self))]

        self.test_mode = test_mode

    def get_bucket_ids_from_prompt_dataset(self):
        ds = self.prompt_dataset
        assert 'height' in ds.column_names and 'width' in ds.column_names, \
            'When bucketize=True and no datalist is provided, the prompt dataset ' \
            'must contain `height` and `width` columns.'
        cols = ['height', 'width']
        if 'frames' in ds.column_names:
            cols = ['frames'] + cols
        ds_arrow = ds.with_format('arrow', columns=cols)
        batch = ds_arrow[:]

        arrs = [batch[c].combine_chunks().to_numpy(zero_copy_only=False) for c in cols]
        arrs = np.stack(arrs, axis=1)

        _, inv = np.unique(arrs, axis=0, return_inverse=True)
        return inv.tolist()

    def parse_datalist(self, dir_path, datalist_path=None, datalist_must_exist=False):
        logger = get_root_logger()

        if datalist_path is not None and FileClient.infer_client(uri=datalist_path).isfile(datalist_path):
            filenames = []
            bucket_ids = []

            datalist_bytesio = BytesIO(FileClient.infer_client(uri=datalist_path).get(datalist_path))
            if datalist_path.endswith('.jsonl.gz') or datalist_path.endswith('.jsonl'):
                if datalist_path.endswith('.jsonl.gz'):
                    with gzip.open(datalist_bytesio, 'rt', encoding='utf-8') as f:
                        datalist = f.readlines()
                else:
                    datalist = datalist_bytesio.read().decode('utf-8').splitlines()
                for line in datalist:
                    data_item = orjson.loads(line)
                    if 'filename' in data_item:
                        filenames.append(data_item['filename'])
                    elif 'image_hash' in data_item:
                        filenames.append(data_item['image_hash'])
                    else:
                        raise ValueError('No valid key to identify data item.')
                    if self.bucketize:
                        if 'size_idx' in data_item:
                            bucket_ids.append(data_item['size_idx'])
                        elif 'bucket_id' in data_item:
                            bucket_ids.append(data_item['bucket_id'])
                        else:
                            raise ValueError(
                                'Either `size_idx` or `bucket_id` must be present in datalist for bucketize.')
            elif datalist_path.endswith('.json'):
                assert not self.bucketize, 'Bucketize not supported for json datalist.'
                datalist = orjson.loads(datalist_bytesio.read())
                for data_item in datalist:
                    filenames.append(os.path.splitext(os.path.basename(data_item))[0])
            else:
                raise ValueError('Datalist file must be .jsonl, .jsonl.gz or .json')

        else:
            assert not datalist_must_exist, f'Datalist file {datalist_path} does not exist.'
            assert not self.bucketize, 'Bucketize not supported when datalist is not provided.'
            mmcv.print_log(
                f'Datalist file {datalist_path} does not exist, directly list all files in the directory.',
                logger=logger,
                level=logging.WARNING)
            # list all files in the directory
            filenames = [os.path.splitext(p)[0] for p in self.file_client.list_dir_or_file(dir_path)]
            filenames.sort()
            bucket_ids = None
            # save the datalist if datalist_path is provided
            if datalist_path is not None:
                if datalist_path.endswith('.jsonl.gz') or datalist_path.endswith('.jsonl'):
                    datalist = []
                    for filename in filenames:
                        datalist.append(orjson.dumps({'filename': filename}).decode('utf-8'))
                    datalist_str = '\n'.join(datalist)
                    if datalist_path.endswith('.jsonl.gz'):
                        datalist_bytesio = BytesIO()
                        with gzip.open(datalist_bytesio, 'wt', encoding='utf-8') as f:
                            f.write(datalist_str)
                        FileClient.infer_client(uri=datalist_path).put(datalist_bytesio.getvalue(), datalist_path)
                    else:
                        FileClient.infer_client(uri=datalist_path).put_text(datalist_str, datalist_path)
                elif datalist_path.endswith('.json'):
                    datalist = filenames
                    FileClient.infer_client(uri=datalist_path).put_text(
                        orjson.dumps(datalist).decode('utf-8'), datalist_path)

        mmcv.print_log(f'Loaded {len(filenames)} samples.', logger=logger)

        return filenames, bucket_ids

    def pad_prompt_embeds(self, prompt_embeds):
        if self.pad_seq_len is not None:
            if prompt_embeds.size(0) > self.pad_seq_len:
                prompt_embeds = prompt_embeds[:self.pad_seq_len]
            else:
                zeros_size = (self.pad_seq_len - prompt_embeds.size(0),) + prompt_embeds.shape[1:]
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds.new_zeros(zeros_size)], dim=0)
        return prompt_embeds

    def parse_prompt_embeds(self, data):
        prompt_embed_kwargs = data.get('prompt_embed_kwargs', {}).copy()

        # Map legacy keys to new ones if not already present
        for legacy_key, new_key in self.PROMPT_KEY_MAPS.items():
            if legacy_key in data and new_key not in prompt_embed_kwargs:
                prompt_embed_kwargs[new_key] = data[legacy_key]

        # Common post-processing
        encoder_hidden_states_scale = prompt_embed_kwargs.pop('encoder_hidden_states_scale', None)
        if 'encoder_hidden_states' in prompt_embed_kwargs:
            encoder_hidden_states = prompt_embed_kwargs['encoder_hidden_states'].float()
            if encoder_hidden_states_scale is not None:
                encoder_hidden_states = encoder_hidden_states * encoder_hidden_states_scale
            prompt_embed_kwargs['encoder_hidden_states'] = self.pad_prompt_embeds(encoder_hidden_states)

        if 'pooled_projections' in prompt_embed_kwargs:
            prompt_embed_kwargs['pooled_projections'] = prompt_embed_kwargs['pooled_projections'].float()

        if 'encoder_hidden_states_mask' in prompt_embed_kwargs:
            prompt_embed_kwargs['encoder_hidden_states_mask'] = self.pad_prompt_embeds(
                prompt_embed_kwargs['encoder_hidden_states_mask'])

        return prompt_embed_kwargs

    def calculate_latent_size(self, image_spatial_size):
        if isinstance(self.vae_scale_factor, int):
            latent_spatial_size = tuple(s // self.vae_scale_factor for s in image_spatial_size)
        else:
            assert len(self.vae_scale_factor) == len(image_spatial_size)
            latent_spatial_size = tuple(
                s // f for s, f in zip(image_spatial_size, self.vae_scale_factor))
        latent_size = (self.latent_size[0],) + latent_spatial_size
        return latent_size

    def calculate_scaled_image_size(self, image_spatial_size):
        if self.image_scale_factor != 1:
            if len(image_spatial_size) == 2:
                new_spatial_size = (int(round(image_spatial_size[0] * self.image_scale_factor)),
                                    int(round(image_spatial_size[1] * self.image_scale_factor)))
            elif len(image_spatial_size) == 3:
                new_spatial_size = (image_spatial_size[0],
                                    int(round(image_spatial_size[1] * self.image_scale_factor)),
                                    int(round(image_spatial_size[2] * self.image_scale_factor)))
            else:
                raise ValueError(f'Unsupported image spatial size {image_spatial_size}.')
        else:
            new_spatial_size = image_spatial_size
        return new_spatial_size

    def scale_image(self, image):
        if self.image_scale_factor != 1:
            new_spatial_size = self.calculate_scaled_image_size(image.shape[1:])
            if len(new_spatial_size) == 2:
                image = F.interpolate(
                    image[None], size=new_spatial_size, mode='bicubic', align_corners=False, antialias=True
                )[0].clamp(min=0, max=1)
            elif len(new_spatial_size) == 3:
                image = F.interpolate(
                    image, size=new_spatial_size[1:], mode='bicubic', align_corners=False, antialias=True
                ).clamp(min=0, max=1)
            else:
                raise ValueError(f'Unsupported image spatial size {image.shape[1:]}.')
        return image

    def _map_idx(self, idx):
        return self.start_ind + (idx // self.repeat)

    def __len__(self):
        return self.repeat * (self.end_ind - self.start_ind)

    def __getitem__(self, idx):
        mapped_idx = self._map_idx(idx)

        prompt_data = None

        if self.cache_dir_path is not None:
            data_path = self.file_client.join_path(
                self.cache_dir_path, f'{self.cache_datalist[mapped_idx]}.zst')
            data_bytesio = BytesIO(self.file_client.get(data_path))
            with zstd.ZstdDecompressor().stream_reader(data_bytesio) as f:
                raw_data = pickle.load(f)
            data = dict(
                ids=DC(idx, cpu_only=True),
                name=DC(raw_data['prompt'], cpu_only=True),
                prompt_embed_kwargs=self.parse_prompt_embeds(raw_data))

            if not self.ignore_cached_latents:  # load latents
                if 'latents' in raw_data:
                    latents = raw_data['latents']
                    if self.test_mode:
                        data['noise'] = torch.randn(
                            latents.size(), dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
                    else:
                        data['latents'] = latents.float()
                        latents_scale = raw_data.get('latents_scale', None)
                        if latents_scale is not None:
                            data['latents'] = data['latents'] * latents_scale
                else:
                    latent_size = raw_data.get('latent_size', self.latent_size)
                    if self.test_mode:
                        data['noise'] = torch.randn(
                            latent_size, dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
                    else:
                        data['latents'] = torch.empty(latent_size, dtype=torch.float32)

        else:
            prompt_data = self.prompt_dataset[mapped_idx]
            if 'prompt_kwargs' in prompt_data:
                prompt_kwargs = {k: DC(v, cpu_only=True) for k, v in prompt_data['prompt_kwargs'].items()}
            else:
                prompt_kwargs = dict(prompt=DC(prompt_data['prompt'], cpu_only=True))
            data = dict(
                ids=DC(idx, cpu_only=True),
                name=DC(prompt_data['prompt'], cpu_only=True),
                prompt_kwargs=prompt_kwargs)

        if self.image_dir_path is not None:
            image_path = self.file_client.join_path(
                self.image_dir_path, self.image_datalist[mapped_idx] + self.image_extension)
            image = load_image(image_path, self.file_client)
            image = np.moveaxis(image, -1, 0)  # channel first
            if self.test_mode:
                data['noise'] = torch.randn(
                    self.calculate_latent_size(self.calculate_scaled_image_size(image.shape[1:])),
                    dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
            else:
                images = torch.from_numpy(image)
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                assert torch.is_floating_point(images), f'Image dtype {images.dtype} not supported.'
                data['images'] = self.scale_image(images.float())
        elif 'latents' not in data and 'noise' not in data:  # allocate latents if not already loaded
            if prompt_data is not None and 'height' in prompt_data and 'width' in prompt_data:
                image_spatial_size = (prompt_data['height'], prompt_data['width'])
                if 'frames' in prompt_data:
                    image_spatial_size = (prompt_data['frames'],) + image_spatial_size
                latent_size = self.calculate_latent_size(self.calculate_scaled_image_size(image_spatial_size))
            else:
                latent_size = self.latent_size
            if self.test_mode:
                data['noise'] = torch.randn(
                    latent_size, dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
            else:
                data['latents'] = torch.empty(latent_size, dtype=torch.float32)

        if self.negative_prompt_embed_kwargs is not None:
            data.update(negative_prompt_embed_kwargs=self.negative_prompt_embed_kwargs)
        if self.negative_prompt_kwargs is not None:
            data.update(negative_prompt_kwargs=self.negative_prompt_kwargs)

        return data
