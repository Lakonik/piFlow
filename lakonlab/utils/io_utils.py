# Copyright (c) 2025 Hansheng Chen

import os
import time
import mimetypes
import tempfile
import subprocess
import numpy as np
import imageio
import boto3
import mmcv
import torch.distributed as dist
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
from functools import wraps
from typing import Generator, Union, Optional, Tuple
from PIL import Image
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from torch.hub import download_url_to_file
from huggingface_hub import hf_hub_download
from mmcv.fileio import BaseStorageBackend, FileClient
from mmgen.utils.io_utils import MMGEN_CACHE_DIR


AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
S3_MULTIPART_THRESHOLD = 5 * 2**30  # 5GB

TMP_DIR = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
S3_TRANSFER_CONFIG = TransferConfig(multipart_threshold=S3_MULTIPART_THRESHOLD)


def retry(tries=5, delay=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries:
                        print(f"Attempt {attempt} failed: {e}. No more retries.")
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


@retry()
def download_from_url(url,
                      dest_path=None,
                      dest_dir=MMGEN_CACHE_DIR,
                      hash_prefix=None):
    """Modified from MMGeneration.
    """
    # get the exact destination path
    if dest_path is None:
        filename = url.split('/')[-1]
        dest_path = os.path.join(dest_dir, filename)

    if dest_path.startswith('~'):
        dest_path = os.path.expanduser('~') + dest_path[1:]

    # advoid downloading existed file
    if os.path.exists(dest_path):
        return dest_path

    is_dist = dist.is_available() and dist.is_initialized()

    if is_dist:
        local_rank = dist.get_node_local_rank()
    else:
        local_rank = 0

    # only download from the master process
    if local_rank == 0:
        # mkdir
        _dir = os.path.dirname(dest_path)
        mmcv.mkdir_or_exist(_dir)
        download_url_to_file(url, dest_path, hash_prefix, progress=True)

    # sync the other processes
    if is_dist:
        dist.barrier()

    return dest_path


@retry()
def download_from_huggingface(filename):
    filename = filename.replace('huggingface://', '').split('/')
    repo_id = '/'.join(filename[:2])
    repo_filename = '/'.join(filename[2:])
    is_dist = dist.is_available() and dist.is_initialized()
    if is_dist:
        local_rank = dist.get_node_local_rank()
    else:
        local_rank = 0
    if local_rank == 0:
        cached_file = hf_hub_download(
            repo_id=repo_id, filename=repo_filename)
    if is_dist:
        dist.barrier()
    if local_rank > 0:
        cached_file = hf_hub_download(
            repo_id=repo_id, filename=repo_filename)
    return cached_file


class S3Backend(BaseStorageBackend):

    _allow_symlink = True

    def __init__(self, anonymous: bool = False):
        if anonymous:
            config = Config(region_name=AWS_REGION, signature_version=UNSIGNED)
        else:
            config = Config(region_name=AWS_REGION)
        self._client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
            config=config)

    def __del__(self):
        self._client.close()

    @staticmethod
    def _split_s3_url(s3_url):
        s3_url = s3_url.removeprefix('s3://')
        bucket, _, prefix = s3_url.partition('/')
        return bucket, prefix

    @staticmethod
    def _infer_s3_extra_args(filepath: str) -> dict:
        extra_args = dict()
        ctype, enc = mimetypes.guess_type(str(filepath), strict=False)
        if ctype is not None:
            extra_args.update(ContentType=ctype)
        if enc == 'gzip':
            extra_args['ContentEncoding'] = 'gzip'
        return extra_args

    @retry()
    def get(self, filepath: Union[str, Path]) -> bytes:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        bytesio = BytesIO()
        self._client.download_fileobj(bucket, prefix, bytesio, Config=S3_TRANSFER_CONFIG)
        bytesio.seek(0)
        return bytesio.read()

    def get_text(self, filepath: Union[str, Path], encoding: str = 'utf-8') -> str:
        return self.get(filepath).decode(encoding)

    @retry()
    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        extra_args = self._infer_s3_extra_args(filepath)
        if len(obj) < S3_MULTIPART_THRESHOLD:
            bucket, prefix = self._split_s3_url(filepath)
            self._client.upload_fileobj(
                BytesIO(obj),
                bucket,
                prefix,
                Config=S3_TRANSFER_CONFIG,
                ExtraArgs=extra_args,
            )
        else:
            ext = os.path.splitext(filepath)[-1].lower()
            with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=ext, delete=False) as tmp:
                cached_file = tmp.name
                tmp.write(obj)
            try:
                cmd = ['aws', 's3', 'cp', cached_file, filepath]
                if 'ContentType' in extra_args:
                    cmd += ['--content-type', extra_args['ContentType']]
                if 'ContentEncoding' in extra_args:
                    cmd += ['--content-encoding', extra_args['ContentEncoding']]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            finally:
                os.remove(cached_file)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        self.put(bytes(obj, encoding=encoding), filepath)

    @retry()
    def remove(self, filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        self._client.delete_object(Bucket=bucket, Key=prefix)

    @retry()
    def exists(self, filepath: Union[str, Path]) -> bool:
        filepath = str(filepath)
        bucket, prefix = self._split_s3_url(filepath)
        if filepath[-1] == '/':
            s3_objects = self._client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/',
                MaxKeys=2)
            files = []
            if 'Contents' in s3_objects:
                files += [obj['Key'] for obj in s3_objects['Contents']]
            if 'CommonPrefixes' in s3_objects:
                files += [obj['Prefix'] for obj in s3_objects['CommonPrefixes']]
            exist_status = len(files) > 0
        else:
            try:
                self._client.head_object(Bucket=bucket, Key=prefix)
                exist_status = True
            except ClientError as e:
                code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                err = e.response.get('Error', {}).get('Code')
                if code == 404 or err in ('404', 'NoSuchKey', 'NotFound'):
                    exist_status = False
                else:
                    raise
        return exist_status

    def isdir(self, filepath: Union[str, Path]) -> bool:
        filepath = str(filepath)
        if not filepath.endswith('/'):
            filepath += '/'
        return self.exists(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        filepath = str(filepath)
        return filepath[-1] != '/' and self.exists(filepath)

    @staticmethod
    def join_path(filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        return os.path.join(str(filepath), *(str(p) for p in filepaths))

    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str, Path],
            **kwargs) -> Generator[Union[str, Path], None, None]:
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False, **kwargs)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    @retry()
    def list_dir_or_file(
            self,
            dir_path: Union[str, Path],
            list_dir: bool = True,
            list_file: bool = True,
            suffix: Optional[Union[str, Tuple[str]]] = None,
            recursive: bool = False):
        assert list_dir and list_file and suffix is None

        dir_path = str(dir_path)
        if not dir_path.endswith('/'):
            dir_path += '/'

        cmd = ['aws', 's3', 'ls', dir_path]
        if recursive:
            cmd.append('--recursive')
            bucket, prefix = self._split_s3_url(dir_path)
            prefix_len = len(prefix)
        else:
            prefix_len = 0

        out = subprocess.check_output(cmd, text=True)

        names = []
        for line in out.splitlines():
            if not line:
                continue

            ls = line.lstrip()

            if not recursive and ls.startswith('PRE '):
                name = ls.split(maxsplit=1)[1].rstrip('/')
            else:
                parts = ls.split(maxsplit=3)
                if len(parts) < 4:
                    continue
                key = parts[3]
                name = key[prefix_len:] if prefix_len else key

            names.append(name)

        return names


FileClient.register_backend(name='s3', backend=S3Backend, force=True, prefixes='s3')


def save_image(image, filepath, file_client):
    img_byte_arr = BytesIO()
    Image.fromarray(image).save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    file_client.put(img_byte_arr, filepath)


def save_video(video, filepath, file_client, fps=16, quality=5, bitrate=None, macro_block_size=16):
    imageio.plugins.ffmpeg.get_exe()
    img_byte_arr = BytesIO()
    with imageio.get_writer(
            img_byte_arr, format='mp4', mode='I', fps=fps,
            quality=quality, bitrate=bitrate, macro_block_size=macro_block_size) as writer:
        for frame in video:
            writer.append_data(frame)
    img_byte_arr = img_byte_arr.getvalue()
    file_client.put(img_byte_arr, filepath)


def resize_and_crop(image: Image, target_hw: tuple[int, int]):
    tgt_h, tgt_w = target_hw
    w, h = image.size
    scale = max(tgt_h / h, tgt_w / w)
    new_h, new_w = round(h * scale), round(w * scale)
    if new_h != h or new_w != w:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - tgt_w) // 2, (new_h - tgt_h) // 2
    return image.crop((left, top, left + tgt_w, top + tgt_h)), scale


def load_image(filepath, file_client, target_size=None):
    img_bytes = file_client.get(filepath)
    extension = os.path.splitext(filepath)[-1].lower()
    arr = imageio.v3.imread(BytesIO(img_bytes), extension=extension)  # (H,W,C) or (H,W)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    if target_size is not None:
        assert arr.ndim == 3
        pil = Image.fromarray(arr)
        pil, _ = resize_and_crop(pil, target_size)
        arr = np.asarray(pil)
    return arr


def load_images_parallel(filepaths, file_client):
    futures = []
    results = [None] * len(filepaths)
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 4) as pool:
        for idx, abs_path in enumerate(filepaths):
            fut = pool.submit(load_image, abs_path, file_client)
            futures.append((idx, fut))

        for idx, fut in futures:
            arr = fut.result()
            results[idx] = arr
    return results


@retry()
def hf_model_downloader(model_class, repo_id, **kwargs):
    is_dist = dist.is_available() and dist.is_initialized()
    if is_dist:
        local_rank = dist.get_node_local_rank()
    else:
        local_rank = 0
    if local_rank == 0:
        model = model_class.from_pretrained(repo_id, **kwargs)
    if is_dist:
        dist.barrier()
    if local_rank > 0:
        model = model_class.from_pretrained(repo_id, **kwargs)
    return model


def hf_model_loader(model_class, repo_id, local_files_only=False, **kwargs):
    try:
        model = model_class.from_pretrained(repo_id, local_files_only=True, **kwargs)
    except Exception as e:
        if local_files_only:
            raise e
        else:
            is_dist = dist.is_available() and dist.is_initialized()
            if is_dist:
                dist.barrier()
            model = hf_model_downloader(model_class, repo_id, **kwargs)
    return model
