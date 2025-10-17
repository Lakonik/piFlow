## Distilling Qwen-Image

### Before Training: Data Preparation

We release the [prompt-only dataset](https://huggingface.co/datasets/Lakonik/t2i-prompts-3m) to reproduce the data-free training (without real images) in the paper. It is highly recommended to preprocess the prompts into prompt embeddings using the text encoder before training. Run the following command to preprocess the prompts using DDP on 1 node with 8 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/cache_image_prompt_data.py configs/piqwen/gmqwen_k8_datafree_piid_4step_32gpus.py --text-encoder configs/piqwen/_text_encoder.py --max-size 2304128 --launcher pytorch --diff_seed
```
By default, the preprocessed data will be saved to `data/t2i_prompts_3m/preproc_qwen/`, which requires 380GB of storage space. You can change the `data_root` in the [config file](_data_trainval.py) to save the data to a different location (AWS S3 URLs are supported).

If you choose not to preprocess the prompts, please modify the [training config file](gmqwen_k8_datafree_piid_4step_32gpus.py) to add `'./_text_encoder.py'` to the `_base_` list, which will load the text encoder during training.

### Training

Run the following command to train the GM-Qwen model under the data-free setting using DDP on 4 nodes with 8 GPUs each:
```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<NODE_RANK> --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> tools/train.py configs/piqwen/gmqwen_k8_datafree_piid_4step_32gpus.py --launcher pytorch --diff_seed
```

The above config specifies a training batch size of 2 images per GPU, and the gradients are accumulated across 4 student steps, effectively making the batch size 8 images per GPU. This requires ~70GB of VRAM per GPU. FSDP can be enabled to reduce VRAM usage to ~20GB per GPU. To use FSDP, modify the [training config file](gmqwen_k8_datafree_piid_4step_32gpus.py) to replace `'./_ddp_train.py'` with `'./_fsdp_train.py'` in the `_base_` list.

32 GPUs are required to reproduce the total batch size of 256 in the paper. If you do not care about exact reproduction, using a smaller batch size with fewer GPUs is totally fine.

### Evaluation (COCO and HPSv2)

Before evaluation, run the following command to preprocess the evaluation prompts (similar to the [Before Training: Data Preparation](#before-training-data-preparation) step):
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/cache_image_prompt_data.py configs/piqwen/gmqwen_k8_datafree_piid_4step_test.py --text-encoder configs/piqwen/_text_encoder.py --launcher pytorch --diff_seed
```

Run the following command to evaluate a pretrained model (downloaded automatically) using DDP on 1 node with 8 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py <PATH_TO_CONFIG> --launcher pytorch --diff_seed
```
where `<PATH_TO_CONFIG>` can be one of the following:
- `configs/piqwen/gmqwen_k8_piid_4step_test.py` (4-NFE GM-Qwen)
- `configs/piqwen/gmqwen_k8_datafree_piid_4step_test.py` (4-NFE GM-Qwen, data-free)
- `configs/piqwen/dxqwen_n10_piid_4step_test.py` (4-NFE DX-Qwen)

To enable FSDP evaluation, please add `'./_fsdp_test.py'` to the `_base_` list in the corresponding config file.

To evaluate a custom checkpoint, add the `--ckpt <PATH_TO_CKPT>` argument:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py <PATH_TO_CONFIG> --ckpt <PATH_TO_CKPT> --launcher pytorch --diff_seed
```
