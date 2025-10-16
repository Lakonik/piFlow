## Distilling ImageNet DiT

### Training
Before training, please follow the instructions [here](../gmflow/README.md#before-training-data-preparation) to prepare the ImageNet dataset.

Run the following command to train the model using DDP on 1 node with 8 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/train.py <PATH_TO_CONFIG> --launcher pytorch --diff_seed
```
where `<PATH_TO_CONFIG>` can be one of the following:
- `configs/piflow_imagenet/gmdit_k32_imagenet_piid_1step_8gpus.py` (1-NFE, FM teacher)
- `configs/piflow_imagenet/gmrepa_k32_imagenet_piid_1step_8gpus.py` (1-NFE, REPA teacher)
- `configs/piflow_imagenet/gmrepa_k32_imagenet_piid_2step_8gpus.py` (2-NFE, REPA teacher)

The above configs specify a training batch size of 512 images per GPU (requiring 32GB of VRAM per GPU), so 8 GPUs are required to reproduce the total batch size of 4096 in the paper. If you do not have enough VRAM, reduce the batch size (`samples_per_gpu`) in the config file accordingly, or enable gradient accumulation by adding `grad_accum_batch_size=<DESIRED_BATCH_SIZE>` to `train_cfg` in the config file.

### Evaluation (ADM’s FID, IS, Precision, Recall)

Run the following command to evaluate a pretrained model (downloaded automatically) using DDP on 1 node with 8 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py <PATH_TO_CONFIG> --launcher pytorch --diff_seed
```
where `<PATH_TO_CONFIG>` can be one of the following:
- `configs/piflow_imagenet/gmdit_k32_imagenet_piid_1step_test.py` (1-NFE, FM teacher)
- `configs/piflow_imagenet/gmrepa_k32_imagenet_piid_1step_test.py` (1-NFE, REPA teacher)
- `configs/piflow_imagenet/gmrepa_k32_imagenet_piid_2step_test.py` (2-NFE, REPA teacher)

To evaluate a custom checkpoint, add the `--ckpt <PATH_TO_CKPT>` argument:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py <PATH_TO_CONFIG> --ckpt <PATH_TO_CKPT> --launcher pytorch --diff_seed
```
