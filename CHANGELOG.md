# Changelog

## [0.1.1] - 2025-12-18

### Changed
- **Breaking:** Rename the pretrained HuggingFace model argument `from_pretrained` to `model_name_or_path`. Please update any custom configurations or scripts accordingly.

### Fixed
- **Important:** Fix a GMFlow batching bug introduced by the recent numerical-stability change.
- Fix loading pi-Flow adapters for TorchAO-quantized base models.
- Fix a rare bug that could cause distributed runs to hang when loading pretrained HuggingFace models.

## [0.1.0] - 2025-12-12

### Added
- FLUX.2 integration and `PiFlux2Pipeline` with example demos.
- `Qwen3VLPromptRewriter` for prompt rewriting.
- Dataset and data-caching enhancements:
  - Support for `condition_images` and `condition_latents` for image-conditioned generation.
  - Improved image rescaling.
  - `ConcatDataset` for combining multiple datasets.
  - `--skip-existing` option in `cache_image_prompt_data.py` for resuming interrupted caching.
- Support for loading quantized base models in pi-Flow pipelines.
- Support for non-local storage backends in `save_inference_weights.py`.
- Support for loading sharded safetensors from the local filesystem ([#17](https://github.com/Lakonik/piFlow/issues/17)).

### Changed
- **Breaking:** Switch all pi-Flow model schedulers to [`FlowMapSDEScheduler`](lakonlab/models/diffusions/schedulers/flow_map_sde.py), which supports both deterministic (`h = 0`) and stochastic (`h > 0`) sampling.
- Reduce peak memory usage during initialization when distilling large models with LoRA.
- Bump Gradio to `5.49.0` for web demos.
- Improve GMFlow numerical stability.

### Fixed
- Reduce the risk of CUDA OOM when saving large checkpoints.
- Fix `S3Backend.list_dir_or_file`.
- Fix errors when loading pretrained HuggingFace models under unstable network conditions.
- Fix test data split configurations.
