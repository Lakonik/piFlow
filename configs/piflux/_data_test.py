data = dict(
    test=dict(
        type='ImagePrompt',
        data_root='data/t2i_prompts_hpsv2/',
        cache_dir='preproc_flux',
        cache_datalist_path='data/t2i_prompts_hpsv2/preproc_flux.jsonl.gz',
        prompt_dataset_kwargs=dict(
            path='Lakonik/t2i-prompts-hpsv2',
            split='test'),
        latent_size=(16, 128, 128),
        test_mode=True,
    ),
    test2=dict(
        type='ImagePrompt',
        data_root='data/t2i_prompts_coco_10k/',
        cache_dir='preproc_flux',
        cache_datalist_path='data/t2i_prompts_coco_10k/preproc_flux.jsonl.gz',
        prompt_dataset_kwargs=dict(
            path='Lakonik/t2i-prompts-coco-10k',
            split='test'),
        latent_size=(16, 128, 128),
        test_mode=True,
    ),
)
