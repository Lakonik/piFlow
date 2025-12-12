data = dict(
    test=dict(
        type='ImagePrompt',
        data_root='data/t2i_prompts_hpsv2/',
        cache_dir='preproc_qwen',
        cache_datalist_path='data/t2i_prompts_hpsv2/preproc_qwen.jsonl.gz',
        prompt_dataset_kwargs=dict(
            path='Lakonik/t2i-prompts-hpsv2',
            split='test'),
        negative_prompt_embeds_path='data/qwen_empty_prompt_embeds.pth',
        pad_seq_len=512,
        latent_size=(16, 128, 128),
        test_mode=True,
    ),
    test2=dict(
        type='ImagePrompt',
        data_root='data/t2i_prompts_coco_10k/',
        cache_dir='preproc_qwen',
        cache_datalist_path='data/t2i_prompts_coco_10k/preproc_qwen.jsonl.gz',
        prompt_dataset_kwargs=dict(
            path='Lakonik/t2i-prompts-coco-10k',
            split='test'),
        negative_prompt_embeds_path='data/qwen_empty_prompt_embeds.pth',
        pad_seq_len=512,
        latent_size=(16, 128, 128),
        test_mode=True,
    ),
)
