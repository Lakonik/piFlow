import random
import gradio as gr


def create_prompt_opts(
        var_dict, create_negative_prompt=True, prompt='', negatove_prompt='', display_label=False):
    if display_label:
        kwargs = dict(show_label=True, container=True, elem_classes=['force-hide-container'])
    else:
        kwargs = dict(show_label=False, container=False)
    var_dict['prompt'] = gr.Textbox(
        prompt, label='Prompt', lines=2, placeholder='Prompt', interactive=True, **kwargs)
    if create_negative_prompt:
        var_dict['negative_prompt'] = gr.Textbox(
            negatove_prompt, label='Negative prompt', lines=2,
            placeholder='Negative prompt', interactive=True, **kwargs)


def create_generate_bar(var_dict, text='Generate', variant='primary', seed=-1):
    with gr.Row(equal_height=False, elem_classes=['generate-bar']):
        var_dict['run_btn'] = gr.Button(text, variant=variant, scale=2)
        var_dict['seed'] = gr.Number(
            label='Seed', value=seed, min_width=100, precision=0, minimum=-1, maximum=2 ** 31,
            elem_classes=['force-hide-container', 'seed-input'])
        var_dict['random_seed'] = gr.Button('\U0001f3b2\ufe0f', elem_classes=['tool'])
        var_dict['reuse_seed'] = gr.Button('\u267b\ufe0f', elem_classes=['tool'])
        with gr.Column(visible=False):
            var_dict['last_seed'] = gr.Number(value=seed, label='Last seed')
    var_dict['reuse_seed'].click(
        fn=lambda x: x,
        inputs=var_dict['last_seed'],
        outputs=var_dict['seed'],
        show_progress=False,
        api_name=False)
    var_dict['random_seed'].click(
        fn=lambda: -1,
        outputs=var_dict['seed'],
        show_progress=False,
        api_name=False)


def create_image_size_bar(var_dict, height=768, width=1360, hw_slider_step=16):
    with gr.Row(equal_height=True, variant='compact', elem_classes=['force-hide-container']):
        var_dict['width'] = gr.Slider(
            label='Width', minimum=64, maximum=2048, step=hw_slider_step, value=width,
            elem_classes=['force-hide-container'])
        var_dict['switch_hw'] = gr.Button('\U000021C6', elem_classes=['tool'])
        var_dict['height'] = gr.Slider(
            label='Height', minimum=64, maximum=2048, step=hw_slider_step, value=height,
            elem_classes=['force-hide-container'])
        var_dict['switch_hw'].click(
            fn=lambda w, h: (h, w),
            inputs=[var_dict['width'], var_dict['height']],
            outputs=[var_dict['width'], var_dict['height']],
            show_progress=False,
            api_name=False)


def create_base_opts(var_dict,
                     steps=24,
                     min_steps=4,
                     max_steps=50,
                     steps_slider_step=1,
                     guidance_scale=None,
                     temperature=None,
                     render=True):
    with gr.Column(variant='compact', elem_classes=['custom-spacing'], render=render) as base_opts:
        with gr.Row(variant='compact', elem_classes=['force-hide-container']):
            var_dict['steps'] = gr.Slider(
                min_steps, max_steps, value=steps, step=steps_slider_step, label='Sampling steps',
                elem_classes=['force-hide-container'])
        if guidance_scale is not None or temperature is not None:
            with gr.Row(variant='compact', elem_classes=['force-hide-container']):
                if guidance_scale is not None:
                    var_dict['guidance_scale'] = gr.Slider(
                        0.0, 30.0, value=guidance_scale, step=0.5, label='Guidance scale',
                        elem_classes=['force-hide-container'])
                if temperature is not None:
                    var_dict['temperature'] = gr.Slider(
                        0.0, 1.0, value=temperature, step=0.01, label='Temperature',
                        elem_classes=['force-hide-container'])
    return base_opts


def set_seed(seed):
    seed = random.randint(0, 2**31) if seed == -1 else seed
    return seed
