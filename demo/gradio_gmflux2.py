import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from mmgen.apis import set_random_seed
from lakonlab.models.diffusions.schedulers.flow_map_sde import FlowMapSDEScheduler
from lakonlab.ui.gradio.create_img_edit import create_interface_img_edit
from lakonlab.pipelines.pipeline_piflux2 import PiFlux2Pipeline
from lakonlab.pipelines.prompt_rewriters.qwen3_vl import Qwen3VLPromptRewriter


def parse_args():
    parser = argparse.ArgumentParser(description='pi-FLUX.2 Gradio Demo')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    return parser.parse_args()


DEFAULT_PROMPT = (
    "Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, "
    "full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire. "
    "Slight upward perspective angle looking up toward the iconic sphere, perfectly centered on clean white "
    "background. Left side labels with thin horizontal connector lines: the text '368m' in extra large bold dark grey "
    "numerals (#2D3748) positioned at exactly the antenna tip with 'TOTAL HEIGHT' in small caps below. The text '207m' "
    "in extra large bold with 'TELECAFÉ' in small caps below, with connector line touching the sphere precisely at the "
    "window level. Right side label with horizontal connector line touching the sphere's equator: the text '32m' in "
    "extra large bold dark grey numerals with 'SPHERE DIAMETER' in small caps below. Bottom section arranged in three "
    "balanced columns: Left - Large text '986' in extra bold dark grey with 'STEPS' in caps below. Center - 'BERLIN TV "
    "TOWER' in bold caps with 'FERNSEHTURM' in lighter weight below. Right - 'INAUGURATED' in bold caps with 'OCTOBER "
    "3, 1969' below. All typography in modern sans-serif font (such as Inter or Helvetica), color #2D3748, clean "
    "minimal technical diagram style. Horizontal connector lines are thin, precise, and clearly visible, touching the "
    "tower structure at exact corresponding measurement points. Professional architectural elevation drawing aesthetic "
    "with dynamic low angle perspective creating sense of height and grandeur, poster-ready infographic design with "
    "perfect visual hierarchy.")

SYSTEM_PROMPT_TEXT_ONLY_PATH = 'lakonlab/pipelines/prompt_rewriters/system_prompts/default_text_only.txt'
SYSTEM_PROMPT_WITH_IMAGES_PATH = 'lakonlab/pipelines/prompt_rewriters/system_prompts/default_with_images.txt'


def main():
    args = parse_args()

    pipe = PiFlux2Pipeline.from_pretrained(
        'diffusers/FLUX.2-dev-bnb-4bit',
        torch_dtype=torch.bfloat16)
    pipe.load_piflow_adapter(
        'Lakonik/pi-FLUX.2',
        subfolder='gmflux2_k8_piid_4step',
        target_module_name='transformer')
    pipe.scheduler = FlowMapSDEScheduler.from_config(  # use fixed shift=3.2
        pipe.scheduler.config, shift=3.2, use_dynamic_shifting=False, final_step_size_scale=0.5)
    pipe = pipe.to('cuda')

    prompt_rewriter = Qwen3VLPromptRewriter(
        device_map="cuda",
        system_prompt_text_only=open(SYSTEM_PROMPT_TEXT_ONLY_PATH, 'r').read(),
        system_prompt_wigh_images=open(SYSTEM_PROMPT_WITH_IMAGES_PATH, 'r').read(),
        max_new_tokens_default=512,
    )

    def run_rewrite_prompt(seed, prompt, rewrite_prompt, in_image, progress=gr.Progress(track_tqdm=True)):
        image_list = None
        if in_image is not None and len(in_image) > 0:
            image_list = []
            for item in in_image:
                image_list.append(item[0])
        if rewrite_prompt:
            set_random_seed(seed)
            progress(0.05, desc="Rewriting prompt...")
            if image_list is None:
                final_prompt = prompt_rewriter.rewrite_text_batch(
                    [prompt])[0]
            else:
                final_prompt = prompt_rewriter.rewrite_edit_batch(
                    [image_list], [prompt])[0]
            return final_prompt, None
        else:
            return '', None

    def generate(
            seed, prompt, rewrite_prompt, rewritten_prompt, in_image, width, height, steps,
            progress=gr.Progress(track_tqdm=True)):
        image_list = None
        if in_image is not None and len(in_image) > 0:
            image_list = []
            for item in in_image:
                image_list.append(item[0])
        return pipe(
            image=image_list,
            prompt=rewritten_prompt if rewrite_prompt else prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

    with gr.Blocks(analytics_enabled=False,
                   title='pi-FLUX.2 Demo',
                   css_paths='lakonlab/ui/gradio/style.css'
                   ) as demo:

        md_txt = '# pi-FLUX.2 Demo\n\n' \
                 'Official demo of the paper [pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974). ' \
                 '**Base model:** [FLUX.2 dev](https://huggingface.co/black-forest-labs/FLUX.2-dev). **Fast policy:** GMFlow. **Code:** [https://github.com/Lakonik/piFlow](https://github.com/Lakonik/piFlow).\n' \
                 '<br> Use and distribution of this app are governed by the [FLUX [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt).'
        gr.Markdown(md_txt)

        create_interface_img_edit(
            generate,
            prompt=DEFAULT_PROMPT,
            steps=4, guidance_scale=None,
            args=['last_seed', 'prompt', 'rewrite_prompt', 'rewritten_prompt', 'in_image', 'width', 'height', 'steps'],
            rewrite_prompt_api=run_rewrite_prompt,
            rewrite_prompt_args=['last_seed', 'prompt', 'rewrite_prompt', 'in_image'],
            height=1024,
            width=1024
        )
        demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
