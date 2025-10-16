# Modified from https://github.com/linzhiqiu/t2v_metrics
# Copyright 2023 Zhiqiu Lin

import os
import re
import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import (
    AutoConfig, AutoModelForSeq2SeqLM, T5Config, T5ForConditionalGeneration,
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig)
from transformers.modeling_outputs import Seq2SeqLMOutput

IMAGE_TOKEN_INDEX = -200
CONTEXT_LEN = 2048
SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"


def t5_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    # Since there's no bos_token_id, simply concatenate the tokenized prompt_chunks with the image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def format_question(question, conversation_style='plain'):
    if conversation_style == 't5_plain':  # for 1st stage t5 model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 't5_chat':  # for 2nd stage t5 model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system':  # for 2nd stage t5 model
        question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system_no_user':  # for 2nd stage t5 model
        question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
    # elif conversation_style == 't5_chat_ood_system': # for 2nd stage t5 model
    #     question = SYSTEM_MSG + " HUMAN: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " GPT: "
    else:
        raise NotImplementedError()
    return question


def format_answer(answer, conversation_style='plain'):
    return answer


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


@dataclass
class ModelArguments:
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
    mm_vision_select_layer: Optional[int] = field(default=-2)  # default to the second last layer in llava1.5
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_vision_select_feature: Optional[str] = field(default="patch")


class CLIPT5Config(T5Config):
    model_type = "clip_t5"


class CLIPT5ForConditionalGeneration(T5ForConditionalGeneration):
    # This class supports both T5 and FlanT5
    config_class = CLIPT5Config

    def __init__(self, config):
        super(CLIPT5ForConditionalGeneration, self).__init__(config)
        self.embed_tokens = self.encoder.embed_tokens
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_model(self):
        return self  # for compatibility with LlavaMetaForCausalLM

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, decoder_attention_mask, past_key_values, labels, images
    ):
        # The labels are now separated from the input_ids.
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            raise NotImplementedError()

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        cur_image_idx = 0
        for _, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                raise NotImplementedError()
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.embed_tokens(cur_input_ids))
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            _input_embeds_lengths = []
            for cur_new_embed in new_input_embeds:
                _input_embeds_lengths.append(cur_new_embed.shape[0])
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, _input_embeds_length in zip(attention_mask, _input_embeds_lengths):
                    new_attn_mask_pad_left = torch.full((_input_embeds_length - input_ids.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((new_input_embeds.shape[1] - _input_embeds_length,), False,
                                                         dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, decoder_attention_mask, past_key_values, new_input_embeds, labels

    def encode_images(self, images):
        image_features = self.get_vision_tower()(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            if not vision_tower.is_loaded:
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'mlp2x_gelu')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            _, attention_mask, decoder_attention_mask, past_key_values, inputs_embeds, labels = \
                self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, decoder_attention_mask,
                                                          past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super(CLIPT5ForConditionalGeneration, self).forward(
            input_ids=None,  # will be None if inputs_embeds is not None
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        assert images is not None, "images must be provided"
        assert inputs is not None, "inputs must be provided"
        assert attention_mask is not None, "attention_mask must be provided"
        _, attention_mask, _, _, inputs_embeds, _ = \
            self.prepare_inputs_labels_for_multimodal(inputs, attention_mask, None, None, None, images)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super(CLIPT5ForConditionalGeneration, self).generate(
            input_ids=None,  # will be None if inputs_embeds is not None
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            inputs_embeds=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        })
        return model_inputs


AutoConfig.register("clip_t5", CLIPT5Config)
AutoModelForSeq2SeqLM.register(CLIPT5Config, CLIPT5ForConditionalGeneration)
