# Modified by the TRAG authors from TinyLLaVA Factory. See NOTICE.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_compodistill import CompoDistillConfig
from ..utils.constants import *

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None


class TAFConnector(nn.Module):
    """
    Teacher Adapter Fetch (TAF) Module for CompoDistill.
    Combines frozen teacher connector with lightweight trainable alignment projector.

    Architecture: Linear + GELU + Linear + GELU + Linear
    - teacher_connector (frozen): Linear + GELU + Linear
    - alignment_projector (trainable): GELU + Linear
    """
    def __init__(self, teacher_connector, teacher_hidden_dim, student_hidden_dim):
        super().__init__()

        # Frozen teacher connector (2 layers: linear+gelu+linear)
        self.teacher_connector = teacher_connector

        # Trainable alignment projector: GELU + Linear
        # teacher_dim -> student_dim
        self.alignment_projector = nn.Sequential(
            nn.GELU(),
            nn.Linear(teacher_hidden_dim, student_hidden_dim)
        )

        # Initialize alignment projector
        for module in self.alignment_projector.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def parameters(self, recurse=True):
        """Override parameters() to only return alignment_projector parameters.
        This prevents training_recipe from accidentally enabling gradients on teacher_connector.
        Note: state_dict() still returns all parameters (teacher_connector + alignment_projector)
        so that checkpoints save the full 3-layer connector."""
        return self.alignment_projector.parameters(recurse=recurse)

    def forward(self, x):
        # Pass through frozen teacher connector
        with torch.no_grad():
            x = self.teacher_connector(x)
        # Pass through trainable alignment projector
        x = self.alignment_projector(x)
        return x

    def train(self, mode=True):
        """Override train() to keep teacher_connector in eval mode"""
        super().train(mode)
        # Always keep teacher connector in eval mode
        if hasattr(self, 'teacher_connector'):
            self.teacher_connector.eval()
        return self



class CompoDistillPreTrainedModel(PreTrainedModel):
    config_class = CompoDistillConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class CompoDistillForConditionalGeneration(CompoDistillPreTrainedModel):
    def __init__(self, config: CompoDistillConfig):

        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)

        # Initialize with standard student connector
        # This will be replaced by TAFConnector when load_teacher_connector is called
        self.connector = ConnectorFactory(config.connector_type)(config)

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        if 'qwen1.5' in config.llm_model_name_or_path.lower():
            self.tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
        self.post_init()


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        vision_token_mask = None
        num_images_per_sample = []
        all_vision_spans = []

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_token_mask,
                num_images_per_sample,
                all_vision_spans
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # LLaVA-KD style return format:
        # (outputs, labels, image_rela, text_rela, image_text_rela, num_images_per_sample, all_vision_spans)
        # For now, we don't compute relation matrices here (computed in trainer if needed)
        image_rela = None
        text_rela = None
        image_text_rela = None

        return outputs, labels, image_rela, text_rela, image_text_rela, num_images_per_sample, all_vision_spans

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _,
                _vision_token_mask,
                _num_images_per_sample,
                _all_vision_spans,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)
        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)

        # connector is either standard connector or TAFConnector
        image_features = self.connector(image_features)

        return image_features



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, [], []


        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # Track vision spans per batch sample: list of list of (start, end) tuples (before padding)
        vision_spans_before_pad = []
        num_images_per_sample = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                vision_spans_before_pad.append([])
                num_images_per_sample.append(0)
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_vision_spans = []
            cur_pos = 0

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_pos += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    img_len = cur_image_features.shape[0]
                    cur_vision_spans.append((cur_pos, cur_pos + img_len))
                    cur_pos += img_len
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((img_len,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            vision_spans_before_pad.append(cur_vision_spans)
            num_images_per_sample.append(int(num_images))

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Adjust vision spans for padding offset (before attention_mask is potentially set to None)
        computed_all_vision_spans = []
        is_left_pad = getattr(self.config, 'tokenizer_padding_side', 'right') == "left"
        for i, spans in enumerate(vision_spans_before_pad):
            if is_left_pad:
                # offset = number of padding tokens at start of this sample
                offset = int((~attention_mask[i].bool()).sum().item())
            else:
                offset = 0
            adjusted_spans = [(s + offset, e + offset) for s, e in spans]
            computed_all_vision_spans.append(adjusted_spans)
        computed_num_images_per_sample = num_images_per_sample

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, computed_num_images_per_sample, computed_all_vision_spans




    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)

        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)


    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)


    def load_connector(self, **kwargs):
        """
        Load connector from checkpoint.
        Handles both split format (connector/pytorch_model.bin) and safetensors format.
        For TAFConnector checkpoints, loads both teacher_connector and alignment_projector.
        """
        import os

        # If connector is already TAFConnector (loaded from safetensors), skip
        if hasattr(self.connector, 'teacher_connector'):
            print("[CompoDistill] TAFConnector already loaded from safetensors - skipping load_connector")
            return

        pretrained_connector_path = get_value_from_kwargs(kwargs, 'pretrained_connector_path')

        # Check if this is a TAFConnector checkpoint (split format)
        if pretrained_connector_path and os.path.exists(pretrained_connector_path):
            connector_bin_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')
            if os.path.exists(connector_bin_path):
                connector_weights = torch.load(connector_bin_path, map_location='cpu')

                # Check if it's a TAFConnector checkpoint
                is_taf = any('teacher_connector' in k or 'alignment_projector' in k for k in connector_weights.keys())

                if is_taf:
                    print("[CompoDistill] Loading TAFConnector from split format checkpoint...")
                    # Load directly into model's connector (which should be a TAFConnector)
                    self.connector.load_state_dict(connector_weights, strict=False)
                    print("[CompoDistill] TAFConnector loaded successfully")
                    return

        # Standard connector loading for non-TAF connectors
        self.connector.load_model(**kwargs)

    def load_teacher_connector(self, teacher_connector_state_dict, teacher_config, teacher_hidden_dim, student_hidden_dim):
        """
        Load teacher connector for TAF module (Teacher Adapter Fetch).
        Replaces self.connector with TAFConnector(teacher_connector + alignment_projector).

        Args:
            teacher_connector_state_dict: State dict of teacher's connector
            teacher_config: Config object from teacher model (to create connector with correct dimensions)
            teacher_hidden_dim: Hidden dimension of teacher LLM
            student_hidden_dim: Hidden dimension of student LLM
        """
        # Create teacher connector instance with TEACHER's config (not student's!)
        from . import ConnectorFactory
        teacher_connector = ConnectorFactory(teacher_config.connector_type)(teacher_config)

        # Load teacher connector weights
        teacher_connector.load_state_dict(teacher_connector_state_dict, strict=False)

        # Freeze teacher connector completely - recursively freeze all modules and parameters
        teacher_connector.eval()
        for module in teacher_connector.modules():
            module.eval()
            module.requires_grad_(False)
            for param in module.parameters():
                param.requires_grad = False

        # Also freeze at top level
        teacher_connector.requires_grad_(False)
        for param in teacher_connector.parameters():
            param.requires_grad = False

        # Replace connector with TAFConnector
        self.connector = TAFConnector(
            teacher_connector=teacher_connector,
            teacher_hidden_dim=teacher_hidden_dim,
            student_hidden_dim=student_hidden_dim
        )

        # Double-check: ensure teacher connector stays frozen
        self.connector.teacher_connector.eval()
        for module in self.connector.teacher_connector.modules():
            module.eval()
            module.requires_grad_(False)
            for param in module.parameters():
                param.requires_grad = False

        # Ensure alignment projector is trainable
        self.connector.alignment_projector.train()
        for module in self.connector.alignment_projector.modules():
            if not isinstance(module, nn.GELU):  # GELU doesn't have parameters
                module.requires_grad_(True)
                for param in module.parameters():
                    param.requires_grad = True

        print(f"[CompoDistill TAF] Teacher connector loaded and frozen")
        print(f"[CompoDistill TAF] Alignment projector initialized: {teacher_hidden_dim} -> {student_hidden_dim}")
        print(f"[CompoDistill TAF] Connector replaced with TAFConnector")

        # Print trainable status
        teacher_trainable = sum(p.numel() for p in self.connector.teacher_connector.parameters() if p.requires_grad)
        alignment_trainable = sum(p.numel() for p in self.connector.alignment_projector.parameters() if p.requires_grad)
        print(f"[CompoDistill TAF] Teacher connector trainable params: {teacher_trainable}")
        print(f"[CompoDistill TAF] Alignment projector trainable params: {alignment_trainable}")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override load_state_dict to automatically reconstruct TAFConnector
        when the state dict contains 'connector.teacher_connector.*' keys.
        This ensures correct loading regardless of how the model is loaded
        (from_pretrained, training_recipe.load, etc.).
        """
        is_taf = any(
            k.startswith('connector.teacher_connector.') or k.startswith('connector.alignment_projector.')
            for k in state_dict.keys()
        )
        if is_taf and not hasattr(self.connector, 'teacher_connector'):
            # Extract connector sub-dict (strip 'connector.' prefix)
            connector_state_dict = {
                k[len('connector.'):]: v
                for k, v in state_dict.items()
                if k.startswith('connector.')
            }
            print("[CompoDistill] Detected TAF keys in state_dict - reconstructing TAFConnector...")
            self.load_taf_from_connector_state_dict(connector_state_dict)

        return super().load_state_dict(state_dict, strict=False)

    def load_taf_from_connector_state_dict(self, connector_state_dict):
        """
        Reconstruct TAFConnector from a saved connector state dict without needing the teacher model.
        The state dict must contain 'teacher_connector.*' and 'alignment_projector.*' keys.
        Used in SFT (train.py) to restore the TAFConnector saved by DFT.

        Infers teacher_hidden_dim and student_hidden_dim from weight shapes:
          - teacher_connector: last Linear output dim = teacher_hidden_dim
          - alignment_projector.1.weight shape = [student_hidden_dim, teacher_hidden_dim]
        """
        # Infer dims from weight shapes
        # teacher_connector is mlp2x_gelu: Linear(vision_dim, teacher_hidden_dim*2) + GELU + Linear(teacher_hidden_dim*2, teacher_hidden_dim)
        # Last linear in teacher_connector: key ending with the highest index '.weight'
        tc_keys = sorted([k for k in connector_state_dict if k.startswith('teacher_connector.')])
        ap_weight_key = 'alignment_projector.1.weight'

        if not tc_keys or ap_weight_key not in connector_state_dict:
            raise ValueError(
                f"[CompoDistill] load_taf_from_connector_state_dict: expected 'teacher_connector.*' "
                f"and '{ap_weight_key}' in state dict. Got: {list(connector_state_dict.keys())}"
            )

        # alignment_projector.1.weight: [student_hidden_dim, teacher_hidden_dim]
        ap_weight = connector_state_dict[ap_weight_key]
        student_hidden_dim = ap_weight.shape[0]
        teacher_hidden_dim = ap_weight.shape[1]

        # Rebuild teacher_connector to match saved key structure.
        # Keys are either:
        #   flat:   teacher_connector.0.weight  (nn.Sequential built from Linear layers)
        #   nested: teacher_connector._connector.0.weight  (MLPConnector wrapping nn.Sequential)
        tc_state = {k[len('teacher_connector.'):]: v for k, v in connector_state_dict.items()
                    if k.startswith('teacher_connector.')}

        # Collect weight keys only (bias optional), find all Linear layers
        weight_keys = [k for k in tc_state if k.endswith('.weight')]

        # Check if nested under '_connector' (MLPConnector style)
        if any(k.startswith('_connector.') for k in weight_keys):
            # nested: _connector.0.weight, _connector.2.weight, ...
            inner_keys = [k[len('_connector.'):] for k in weight_keys if k.startswith('_connector.')]
            linear_indices = sorted(set(int(k.split('.')[0]) for k in inner_keys))
            modules = []
            for i, idx in enumerate(linear_indices):
                w = tc_state.get(f'_connector.{idx}.weight')
                b = tc_state.get(f'_connector.{idx}.bias')
                linear = nn.Linear(w.shape[1], w.shape[0], bias=(b is not None))
                modules.append(linear)
                if i < len(linear_indices) - 1:
                    modules.append(nn.GELU())
            inner_seq = nn.Sequential(*modules)
            teacher_connector = nn.Module()
            teacher_connector._connector = inner_seq
            teacher_connector.forward = lambda x: teacher_connector._connector(x)
        else:
            # flat: 0.weight, 2.weight, ...
            linear_indices = sorted(set(int(k.split('.')[0]) for k in weight_keys))
            modules = []
            for i, idx in enumerate(linear_indices):
                w = tc_state.get(f'{idx}.weight')
                b = tc_state.get(f'{idx}.bias')
                linear = nn.Linear(w.shape[1], w.shape[0], bias=(b is not None))
                modules.append(linear)
                if i < len(linear_indices) - 1:
                    modules.append(nn.GELU())
            teacher_connector = nn.Sequential(*modules)

        # Build TAFConnector and load all weights
        self.connector = TAFConnector(
            teacher_connector=teacher_connector,
            teacher_hidden_dim=teacher_hidden_dim,
            student_hidden_dim=student_hidden_dim
        )
        self.connector.load_state_dict(connector_state_dict, strict=True)

        # Freeze teacher_connector
        self.connector.teacher_connector.eval()
        for param in self.connector.teacher_connector.parameters():
            param.requires_grad = False

        # Ensure alignment_projector is trainable
        for param in self.connector.alignment_projector.parameters():
            param.requires_grad = True

        print(f"[CompoDistill TAF] Reconstructed TAFConnector from state dict "
              f"(teacher_dim={teacher_hidden_dim}, student_dim={student_hidden_dim})")
