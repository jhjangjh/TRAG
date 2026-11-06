# Modified by the TRAG authors from TinyLLaVA Factory. See NOTICE.
"""Distillation entry point for the DPT and DFT stages (teacher -> student)."""
import os

import torch
import transformers

from tinyllava.train.distill_trainer import DistillLLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.model.configuration_compodistill import CompoDistillConfig
from tinyllava.model.modeling_compodistill import CompoDistillForConditionalGeneration
from tinyllava.model.modeling_tinyllava_distill import TinyLlavaDistillForConditionalGeneration
from tinyllava.data.dataset import make_supervised_data_module


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments)
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    if model_arguments.distill_mode == 'compodistill' and model_arguments.train_stage != 'sft':
        llm_args['attn_implementation'] = 'eager'
    else:
        llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def get_connector_weights(weights, keyword):
    """Strip ``<keyword>.`` from the matching keys of a connector state dict."""
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

def load_connector_state_dict(checkpoint_dir):
    """Connector weights from a split (connector/pytorch_model.bin) or unified
    (model.safetensors, possibly sharded) checkpoint."""
    connector_bin = os.path.join(checkpoint_dir, 'connector', 'pytorch_model.bin')
    if os.path.exists(connector_bin):
        return torch.load(connector_bin, map_location='cpu')
    from safetensors.torch import load_file
    single = os.path.join(checkpoint_dir, 'model.safetensors')
    index = os.path.join(checkpoint_dir, 'model.safetensors.index.json')
    if os.path.exists(single):
        files = [single]
    elif os.path.exists(index):
        import json
        with open(index) as f:
            files = [os.path.join(checkpoint_dir, s) for s in sorted(set(json.load(f)['weight_map'].values()))]
    else:
        raise FileNotFoundError(f"No connector weights found in {checkpoint_dir}")
    state = {}
    for file in files:
        state.update({k[len('connector.'):]: v for k, v in load_file(file).items() if k.startswith('connector.')})
    return state


def get_training_torch_dtype(training_arguments):
    if getattr(training_arguments, 'bf16', False):
        return torch.bfloat16
    if getattr(training_arguments, 'fp16', False):
        return torch.float16
    return torch.float32

def train():

    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)

    if model_arguments.teacher_model_path is None:
        raise ValueError("--teacher_model_path is required for distillation training.")

    # The language models are built from their configs here, so the attention
    # implementation is set explicitly. CompoDistill needs eager attention maps.
    teacher_attn = 'eager' if model_arguments.distill_mode == 'compodistill' else model_arguments.attn_implementation

    # Teacher checkpoints are either TinyLLaVA "split" directories
    # (language_model/, vision_tower/, connector/) or a single safetensors model.
    teacher_language_model_path = os.path.join(model_arguments.teacher_model_path, 'language_model')
    is_split_format = os.path.exists(teacher_language_model_path)

    if is_split_format:
        pre_vision_tower_path = os.path.join(model_arguments.teacher_model_path, 'vision_tower/pytorch_model.bin')
        pre_vision = torch.load(pre_vision_tower_path, map_location='cpu')
        pre_connector_path = os.path.join(model_arguments.teacher_model_path, 'connector/pytorch_model.bin')
        pre_connector = torch.load(pre_connector_path, map_location='cpu')

        teacher_model_config = TinyLlavaConfig.from_pretrained(model_arguments.teacher_model_path)
        teacher_model_config.text_config._attn_implementation = teacher_attn
        teacher_model = TinyLlavaDistillForConditionalGeneration(teacher_model_config)

        # Load LLM
        teacher_model.language_model = teacher_model.language_model.from_pretrained(
            os.path.join(model_arguments.teacher_model_path, 'language_model'),
            attn_implementation=teacher_attn,
            torch_dtype=get_training_torch_dtype(training_arguments),
        )
        print(f"Teacher LLM loaded from {os.path.join(model_arguments.teacher_model_path, 'language_model')}")

        # Load vision tower
        teacher_model.vision_tower._vision_tower.load_state_dict(pre_vision)
        print(f"Teacher Vision Tower loaded from {pre_vision_tower_path}")

        # Load connector
        teacher_model.connector._connector.load_state_dict(get_connector_weights(pre_connector, '_connector'))
        print(f"Teacher Connector loaded from {pre_connector_path}")
    else:
        # The config is also needed for CompoDistill's alignment projector.
        teacher_model_config = TinyLlavaConfig.from_pretrained(model_arguments.teacher_model_path)
        teacher_model_config.text_config._attn_implementation = teacher_attn

        teacher_model = TinyLlavaDistillForConditionalGeneration.from_pretrained(
            model_arguments.teacher_model_path,
            config=teacher_model_config,
            torch_dtype=get_training_torch_dtype(training_arguments),
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"Teacher model loaded from {model_arguments.teacher_model_path}")

    teacher_model.eval()
    teacher_model.requires_grad_(False)
    # Same truncation as the student, so teacher and student sequences align;
    # no KV cache (also required by FlashAttention-2 with right padding).
    teacher_model.config.tokenizer_model_max_length = model_arguments.model_max_length
    teacher_model.config.use_cache = False

    # Student model. model_args holds the from_pretrained() arguments.
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    if model_arguments.distill_mode != "compodistill":
        model_config = TinyLlavaConfig()
        model_config.load_from_config(model_arguments)
        # Used when DFT fills the LLM from a unified checkpoint (no from_pretrained).
        model_config.text_config._attn_implementation = model_args['llm']['attn_implementation']

        model = TinyLlavaDistillForConditionalGeneration(model_config)
    else:
        model_config = CompoDistillConfig()
        model_config.load_from_config(model_arguments)

        model = CompoDistillForConditionalGeneration(model_config)

    # Previous-stage checkpoint, or fresh LLM / vision tower / connector.
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
        print(f"Student model loaded from pretrained: {training_arguments.pretrained_model_path}")
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])
        print("Student model initialized from scratch")

    # CompoDistill baseline: Teacher Adapter Fetch (TAF) connector.
    if model_arguments.distill_mode == "compodistill":
        if model_arguments.train_stage == "dpt":
            # DPT: build TAFConnector fresh from teacher model
            print("[CompoDistill DPT] Creating TAFConnector from teacher model...")

            if is_split_format:
                teacher_connector_state_dict = get_connector_weights(pre_connector, '_connector')
            else:
                teacher_connector_state_dict = {}
                for k, v in teacher_model.connector.state_dict().items():
                    if k.startswith('_connector.'):
                        teacher_connector_state_dict[k.replace('_connector.', '')] = v
                    else:
                        teacher_connector_state_dict[k] = v

            teacher_hidden_dim = teacher_model_config.hidden_size
            student_hidden_dim = model.config.hidden_size
            print(f"Teacher hidden dim: {teacher_hidden_dim}, Student hidden dim: {student_hidden_dim}")

            model.load_teacher_connector(
                teacher_connector_state_dict=teacher_connector_state_dict,
                teacher_config=teacher_model_config,
                teacher_hidden_dim=teacher_hidden_dim,
                student_hidden_dim=student_hidden_dim
            )
            print("[CompoDistill DPT] TAFConnector created successfully!")

        else:
            # DFT: restore the TAFConnector from the previous (SFT) checkpoint.
            print("[CompoDistill DFT] Restoring TAFConnector from previous stage checkpoint...")
            connector_state_dict = load_connector_state_dict(training_arguments.pretrained_model_path)
            is_taf = any('teacher_connector' in k or 'alignment_projector' in k
                         for k in connector_state_dict.keys())
            if not is_taf:
                raise ValueError(
                    f"[CompoDistill] Expected TAF keys in connector checkpoint, got: "
                    f"{list(connector_state_dict.keys())}"
                )

            model.load_taf_from_connector_state_dict(connector_state_dict)
            print("[CompoDistill DFT] TAFConnector restored successfully!")

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio

    # training_recipe changes requires_grad; re-freeze CompoDistill's teacher connector.
    if model_arguments.distill_mode == "compodistill" and hasattr(model.connector, 'teacher_connector'):
        print("[CompoDistill] Re-freezing teacher connector after training_recipe...")
        model.connector.teacher_connector.eval()
        for module in model.connector.teacher_connector.modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    log_trainable_params(model)  # Counts are inaccurate under ZeRO-3.

    trainer = DistillLLaVATrainer(
        teacher_model=teacher_model,
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        distill_mode=model_arguments.distill_mode,
        train_stage=model_arguments.train_stage,
        **data_module
    )

    trainer.train(resume_from_checkpoint=training_arguments.resume_from_checkpoint)

    training_recipe.save(model, trainer)

if __name__ == "__main__":
    transformers.set_seed(1234)
    train()
