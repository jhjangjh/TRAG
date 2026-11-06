# Modified by the TRAG authors from TinyLLaVA Factory. See NOTICE.
"""Supervised fine-tuning (SFT) entry point: language-modeling loss only."""
import os

import torch
import transformers

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.model.configuration_compodistill import CompoDistillConfig
from tinyllava.model.modeling_compodistill import CompoDistillForConditionalGeneration
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
    if model_arguments.distill_mode == 'compodistill' and model_arguments.train_stage == 'dft':
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


def train():

    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    # model_args holds the from_pretrained() arguments.
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    # The CompoDistill baseline uses its own config/model with a TAF connector.
    if model_arguments.distill_mode != 'compodistill':
        model_config = TinyLlavaConfig()
        model_config.load_from_config(model_arguments)
        model = TinyLlavaForConditionalGeneration(model_config)
    else:
        model_config = CompoDistillConfig()
        model_config.load_from_config(model_arguments)
        model = CompoDistillForConditionalGeneration(model_config)
    # Previous-stage (DPT) checkpoint, or fresh LLM / vision tower / connector.
    if training_arguments.pretrained_model_path is not None:
        # CompoDistill SFT: reconstruct TAFConnector before loading checkpoint
        if model_arguments.distill_mode == "compodistill":
            connector_bin = os.path.join(
                training_arguments.pretrained_model_path, 'connector', 'pytorch_model.bin'
            )
            if os.path.exists(connector_bin):
                connector_state_dict = torch.load(connector_bin, map_location='cpu')
                is_taf = any('teacher_connector' in k or 'alignment_projector' in k
                             for k in connector_state_dict.keys())
                if is_taf:
                    print("[CompoDistill SFT] Reconstructing TAFConnector from checkpoint...")
                    model.load_taf_from_connector_state_dict(connector_state_dict)
                    print("[CompoDistill SFT] TAFConnector reconstructed successfully!")

        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

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
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)

    trainer.train(resume_from_checkpoint=training_arguments.resume_from_checkpoint)

    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
