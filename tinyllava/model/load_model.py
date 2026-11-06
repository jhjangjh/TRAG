# Modified by the TRAG authors from TinyLLaVA Factory. See NOTICE.
import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from .modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig
from .modeling_compodistill import CompoDistillForConditionalGeneration
from .configuration_compodistill import CompoDistillConfig

def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp


def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map=None,
                          device="cuda", torch_dtype=None, **kwargs):
    """Load a TinyLLaVA-format checkpoint for evaluation or serving.

    By default the weights are loaded in full precision on the CPU and the
    caller moves the model to its device; the reported evaluation used this
    setting. Pass ``torch_dtype`` (e.g. ``torch.float16``) to load in lower
    precision, or ``load_8bit`` / ``load_4bit`` for bitsandbytes quantization,
    which places the model on ``device`` directly.
    """
    if load_8bit or load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=load_8bit,
            load_in_4bit=load_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            # Quantize the LLM only; SigLIP's attention-pooling head is not bnb-compatible.
            llm_int8_skip_modules=['vision_tower', 'connector', 'lm_head'],
        )
        device_map = device_map or {"": device}
        torch_dtype = torch_dtype or torch.float16
    if torch_dtype is not None:
        kwargs['torch_dtype'] = torch_dtype
    if device_map is not None:
        kwargs['device_map'] = device_map

    # Check if it's a CompoDistill checkpoint by looking at architectures in config
    if model_name_or_path is not None and 'lora' not in model_name_or_path:
        config_path = os.path.join(model_name_or_path, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            architectures = config_dict.get('architectures', [])

            if 'CompoDistillForConditionalGeneration' in architectures:
                print(f"Loading CompoDistill model from {model_name_or_path}")
                # Load model - connector might not match if it's TAFConnector
                model = CompoDistillForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True,  # Ignore connector size mismatch
                    **kwargs,
                )

                # Check if checkpoint has TAFConnector and manually load it
                safetensors_path = os.path.join(model_name_or_path, 'model.safetensors')
                if os.path.exists(safetensors_path):
                    from safetensors.torch import load_file
                    weights = load_file(safetensors_path)

                    # Check if it's TAFConnector checkpoint
                    is_taf = any('teacher_connector' in k or 'alignment_projector' in k for k in weights.keys())

                    if is_taf:
                        print("[CompoDistill] Detected TAFConnector checkpoint, creating TAFConnector...")
                        # Create TAFConnector from loaded weights
                        from .modeling_compodistill import TAFConnector

                        # Infer teacher_hidden_dim from alignment_projector weight shape
                        # alignment_projector.1.weight has shape [student_hidden_dim, teacher_hidden_dim]
                        alignment_weight = weights['connector.alignment_projector.1.weight']
                        student_hidden_dim = alignment_weight.shape[0]  # Output dim
                        teacher_hidden_dim = alignment_weight.shape[1]  # Input dim

                        print(f"[CompoDistill] Inferred dimensions: teacher={teacher_hidden_dim}, student={student_hidden_dim}")

                        # Get vision hidden size from teacher_connector first layer
                        # teacher_connector.0.weight has shape [teacher_hidden_dim, vision_hidden_size]
                        first_layer_weight = weights['connector.teacher_connector._connector.0.weight']
                        vision_hidden_size = first_layer_weight.shape[1]

                        print(f"[CompoDistill] Vision hidden size: {vision_hidden_size}")

                        # Create teacher connector with correct dimensions
                        from . import ConnectorFactory

                        # Create a minimal config with correct dimensions for teacher connector
                        from .configuration_compodistill import CompoDistillConfig
                        # Load the base config first
                        temp_config = CompoDistillConfig.from_pretrained(model_name_or_path)
                        # Override with teacher dimensions
                        temp_config.hidden_size = teacher_hidden_dim
                        temp_config.vision_hidden_size = vision_hidden_size

                        teacher_connector = ConnectorFactory(config_dict['connector_type'])(temp_config)

                        # Create TAFConnector
                        taf_connector = TAFConnector(teacher_connector, teacher_hidden_dim, student_hidden_dim)

                        # Load weights into TAFConnector
                        # Remove 'connector.' prefix from keys
                        connector_weights = {}
                        for k, v in weights.items():
                            if k.startswith('connector.'):
                                # Remove 'connector.' prefix (first 10 characters)
                                new_key = k[10:]  # 'connector.' is 10 characters
                                connector_weights[new_key] = v

                        taf_connector.load_state_dict(connector_weights, strict=True)

                        # Replace model's connector with TAFConnector
                        model.connector = taf_connector
                        print("[CompoDistill] TAFConnector loaded successfully for inference")
            else:
                model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, **kwargs)
        else:
            model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, **kwargs)

    elif model_name_or_path is not None and 'lora' in model_name_or_path:
        if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
            model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
            model = TinyLlavaForConditionalGeneration(model_config)
            language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
            language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
            model.language_model.load_state_dict(language_model_ckp)
            vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
            vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
            model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
            connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
            connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
            model.connector.load_state_dict(connector_ckp, strict=False)
            model.to(torch.float16)
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_name_or_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    tokenizer = model.tokenizer
    return model, tokenizer, image_processor, context_len
