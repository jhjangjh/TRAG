#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/paths.env}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}. Copy configs/paths.example.env first." >&2
  exit 2
fi

set -a
source "${CONFIG_PATH}"
set +a

required=(DATA_ROOT PRETRAIN_JSON INSTRUCT_JSON STUDENT_MODEL TEACHER_MODEL VISION_TOWER OUTPUT_ROOT)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required setting: ${variable}" >&2
    exit 2
  fi
done

CONV_VERSION="${CONV_VERSION:-qwen2_base}"
RUN_NAME="${RUN_NAME:-trag-run}"
GPUS="${GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29501}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DPT_BATCH_SIZE="${DPT_BATCH_SIZE:-4}"
DPT_GRAD_ACCUM="${DPT_GRAD_ACCUM:-32}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
DFT_BATCH_SIZE="${DFT_BATCH_SIZE:-2}"
DFT_GRAD_ACCUM="${DFT_GRAD_ACCUM:-32}"

DPT_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}/dpt"
SFT_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}/sft"
DFT_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}/dft"

common_args=(
  --deepspeed configs/zero2.json
  --is_multimodal True
  --conv_version "${CONV_VERSION}"
  --model_name_or_path "${STUDENT_MODEL}"
  --vision_tower "${VISION_TOWER}"
  --vision_tower2 ""
  --connector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --image_aspect_ratio square
  --attn_implementation flash_attention_2
  --bf16 True
  --training_recipe common
  --tune_type_vision_tower frozen
  --tune_vision_tower_from_layer 0
  --tune_type_connector full
  --group_by_modality_length True
  --num_train_epochs 1
  --evaluation_strategy no
  --save_strategy steps
  --save_steps 1000
  --save_total_limit 2
  --weight_decay 0.0
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --adam_beta1 0.9
  --adam_beta2 0.99
  --logging_steps 1
  --tf32 False
  --model_max_length "${MODEL_MAX_LENGTH}"
  --gradient_checkpointing True
  --dataloader_num_workers 4
  --lazy_preprocess True
  --report_to tensorboard
  --tokenizer_use_fast False
  --seed 42
  --data_seed 42
)

mkdir -p "${DPT_OUTPUT}" "${SFT_OUTPUT}" "${DFT_OUTPUT}"

deepspeed --include "localhost:${GPUS}" --master_port "${MASTER_PORT}" \
  tinyllava/train/distill_train.py \
  "${common_args[@]}" \
  --data_path "${PRETRAIN_JSON}" \
  --image_folder "${DATA_ROOT}/LLaVA-Pretrain/images" \
  --teacher_model_path "${TEACHER_MODEL}" \
  --distill_mode trag \
  --train_stage dpt \
  --tune_type_llm frozen \
  --output_dir "${DPT_OUTPUT}" \
  --per_device_train_batch_size "${DPT_BATCH_SIZE}" \
  --gradient_accumulation_steps "${DPT_GRAD_ACCUM}" \
  --learning_rate 1e-3 \
  --run_name "${RUN_NAME}-dpt"

deepspeed --include "localhost:${GPUS}" --master_port "$((MASTER_PORT + 1))" \
  tinyllava/train/train.py \
  "${common_args[@]}" \
  --data_path "${INSTRUCT_JSON}" \
  --image_folder "${DATA_ROOT}/LLaVA-Finetune" \
  --pretrained_model_path "${DPT_OUTPUT}" \
  --distill_mode trag \
  --train_stage sft \
  --tune_type_llm full \
  --output_dir "${SFT_OUTPUT}" \
  --per_device_train_batch_size "${SFT_BATCH_SIZE}" \
  --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
  --learning_rate 2e-5 \
  --run_name "${RUN_NAME}-sft"

deepspeed --include "localhost:${GPUS}" --master_port "$((MASTER_PORT + 2))" \
  tinyllava/train/distill_train.py \
  "${common_args[@]}" \
  --data_path "${INSTRUCT_JSON}" \
  --image_folder "${DATA_ROOT}/LLaVA-Finetune" \
  --pretrained_model_path "${SFT_OUTPUT}" \
  --teacher_model_path "${TEACHER_MODEL}" \
  --distill_mode trag \
  --train_stage dft \
  --tune_type_llm full \
  --output_dir "${DFT_OUTPUT}" \
  --per_device_train_batch_size "${DFT_BATCH_SIZE}" \
  --gradient_accumulation_steps "${DFT_GRAD_ACCUM}" \
  --learning_rate 2e-5 \
  --run_name "${RUN_NAME}-dft"

echo "TRAG training complete: ${DFT_OUTPUT}"
