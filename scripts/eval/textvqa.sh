#!/usr/bin/env bash
# TextVQA validation.
source "$(dirname "$0")/_common.sh"
OUT="${EVAL_DIR}/textvqa/answers/${MODEL_NAME}.jsonl"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -m tinyllava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
  --image-folder "${EVAL_DIR}/textvqa/train_images" --answers-file "${OUT}" --temperature 0 --conv-mode "${CONV_MODE}"
python -m tinyllava.eval.eval_textvqa --annotation-file "${EVAL_DIR}/textvqa/TextVQA_0.5.1_val.json" --result-file "${OUT}"
