#!/usr/bin/env bash
# Compositional reasoning: SugarCrepe, BiVLC and Winoground (accuracy is printed by each script).
source "$(dirname "$0")/_common.sh"
export CUDA_VISIBLE_DEVICES="${FIRST_GPU}"
python -m tinyllava.eval.model_vqa_sugarcrepe --model-path "${MODEL_PATH}" \
  --data-dir "${EVAL_DIR}/SugarCrepe/sugar-crepe/data" --image-folder "${EVAL_DIR}/SugarCrepe/val2017" \
  --output-dir "${EVAL_DIR}/SugarCrepe/results/${MODEL_NAME}"
python -m tinyllava.eval.model_vqa_bivlc --model-path "${MODEL_PATH}" \
  --data-dir "${EVAL_DIR}/BiVLC/data" --output-dir "${EVAL_DIR}/BiVLC/results/${MODEL_NAME}"
python -m tinyllava.eval.model_vqa_winoground --model-path "${MODEL_PATH}" \
  --data-file "${EVAL_DIR}/Winoground/data/examples.jsonl" --image-folder "${EVAL_DIR}/Winoground/data/images" \
  --output-dir "${EVAL_DIR}/Winoground/results/${MODEL_NAME}"
