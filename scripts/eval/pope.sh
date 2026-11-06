#!/usr/bin/env bash
# POPE (the paper reports F1).
source "$(dirname "$0")/_common.sh"
OUT="${EVAL_DIR}/pope/answers/${MODEL_NAME}.jsonl"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -m tinyllava.eval.model_vqa_pope \
  --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/pope/llava_pope_test.jsonl" \
  --image-folder "${EVAL_DIR}/pope/val2014" --answers-file "${OUT}" --temperature 0 --conv-mode "${CONV_MODE}"
python -m tinyllava.eval.eval_pope --annotation-dir "${EVAL_DIR}/pope/coco" \
  --question-file "${EVAL_DIR}/pope/llava_pope_test.jsonl" --result-file "${OUT}"
