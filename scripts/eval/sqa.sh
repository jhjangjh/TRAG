#!/usr/bin/env bash
# ScienceQA (the paper reports IMG-Accuracy).
source "$(dirname "$0")/_common.sh"
OUT="${EVAL_DIR}/scienceqa/answers/${MODEL_NAME}"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -m tinyllava.eval.model_vqa_science \
  --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/scienceqa/llava_test_CQM-A.json" \
  --image-folder "${EVAL_DIR}/scienceqa/images/test" --answers-file "${OUT}.jsonl" \
  --single-pred-prompt --temperature 0 --conv-mode "${CONV_MODE}"
python -m tinyllava.eval.eval_science_qa --base-dir "${EVAL_DIR}/scienceqa" --result-file "${OUT}.jsonl" \
  --output-file "${OUT}_output.jsonl" --output-result "${OUT}_result.json"
