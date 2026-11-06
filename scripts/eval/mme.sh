#!/usr/bin/env bash
# MME. Needs LLaVA-1.5's EVAL_DIR/MME/{convert_answer_to_mme.py,eval_tool/}.
source "$(dirname "$0")/_common.sh"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -m tinyllava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/MME/llava_mme.jsonl" \
  --image-folder "${EVAL_DIR}/MME/MME_Benchmark_release_version" \
  --answers-file "${EVAL_DIR}/MME/answers/${MODEL_NAME}.jsonl" --temperature 0 --conv-mode "${CONV_MODE}"
cd "${EVAL_DIR}/MME" && python convert_answer_to_mme.py --experiment "${MODEL_NAME}"
cd eval_tool && python calculation.py --results_dir "answers/${MODEL_NAME}"
