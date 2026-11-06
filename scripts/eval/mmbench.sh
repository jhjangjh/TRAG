#!/usr/bin/env bash
# MMBench / MMBench-CN dev. Set LANG_SPLIT=cn for MMBench-CN. Upload the .xlsx to the official server.
source "$(dirname "$0")/_common.sh"
if [[ "${LANG_SPLIT:-en}" == cn ]]; then
  DIR=mmbench_cn; SPLIT=MMBench_DEV_CN_legacy; LANG_ARGS=(--lang cn)
else
  DIR=mmbench; SPLIT=MMBench_DEV_EN_legacy; LANG_ARGS=()
fi
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -m tinyllava.eval.model_vqa_mmbench \
  --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/${DIR}/${SPLIT}.tsv" \
  --answers-file "${EVAL_DIR}/${DIR}/answers/${SPLIT}/${MODEL_NAME}.jsonl" \
  "${LANG_ARGS[@]}" --single-pred-prompt --temperature 0 --conv-mode "${CONV_MODE}"
mkdir -p "${EVAL_DIR}/${DIR}/answers_upload/${SPLIT}"
python scripts/eval/convert_mmbench_for_submission.py --annotation-file "${EVAL_DIR}/${DIR}/${SPLIT}.tsv" \
  --result-dir "${EVAL_DIR}/${DIR}/answers/${SPLIT}" --upload-dir "${EVAL_DIR}/${DIR}/answers_upload/${SPLIT}" \
  --experiment "${MODEL_NAME}"
