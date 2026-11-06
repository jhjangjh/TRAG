#!/usr/bin/env bash
# MMMU validation. Scoring uses main_eval_only.py from the official MMMU repository
# (https://github.com/MMMU-Benchmark/MMMU, mmmu/); set MMMU_EVAL_DIR to that directory.
source "$(dirname "$0")/_common.sh"
OUT="${EVAL_DIR}/MMMU/answers/${MODEL_NAME}"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -m tinyllava.eval.model_vqa_mmmu \
  --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/MMMU/anns_for_eval.json" \
  --image-folder "${EVAL_DIR}/MMMU/all_images" --answers-file "${OUT}.jsonl" --temperature 0 --conv-mode "${CONV_MODE}"
python scripts/eval/convert_answer_to_mmmu.py --answers-file "${OUT}.jsonl" --answers-output "${OUT}_output.json"
cd "${MMMU_EVAL_DIR:?Set MMMU_EVAL_DIR to the mmmu directory of the MMMU repository}" && python main_eval_only.py --output_path "${OUT}_output.json"
