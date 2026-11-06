#!/usr/bin/env bash
# GQA testdev-balanced. Needs EVAL_DIR/gqa/{llava_gqa_testdev_balanced.jsonl,data/images,eval.py}.
source "$(dirname "$0")/_common.sh"
SPLIT=llava_gqa_testdev_balanced
OUT="${EVAL_DIR}/gqa/answers/${SPLIT}/${MODEL_NAME}"
mkdir -p "${OUT}"
IFS=',' read -ra GPUS <<< "${GPU_LIST}"
for IDX in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$IDX]}" python -m tinyllava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" --question-file "${EVAL_DIR}/gqa/${SPLIT}.jsonl" \
    --image-folder "${EVAL_DIR}/gqa/data/images" --answers-file "${OUT}/${#GPUS[@]}_${IDX}.jsonl" \
    --num-chunks "${#GPUS[@]}" --chunk-idx "${IDX}" --temperature 0 --conv-mode "${CONV_MODE}" &
done
wait
cat "${OUT}"/${#GPUS[@]}_*.jsonl > "${OUT}/merge.jsonl"
python scripts/eval/convert_gqa_for_eval.py --src "${OUT}/merge.jsonl" --dst "${EVAL_DIR}/gqa/testdev_balanced_predictions.json"
cd "${EVAL_DIR}/gqa" && python eval.py --tier testdev_balanced
