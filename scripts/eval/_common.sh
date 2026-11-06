# Shared setup for the evaluation scripts. Usage: bash scripts/eval/<benchmark>.sh MODEL_PATH [GPU_LIST]
# EVAL_DIR must point to the LLaVA-1.5 evaluation data layout (see README).
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 MODEL_PATH [GPU_LIST, e.g. 0,1,2,3]" >&2
  exit 2
fi
MODEL_PATH="$1"
MODEL_NAME="$(basename "${MODEL_PATH}")"
GPU_LIST="${2:-0}"
CONV_MODE="${CONV_MODE:-qwen2_base}"
EVAL_DIR="${EVAL_DIR:?Set EVAL_DIR to the evaluation data directory}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"
FIRST_GPU="${GPU_LIST%%,*}"
