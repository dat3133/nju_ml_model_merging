#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KNOTS_DIR="${REPO_ROOT}/external/knots"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/hf-cache}"
OUTPUT_CSV="${OUTPUT_CSV:-${REPO_ROOT}/results/tables/level2_knots_mnist.csv}"
CONFIG_NAME="${CONFIG_NAME:-vitB_r16_knots_ties_8merge_mnist_eval}"
HEAD_PATH="${HEAD_PATH:-ViT-B-32-CLIP/mnist_head.pt}"

cd "${KNOTS_DIR}"

export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -f "${HEAD_PATH}" ]]; then
  python tools/generate_mnist_clip_head.py \
    --cache-dir "${CACHE_DIR}" \
    --output "${HEAD_PATH}"
fi

python -m eval_scripts.pertask_eval_config \
  --config-name "${CONFIG_NAME}" \
  --eval-split test \
  --output-csv "${OUTPUT_CSV}"
