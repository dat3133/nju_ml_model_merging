#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
CACHE_DIR=${CACHE_DIR:-/root/autodl-tmp/hf-cache}
MED_ADAPTER=${MED_ADAPTER:-adapters/medical_lora_full}
LEGAL_ADAPTER=${LEGAL_ADAPTER:-adapters/legal_lora_full}
MERGED_DIR=${MERGED_DIR:-merged}
MODEL_PREFIX=${MODEL_PREFIX:-full_}
MERGED_MODEL_LIST=${MERGED_MODEL_LIST:-${MERGED_DIR}/full_level1_sweep_models.txt}

require_adapter() {
  local adapter=$1
  if [[ ! -f "${adapter}/adapter_config.json" || ! -f "${adapter}/adapter_model.safetensors" ]]; then
    echo "Missing PEFT adapter files in ${adapter}" >&2
    echo "Set MED_ADAPTER and LEGAL_ADAPTER to concrete adapter directories." >&2
    exit 1
  fi
}

register_model() {
  local model_dir=$1
  printf '%s\n' "${model_dir}" >> "${MERGED_MODEL_LIST}"
}

require_adapter "${MED_ADAPTER}"
require_adapter "${LEGAL_ADAPTER}"

mkdir -p "${MERGED_DIR}"
mkdir -p "$(dirname "${MERGED_MODEL_LIST}")"
: > "${MERGED_MODEL_LIST}"

for pair in "0.5 0.5" "0.7 0.7" "1.0 1.0" "1.2 1.2"; do
  read -r med_lambda legal_lambda <<< "${pair}"
  tag="m${med_lambda//./p}_l${legal_lambda//./p}"
  output_dir="${MERGED_DIR}/${MODEL_PREFIX}task_arithmetic_${tag}"
  python src/merge_task_arithmetic.py \
    --base-model "${BASE_MODEL}" \
    --adapters "${MED_ADAPTER}" "${LEGAL_ADAPTER}" \
    --lambdas "${med_lambda}" "${legal_lambda}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${output_dir}"
  register_model "${output_dir}"
done

for density in 0.2 0.4 0.6 0.8; do
  tag="d${density//./p}"
  output_dir="${MERGED_DIR}/${MODEL_PREFIX}ties_${tag}"
  python src/merge_adapters.py \
    --base-model "${BASE_MODEL}" \
    --adapters "${MED_ADAPTER}" "${LEGAL_ADAPTER}" \
    --adapter-names medical legal \
    --combination ties \
    --density "${density}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${output_dir}"
  register_model "${output_dir}"
done

for density in 0.2 0.4 0.6; do
  for seed in 42 43 44; do
    tag="d${density//./p}_s${seed}"
    output_dir="${MERGED_DIR}/${MODEL_PREFIX}dare_ties_${tag}"
    python src/merge_adapters.py \
      --base-model "${BASE_MODEL}" \
      --adapters "${MED_ADAPTER}" "${LEGAL_ADAPTER}" \
      --adapter-names medical legal \
      --combination dare_ties \
      --density "${density}" \
      --seed "${seed}" \
      --cache-dir "${CACHE_DIR}" \
      --output-dir "${output_dir}"
    register_model "${output_dir}"
  done
done

echo "Wrote merged model list to ${MERGED_MODEL_LIST}"
