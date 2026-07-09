#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
CACHE_DIR=${CACHE_DIR:-/root/autodl-tmp/hf-cache}
MED_ADAPTER=${MED_ADAPTER:-adapters/medical_lora_full}
LEGAL_ADAPTER=${LEGAL_ADAPTER:-adapters/legal_lora_full}
MAX_SAMPLES=${MAX_SAMPLES:-0}
RESULTS_DIR=${RESULTS_DIR:-results/raw_full}
MERGED_DIR=${MERGED_DIR:-merged}
MODEL_PREFIX=${MODEL_PREFIX:-full_}
MERGED_MODEL_LIST=${MERGED_MODEL_LIST:-${MERGED_DIR}/full_level1_sweep_models.txt}
PLOT_RESULTS=${PLOT_RESULTS:-1}
TABLE_OUTPUT=${TABLE_OUTPUT:-results/tables/full_main_results.csv}
FIGURE_DIR=${FIGURE_DIR:-results/figures_full}

require_adapter() {
  local adapter=$1
  if [[ ! -f "${adapter}/adapter_config.json" || ! -f "${adapter}/adapter_model.safetensors" ]]; then
    echo "Missing PEFT adapter files in ${adapter}" >&2
    echo "Set MED_ADAPTER and LEGAL_ADAPTER to concrete adapter directories." >&2
    exit 1
  fi
}

load_merged_models() {
  MERGED_MODELS=()
  if [[ -f "${MERGED_MODEL_LIST}" ]]; then
    while IFS= read -r model_dir; do
      [[ -n "${model_dir}" && "${model_dir}" != \#* ]] || continue
      MERGED_MODELS+=("${model_dir}")
    done < "${MERGED_MODEL_LIST}"
  else
    local names=(
      task_arithmetic_m0p5_l0p5
      task_arithmetic_m0p7_l0p7
      task_arithmetic_m1p0_l1p0
      task_arithmetic_m1p2_l1p2
      ties_d0p2
      ties_d0p4
      ties_d0p6
      ties_d0p8
      dare_ties_d0p2_s42
      dare_ties_d0p2_s43
      dare_ties_d0p2_s44
      dare_ties_d0p4_s42
      dare_ties_d0p4_s43
      dare_ties_d0p4_s44
      dare_ties_d0p6_s42
      dare_ties_d0p6_s43
      dare_ties_d0p6_s44
    )
    for name in "${names[@]}"; do
      MERGED_MODELS+=("${MERGED_DIR}/${MODEL_PREFIX}${name}")
    done
  fi
}

require_input() {
  local input=$1
  if [[ ! -f "${input}" ]]; then
    echo "Missing evaluation input: ${input}" >&2
    exit 1
  fi
}

require_model_dir() {
  local model_dir=$1
  if [[ ! -d "${model_dir}" ]]; then
    echo "Missing merged model directory: ${model_dir}" >&2
    echo "Run scripts/run_level1_sweep.sh first, or set MERGED_MODEL_LIST." >&2
    exit 1
  fi
}

summary_for_output() {
  local output=$1
  printf '%s.summary.json\n' "${output%.csv}"
}

require_adapter "${MED_ADAPTER}"
require_adapter "${LEGAL_ADAPTER}"
load_merged_models

mkdir -p "${RESULTS_DIR}"
SUMMARY_PATHS=()

evaluate_full() {
  local name=$1
  local model=$2
  local split=$3
  local task=$4
  local input="data/processed/${task}_${split}.jsonl"
  local output="${RESULTS_DIR}/${name}_${task}_${split}.csv"
  require_input "${input}"
  python src/evaluate_mcq.py \
    --model "${model}" \
    --input "${input}" \
    --output "${output}" \
    --cache-dir "${CACHE_DIR}" \
    --max-samples "${MAX_SAMPLES}"
  SUMMARY_PATHS+=("$(summary_for_output "${output}")")
}

evaluate_adapter() {
  local name=$1
  local adapter=$2
  local split=$3
  local task=$4
  local input="data/processed/${task}_${split}.jsonl"
  local output="${RESULTS_DIR}/${name}_${task}_${split}.csv"
  require_input "${input}"
  python src/evaluate_mcq.py \
    --model "${BASE_MODEL}" \
    --base-model "${BASE_MODEL}" \
    --adapter "${adapter}" \
    --input "${input}" \
    --output "${output}" \
    --cache-dir "${CACHE_DIR}" \
    --max-samples "${MAX_SAMPLES}"
  SUMMARY_PATHS+=("$(summary_for_output "${output}")")
}

for split in val test; do
  for task in medmcqa casehold; do
    evaluate_full base "${BASE_MODEL}" "${split}" "${task}"
    evaluate_adapter medical_expert "${MED_ADAPTER}" "${split}" "${task}"
    evaluate_adapter legal_expert "${LEGAL_ADAPTER}" "${split}" "${task}"
    for model_dir in "${MERGED_MODELS[@]}"; do
      require_model_dir "${model_dir}"
      evaluate_full "$(basename "${model_dir}")" "${model_dir}" "${split}" "${task}"
    done
  done
done

if [[ "${PLOT_RESULTS}" == "1" ]]; then
  python src/plot_results.py \
    --summaries "${SUMMARY_PATHS[@]}" \
    --table-output "${TABLE_OUTPUT}" \
    --figure-dir "${FIGURE_DIR}"
fi
