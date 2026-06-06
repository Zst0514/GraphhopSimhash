#!/usr/bin/env bash
set -euo pipefail

# Final BFP validation suite for the current paper mainline.
#
# It runs three groups:
#   1. BFPA safety boundary: BFPA8 reference vs BFPA6/5/4/3.
#   2. Dynamic refinement necessity: BFPA4 base + selected BFPA6 lift.
#   3. Full-stack dynamic BFP: SimHash/residual frontend + dynamic BFP backend.
#
# Default choices are intentionally conservative for overnight/background runs:
#   Cora:   5 runs
#   PubMed: 3 runs
#   Arxiv:  1 run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS=(${DATASETS:-cora pubmed arxiv})
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/final_bfp_validation}"
RUN_BOUNDARY="${RUN_BOUNDARY:-1}"
RUN_REFINEMENT="${RUN_REFINEMENT:-1}"
RUN_FULLSTACK="${RUN_FULLSTACK:-1}"
GENERATE_MISSING="${GENERATE_MISSING:-1}"

mkdir -p "${OUT_ROOT}/logs"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

dataset_runs() {
  case "$1" in
    cora) echo "${CORA_RUNS:-5}" ;;
    pubmed) echo "${PUBMED_RUNS:-3}" ;;
    arxiv) echo "${ARXIV_RUNS:-1}" ;;
    *) echo 1 ;;
  esac
}

frontend_t() {
  case "$1" in
    cora) echo "${CORA_T:-31}" ;;
    pubmed) echo "${PUBMED_T:-31}" ;;
    arxiv) echo "${ARXIV_T:-22}" ;;
    *) echo 31 ;;
  esac
}

pool_path() {
  local dataset="$1"
  local tag="$2"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("${dataset}", "llama2_7b", "${tag}"))
PY
}

ensure_pool() {
  local dataset="$1"
  local tag="$2"
  local path
  path="$(pool_path "${dataset}" "${tag}")"
  if [[ -f "${path}" ]]; then
    echo "[$(timestamp)] [Pool] ${dataset} ${tag}: exists ${path}"
    return 0
  fi
  if [[ "${GENERATE_MISSING}" != "1" ]]; then
    echo "[$(timestamp)] [Pool] ${dataset} ${tag}: missing, skip generation"
    return 1
  fi
  echo "[$(timestamp)] [Pool] generating ${dataset} ${tag}"
  "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets "${dataset}" \
    --llm_name llama2_7b \
    --configs "${tag}" \
    --batch_size 4 \
    --max_length 512 \
    --awq_calib_samples 128 \
    --awq_seqlen 512 \
    --awq_q_group_size 128
}

run_boundary() {
  local dataset="$1"
  local runs="$2"
  local out_dir="${OUT_ROOT}/boundary/${dataset}"
  mkdir -p "${out_dir}"

  for tag in W4BFPA8_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128 W4BFPA3_B128; do
    ensure_pool "${dataset}" "${tag}"
  done

  for bit in 6 5 4 3; do
    local tag="W4BFPA${bit}_B128"
    local log="${out_dir}/bfpa8_vs_p${bit}_runs${runs}.log"
    if [[ "${FORCE_BOUNDARY:-0}" != "1" ]] && [[ -f "${log}" ]] && grep -q "FINAL PRECISION-DEPTH SUMMARY" "${log}"; then
      echo "[$(timestamp)] [Boundary] ${dataset} P8 vs P${bit}: reuse existing ${log}"
      continue
    fi
    echo "[$(timestamp)] [Boundary] ${dataset} P8 vs P${bit} runs=${runs} -> ${log}"
    "${PYTHON_BIN}" -m GraphhopSimhash \
      --datasets "${dataset}" \
      --runs "${runs}" \
      --experiment_suite precision_depth_ablation \
      --real_quant_model_name llama2_7b \
      --precision_depth_reference_tag W4BFPA8_B128 \
      --precision_depth_reference_bits 8 \
      --precision_depth_tags "${tag}" \
      --precision_depth_bits "${bit}" \
      --precision_depth_cost_scale 0.50 \
      --precision_depth_fixed_cost 0.15 \
      --precision_depth_high_ratio 0.0 \
      --precision_depth_mid_ratio 0.0 \
      --precision_depth_low_ratio 0.0 \
      --precision_depth_budget_priorities random degree tser \
      --learned_hash_epochs 10 \
      --learned_hash_dim 128 \
      --hash_heads_per_route 8 \
      --main_hash_head_bits 16 16 16 16 16 16 16 16 \
      --radius 2 \
      --hamming_only_acceptor \
      --enable_score_gate \
      --allow_rare_fuzzy \
      --score_reuse_threshold "$(frontend_t "${dataset}")" \
      --score_propagation_weight 3 \
      --score_graph_context_weight 1 \
      --score_low_unique_weight 1 2>&1 | tee "${log}"
  done
}

run_refinement() {
  local dataset="$1"
  local runs="$2"
  local out_dir="${OUT_ROOT}/refinement/${dataset}_runs${runs}"
  mkdir -p "${out_dir}"
  echo "[$(timestamp)] [Refinement] ${dataset} runs=${runs} -> ${out_dir}"
  "${PYTHON_BIN}" GraphhopSimhash/scripts/evaluate_graphbfp_stress_refinement.py \
    --dataset "${dataset}" \
    --runs "${runs}" \
    --reference_tag W4BFPA8_B128 \
    --base_tag W4BFPA4_B128 \
    --refine_tag W4BFPA6_B128 \
    --ratios 0.05 0.10 0.15 0.20 0.25 0.30 0.40 \
    --policies Random Stress Degree TSER DegreeXStress TSERXStress DegreePlusStress TSERPlusStress \
    --stress_metric outlier_p90 \
    --zero_weight 0.25 \
    --output_dir "${out_dir}" 2>&1 | tee "${out_dir}/run.log"
}

run_fullstack() {
  local dataset="$1"
  local runs="$2"
  local t
  t="$(frontend_t "${dataset}")"
  local log="${OUT_ROOT}/fullstack/${dataset}_T${t}_runs${runs}.log"
  mkdir -p "$(dirname "${log}")"
  echo "[$(timestamp)] [FullStack] ${dataset} T=${t} runs=${runs} -> ${log}"
  FORCE_DYNAMIC="${FORCE_DYNAMIC:-0}" \
  FORCE_FULLSTACK="${FORCE_FULLSTACK:-0}" \
  DATASETS="${dataset}" \
  RUNS="${runs}" \
  THRESHOLD="${t}" \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh 2>&1 | tee "${log}"
}

echo "================================================================"
echo "[$(timestamp)] Final BFP validation suite"
echo "datasets=${DATASETS[*]}"
echo "out=${OUT_ROOT}"
echo "run_boundary=${RUN_BOUNDARY} run_refinement=${RUN_REFINEMENT} run_fullstack=${RUN_FULLSTACK}"
echo "================================================================"

for dataset in "${DATASETS[@]}"; do
  runs="$(dataset_runs "${dataset}")"
  echo "================================================================"
  echo "[$(timestamp)] Dataset=${dataset} runs=${runs}"
  echo "================================================================"
  if [[ "${RUN_BOUNDARY}" == "1" ]]; then
    run_boundary "${dataset}" "${runs}"
  fi
  if [[ "${RUN_REFINEMENT}" == "1" ]]; then
    run_refinement "${dataset}" "${runs}"
  fi
  if [[ "${RUN_FULLSTACK}" == "1" ]]; then
    run_fullstack "${dataset}" "${runs}"
  fi
done

echo "[$(timestamp)] Final BFP validation suite completed."
