#!/usr/bin/env bash
set -euo pipefail

OFA_DIR="${OFA_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/final_bfp_validation_runs10/boundary/tape_products}"
RUNS="${RUNS:-10}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"

mkdir -p "${OUT_DIR}"
cd "${OFA_DIR}"

pool_path() {
  local tag="$1"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("tape_products", "llama2_7b", "${tag}"))
PY
}

while true; do
  missing=0
  for tag in W4BFPA8_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128 W4BFPA3_B128; do
    path="$(pool_path "${tag}")"
    if [[ ! -f "${path}" ]]; then
      echo "[$(date '+%F %T')] waiting for ${path}"
      missing=1
    fi
  done
  [[ "${missing}" == "0" ]] && break
  sleep "${WAIT_SECONDS}"
done

for bit in 6 5 4 3; do
  tag="W4BFPA${bit}_B128"
  log="${OUT_DIR}/bfpa8_vs_p${bit}_runs${RUNS}.log"
  if [[ -f "${log}" ]] && grep -q "FINAL PRECISION-DEPTH SUMMARY" "${log}"; then
    echo "[$(date '+%F %T')] reuse existing ${log}"
    continue
  fi
  echo "[$(date '+%F %T')] tape_products P8 vs P${bit} runs=${RUNS} -> ${log}"
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets tape_products \
    --runs "${RUNS}" \
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
    --score_reuse_threshold 31 \
    --score_propagation_weight 3 \
    --score_graph_context_weight 1 \
    --score_low_unique_weight 1 2>&1 | tee "${log}"
done
