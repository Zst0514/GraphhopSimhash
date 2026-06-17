#!/usr/bin/env bash
set -euo pipefail

OFA_DIR="${OFA_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
MODEL_DIR="${MODEL_DIR:-${OFA_DIR}/models/llama-13b/modelscope/Llama-2-13b-ms}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/llama13b_bfpa256_validation}"
DATASETS=(${DATASETS:-cora pubmed arxiv wikics tape_products tape_arxiv23})
REFERENCE_DATASETS=(${REFERENCE_DATASETS:-pubmed arxiv wikics tape_products tape_arxiv23})
WAIT_PID="${WAIT_PID:-}"
RUNS="${RUNS:-10}"

mkdir -p "${OUT_ROOT}/logs"
cd "${OFA_DIR}"

timestamp() {
  date +"%F %T"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

if [[ -n "${WAIT_PID}" ]]; then
  while ps -p "${WAIT_PID}" >/dev/null 2>&1; do
    log_msg "waiting pid=${WAIT_PID}"
    sleep 600
  done
fi

log_msg "start llama2_13b W4A8 reference generation"
for dataset in "${REFERENCE_DATASETS[@]}"; do
  path="cache_data/${dataset}_llama2_13b_oracle_W4A8.pt"
  if [[ -s "${path}" ]]; then
    log_msg "skip existing ${path}"
    continue
  fi

  log_msg "generate ${dataset} W4A8"
  GRAPHHOP_LLAMA2_13B_PATH="${MODEL_DIR}" \
    "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
      --datasets "${dataset}" \
      --llm_name llama2_13b \
      --configs W4A8 \
      --batch_size 1 \
      --max_length 500 \
      --cache_dir cache_data/model \
      --awq_calib_samples 128 \
      --awq_seqlen 512 \
      --awq_q_group_size 128 \
      > "${OUT_ROOT}/logs/${dataset}_W4A8_generate.log" 2>&1
done

log_msg "start llama2_13b BFPA B256 validation"
for dataset in "${DATASETS[@]}"; do
  out_dir="${OUT_ROOT}/${dataset}"
  log="${out_dir}/w4a8_vs_bfpa3456_b256_runs${RUNS}.log"
  mkdir -p "${out_dir}"

  if [[ -f "${log}" ]] && grep -q "FINAL PRECISION-DEPTH SUMMARY" "${log}"; then
    log_msg "skip existing validation ${dataset}"
    continue
  fi

  log_msg "validate ${dataset}"
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets "${dataset}" \
    --runs "${RUNS}" \
    --experiment_suite precision_depth_ablation \
    --real_quant_model_name llama2_13b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_reference_bits 8 \
    --precision_depth_tags W4BFPA6_B256 W4BFPA5_B256 W4BFPA4_B256 W4BFPA3_B256 \
    --precision_depth_bits 6 5 4 3 \
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
    --score_low_unique_weight 1 \
    > "${log}" 2>&1
  log_msg "done validation ${dataset} -> ${log}"
done

log_msg "llama2_13b BFPA B256 validation queue completed"
