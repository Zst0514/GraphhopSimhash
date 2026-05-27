#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

cd "${ROOT_DIR}"

mkdir -p output/graph_eager_token_pools/cora output/graph_eager_token/cora

generate_if_missing() {
  local length="$1"
  local out_path="cache_data/cora_llama2_7b_oracle_W4A8_S${length}.pt"
  local log_path="output/graph_eager_token_pools/cora/cora_llama2_7b_W4A8_S${length}.log"
  if [[ -f "${out_path}" ]]; then
    echo "[Skip] ${out_path} exists"
    return
  fi

  echo "[Generate] ${out_path} -> ${log_path}"
  "${PYTHON}" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets cora \
    --llm_name llama2_7b \
    --configs W4A8 \
    --batch_size 4 \
    --max_length "${length}" \
    --awq_calib_samples 128 \
    --awq_seqlen 512 \
    --awq_q_group_size 128 \
    --output_path "${out_path}" \
    --overwrite > "${log_path}" 2>&1
}

generate_if_missing 128
generate_if_missing 256

eval_log="output/graph_eager_token/cora/cora_llama2_7b_graph_eager_token_$(date +%Y%m%d_%H%M%S).log"
echo "[Eval] graph_eager_token -> ${eval_log}"
"${PYTHON}" -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite graph_eager_token \
  --real_quant_model_name llama2_7b \
  --graph_eager_reference_tag W4A16 \
  --graph_eager_full_tag W4A8 \
  --graph_eager_token_tag_prefix W4A8_S \
  --graph_eager_token_lengths 128 256 \
  --graph_eager_full_length 512 \
  --graph_eager_full_ratio 0.20 \
  --graph_eager_mid_ratio 0.30 \
  --graph_eager_predictor_calib_samples 512 \
  --graph_eager_predictor_target embedding \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 | tee "${eval_log}"

echo "[Done] ${eval_log}"
