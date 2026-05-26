#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

cd "${ROOT_DIR}"

mkdir -p output/token_compaction_pools/cora output/token_compaction/cora

generate_compact_pool() {
  local strategy="$1"
  local tag="$2"
  local out_path="cache_data/cora_llama2_7b_oracle_${tag}.pt"
  local log_path="output/token_compaction_pools/cora/cora_llama2_7b_${tag}.log"
  if [[ -f "${out_path}" ]]; then
    echo "[Skip] ${out_path} exists"
    return
  fi

  echo "[Generate] ${strategy} -> ${out_path}"
  "${PYTHON}" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets cora \
    --llm_name llama2_7b \
    --configs W4A8 \
    --batch_size 4 \
    --max_length 128 \
    --text_compaction_strategy "${strategy}" \
    --text_compaction_budget 128 \
    --text_compaction_chunk_words 32 \
    --awq_calib_samples 128 \
    --awq_seqlen 512 \
    --awq_q_group_size 128 \
    --output_path "${out_path}" \
    --overwrite > "${log_path}" 2>&1
}

# Prefix pool is shared with the graph_eager_token experiment.
if [[ ! -f cache_data/cora_llama2_7b_oracle_W4A8_S128.pt ]]; then
  generate_compact_pool prefix W4A8_S128
else
  echo "[Skip] cache_data/cora_llama2_7b_oracle_W4A8_S128.pt exists"
fi
generate_compact_pool random W4A8_S128_RANDOM
generate_compact_pool tfidf W4A8_S128_TFIDF
generate_compact_pool graph_context W4A8_S128_GRAPHCTX

eval_log="output/token_compaction/cora/cora_llama2_7b_token_compaction_$(date +%Y%m%d_%H%M%S).log"
echo "[Eval] token_compaction -> ${eval_log}"
"${PYTHON}" -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite token_compaction \
  --real_quant_model_name llama2_7b \
  --token_compaction_reference_tag W4A16 \
  --token_compaction_full_tag W4A8 \
  --token_compaction_tags W4A8_S128 W4A8_S128_RANDOM W4A8_S128_TFIDF W4A8_S128_GRAPHCTX \
  --token_compaction_names Prefix128 Random128 TFIDF128 GraphContext128 \
  --token_compaction_length 128 \
  --graph_eager_full_length 512 \
  --graph_eager_cost_scale 0.50 \
  --graph_eager_attn_weight 0.35 \
  --graph_eager_ffn_weight 0.65 | tee "${eval_log}"

echo "[Done] ${eval_log}"
