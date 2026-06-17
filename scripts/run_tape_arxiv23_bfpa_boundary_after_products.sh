#!/usr/bin/env bash
set -euo pipefail

OFA_DIR="${OFA_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
WAIT_PID="${WAIT_PID:-}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/final_bfp_validation_runs10/boundary/tape_arxiv23}"
RUNS="${RUNS:-10}"

mkdir -p "${OUT_DIR}"
cd "${OFA_DIR}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "[$(date '+%F %T')] waiting for pid ${WAIT_PID} before tape_arxiv23 generation"
    sleep 300
  done
fi

"${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
  --datasets tape_arxiv23 \
  --llm_name llama2_7b \
  --configs W4BFPA8_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128 W4BFPA3_B128 \
  --batch_size 4 \
  --max_length 512 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --awq_q_group_size 128

for bit in 6 5 4 3; do
  tag="W4BFPA${bit}_B128"
  log="${OUT_DIR}/bfpa8_vs_p${bit}_runs${RUNS}.log"
  echo "[$(date '+%F %T')] tape_arxiv23 P8 vs P${bit} runs=${RUNS} -> ${log}"
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets tape_arxiv23 \
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
