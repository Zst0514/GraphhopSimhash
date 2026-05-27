#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m GraphhopSimhash \
  --datasets cora \
  --runs "${RUNS:-10}" \
  --experiment_suite hierarchical_encoder \
  --real_quant_model_name ST \
  --hierarchical_reference_tag FP16 \
  --hierarchical_full_tag W4A8_PTQ_TEST \
  --hierarchical_gated_tag W4A8_FFN75 \
  --hierarchical_gated_keep_ratio 0.75 \
  --hierarchical_gated_route_ratio "${GATED_ROUTE_RATIO:-0.20}" \
  --hierarchical_gated_route_policy "${GATED_ROUTE_POLICY:-tser}" \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --score_reuse_threshold "${SCORE_REUSE_THRESHOLD:-30}" \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --residual_rank 32 \
  --residual_epochs "${RESIDUAL_EPOCHS:-60}" \
  --residual_max_train_pairs 1024 \
  --residual_min_dist 1 \
  --residual_direct_threshold -1
