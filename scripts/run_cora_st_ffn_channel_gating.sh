#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

for item in "0.75 FFN75" "0.50 FFN50" "0.25 FFN25"; do
  read -r keep suffix <<< "${item}"
  "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets cora \
    --llm_name ST \
    --configs W4A8 \
    --batch_size 128 \
    --awq_calib_samples 16 \
    --awq_seqlen 128 \
    --ffn_channel_gating \
    --ffn_gate_keep_ratio "${keep}" \
    --ffn_gate_group_size 64 \
    --ffn_gate_calib_samples 256 \
    --ffn_gate_calibration_strategy random \
    --tag_suffix "${suffix}" \
    --overwrite
done

"${PYTHON_BIN}" -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite ffn_channel_gating \
  --real_quant_model_name ST \
  --ffn_gating_reference_tag FP16 \
  --ffn_gating_full_tag W4A8 \
  --ffn_gating_tags W4A8_FFN75 W4A8_FFN50 \
  --ffn_gating_names FFN75 FFN50 \
  --ffn_gating_keep_ratios 0.75 0.50 \
  --ffn_gating_route_ratios 0.20 0.40 0.60
