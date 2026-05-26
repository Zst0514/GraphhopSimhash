#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zhangshangtong/Transformer/OFA"
PYTHON_BIN="/home/zhangshangtong/.conda/envs/OFA/bin/python"
PARTIAL_LOG_DIR="${ROOT_DIR}/output/partial_encoder/cora"
PUBMED_DRIVER_LOG_DIR="${ROOT_DIR}/output/residual_reuse"
mkdir -p "${PARTIAL_LOG_DIR}" "${PUBMED_DRIVER_LOG_DIR}"

cd "${ROOT_DIR}"

echo "[PartialEvalQueue] $(date '+%F %T') waiting for Cora/LLaMA W4A8 partial pools..."
while true; do
  missing=0
  for layer in 4 8 16; do
    path="cache_data/cora_llama2_7b_oracle_W4A8_L${layer}.pt"
    if [[ ! -s "${path}" ]]; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  pgrep -af "run_cora_llama_partial_pools|python -m GraphhopSimhash.generate_real_quant_pools" || true
  sleep 300
done

partial_log="${PARTIAL_LOG_DIR}/cora_llama2_7b_partial_encoder_$(date +%Y%m%d_%H%M%S).log"
echo "[PartialEvalQueue] $(date '+%F %T') running Cora/LLaMA partial_encoder eval -> ${partial_log}"
"${PYTHON_BIN}" -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite partial_encoder \
  --real_quant_model_name llama2_7b \
  --partial_encoder_reference_tag W4A16 \
  --partial_encoder_full_tag W4A8 \
  --partial_encoder_partial_tag W4A8 \
  --partial_encoder_layers 4 8 16 \
  --partial_encoder_full_ratio 0.20 \
  --partial_encoder_deep_ratio 0.30 \
  --partial_encoder_mid_ratio 0.30 \
  > "${partial_log}" 2>&1

echo "[PartialEvalQueue] $(date '+%F %T') Cora partial_encoder eval finished"
echo "[PartialEvalQueue] $(date '+%F %T') starting PubMed/ST residual bias sweep"
bash GraphhopSimhash/run_pubmed_st_residual_bias_sweep.sh
echo "[PartialEvalQueue] $(date '+%F %T') all queued jobs finished"
