#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zhangshangtong/Transformer/OFA"
PYTHON_BIN="/home/zhangshangtong/.conda/envs/OFA/bin/python"
LOG_DIR="${ROOT_DIR}/output/partial_pools"
mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

echo "[PartialPoolJob] $(date '+%F %T') waiting for existing generate_real_quant_pools jobs..."
while true; do
  existing_jobs="$(pgrep -af "python -m GraphhopSimhash.generate_real_quant_pools" || true)"
  if [[ -z "${existing_jobs}" ]]; then
    break
  fi
  echo "${existing_jobs}"
  sleep 300
done

echo "[PartialPoolJob] $(date '+%F %T') starting Cora/LLaMA W4A8 partial layers L4/L8/L16"
"${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A8 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --partial_layers 4 8 16 \
  --overwrite
echo "[PartialPoolJob] $(date '+%F %T') finished"
