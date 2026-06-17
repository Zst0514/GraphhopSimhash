#!/usr/bin/env bash
set -euo pipefail

OFA_DIR="${OFA_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
DATASETS="${DATASETS:-cora pubmed arxiv}"
CONFIGS="${CONFIGS:-W4BFPA4_B256 W4BFPA4_B512}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/bfp_block_size_pool_generation}"
WAIT_FOR_EXISTING="${WAIT_FOR_EXISTING:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
RETRIES="${RETRIES:-2}"

mkdir -p "${OUT_DIR}"
cd "${OFA_DIR}"

pool_path() {
  local dataset="$1"
  local tag="$2"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("${dataset}", "llama2_7b", "${tag}"))
PY
}

if [[ "${WAIT_FOR_EXISTING}" == "1" ]]; then
  while pgrep -f "GraphhopSimhash.generate_real_quant_pools" >/dev/null; do
    echo "[$(date '+%F %T')] waiting for existing generate_real_quant_pools jobs"
    sleep "${WAIT_SECONDS}"
  done
fi

failed=()

for dataset in ${DATASETS}; do
  missing=()
  for tag in ${CONFIGS}; do
    path="$(pool_path "${dataset}" "${tag}")"
    if [[ -f "${path}" ]]; then
      echo "[$(date '+%F %T')] exists: ${path}"
    else
      missing+=("${tag}")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "[$(date '+%F %T')] ${dataset}: all requested pools exist"
    continue
  fi

  log="${OUT_DIR}/${dataset}_$(date '+%Y%m%d_%H%M%S').log"
  echo "[$(date '+%F %T')] ${dataset}: generating ${missing[*]} -> ${log}"
  ok=0
  for attempt in $(seq 1 "${RETRIES}"); do
    echo "[$(date '+%F %T')] ${dataset}: attempt ${attempt}/${RETRIES}" | tee -a "${log}"
    if "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
      --datasets "${dataset}" \
      --llm_name llama2_7b \
      --configs "${missing[@]}" \
      --batch_size 4 \
      --max_length 512 \
      --awq_calib_samples 128 \
      --awq_seqlen 512 \
      --awq_q_group_size 128 \
      2>&1 | tee -a "${log}"; then
      ok=1
      break
    fi
    echo "[$(date '+%F %T')] ${dataset}: attempt ${attempt} failed" | tee -a "${log}"
    sleep 60
  done
  if [[ "${ok}" != "1" ]]; then
    echo "[$(date '+%F %T')] ${dataset}: FAILED after ${RETRIES} attempts" | tee -a "${log}"
    failed+=("${dataset}:${missing[*]}")
  fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "[$(date '+%F %T')] done with failures:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi

echo "[$(date '+%F %T')] done"
