#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="$(cd "${REPO_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
MODEL_ID="${MODEL_ID:-modelscope/Llama-2-13b-ms}"
REVISION="${REVISION:-v1.0.2}"
LOCAL_DIR="${LOCAL_DIR:-${OFA_DIR}/models/llama-13b/modelscope/Llama-2-13b-ms}"
MAX_WORKERS="${MAX_WORKERS:-4}"

mkdir -p "$(dirname "${LOCAL_DIR}")"

echo "[Download] model=${MODEL_ID}"
echo "[Download] revision=${REVISION}"
echo "[Download] local_dir=${LOCAL_DIR}"
echo "[Download] mode=safetensors-only"

"${PYTHON_BIN}" - <<PY
from modelscope import snapshot_download

out = snapshot_download(
    "${MODEL_ID}",
    revision="${REVISION}",
    local_dir="${LOCAL_DIR}",
    ignore_file_pattern=[
        r"pytorch_model-.*\\.bin",
        r"pytorch_model\\.bin",
        r"pytorch_model\\.bin\\.index\\.json",
    ],
    max_workers=int("${MAX_WORKERS}"),
)
print(out)
PY

echo "[Download] done: ${LOCAL_DIR}"
