#!/usr/bin/env bash
set -euo pipefail

OFA_DIR="${OFA_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
DATASETS="${DATASETS:-tape_products arxiv}"
RUNS="${RUNS:-5}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/bfp_block_stress_sweep}"
GEN_CONFIGS="${GEN_CONFIGS:-W4BFPA4_B256 W4BFPA4_B512 W4BFPA4_B128_T16X8 W4BFPA4_B128_T32X4}"

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

for dataset in ${DATASETS}; do
  ds_out="${OUT_DIR}/${dataset}"
  mkdir -p "${ds_out}"

  missing=()
  for tag in ${GEN_CONFIGS}; do
    path="$(pool_path "${dataset}" "${tag}")"
    if [[ ! -f "${path}" ]]; then
      missing+=("${tag}")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "[$(date '+%F %T')] generating ${dataset}: ${missing[*]}"
    "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
      --datasets "${dataset}" \
      --llm_name llama2_7b \
      --configs "${missing[@]}" \
      --batch_size 4 \
      --max_length 512 \
      --awq_calib_samples 128 \
      --awq_seqlen 512 \
      --awq_q_group_size 128 \
      2>&1 | tee "${ds_out}/pool_generation.log"
  else
    echo "[$(date '+%F %T')] all requested pools already exist for ${dataset}"
  fi

  for tag in ${GEN_CONFIGS}; do
    log="${ds_out}/bfpa8_b128_vs_${tag}_runs${RUNS}.log"
    if [[ -f "${log}" ]] && grep -q "FINAL PRECISION-DEPTH SUMMARY" "${log}"; then
      echo "[$(date '+%F %T')] reuse existing ${log}"
      continue
    fi
    echo "[$(date '+%F %T')] ${dataset}: W4BFPA8_B128 vs ${tag}, runs=${RUNS}"
    "${PYTHON_BIN}" -m GraphhopSimhash \
      --datasets "${dataset}" \
      --runs "${RUNS}" \
      --experiment_suite precision_depth_ablation \
      --real_quant_model_name llama2_7b \
      --precision_depth_reference_tag W4BFPA8_B128 \
      --precision_depth_reference_bits 8 \
      --precision_depth_tags "${tag}" \
      --precision_depth_bits 4 \
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
      2>&1 | tee "${log}"
  done
done

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re

base = Path("output/bfp_block_stress_sweep")
rows = []
for log in sorted(base.glob("*/bfpa8_b128_vs_*_runs*.log")):
    text = log.read_text(errors="ignore")
    if "FINAL PRECISION-DEPTH SUMMARY" not in text:
        continue
    dataset = log.parent.name
    tag = log.name.split("bfpa8_b128_vs_", 1)[1].rsplit("_runs", 1)[0]
    baseline = re.findall(r"Baseline Acc:\s*([0-9.]+)", text)
    allp4 = re.findall(r"AllP4\s+\|.*?\|\s*([0-9.]+)\s*\|\s*([-0-9.]+)%", text)
    if baseline and allp4:
        acc, drop = allp4[-1]
        rows.append((dataset, tag, baseline[-1], acc, drop, str(log)))

out = base / "summary.tsv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    f.write("dataset\ttag\tfullp8_acc\ttarget_acc\tdrop\tlog\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
print(f"[BFPBlockStress] wrote {out}")
for row in rows:
    print("\t".join(row[:5]))
PY
