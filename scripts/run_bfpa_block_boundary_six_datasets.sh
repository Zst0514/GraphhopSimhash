#!/usr/bin/env bash
set -uo pipefail

OFA_DIR="${OFA_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS="${DATASETS:-cora pubmed wikics arxiv tape_products tape_arxiv23}"
BLOCKS="${BLOCKS:-128 256 512}"
BITS="${BITS:-6 5 4 3}"
RUNS="${RUNS:-10}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/bfpa_block_boundary_six_datasets}"
REFERENCE_TAG="${REFERENCE_TAG:-W4BFPA8_B128}"
WAIT_FOR_EXISTING="${WAIT_FOR_EXISTING:-1}"
RETRIES="${RETRIES:-1}"
FORCE_VALIDATE="${FORCE_VALIDATE:-0}"

mkdir -p "${OUT_ROOT}/logs"
cd "${OFA_DIR}" || exit 1

timestamp() {
  date +"%F %T"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

pool_path() {
  local dataset="$1"
  local tag="$2"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("${dataset}", "llama2_7b", "${tag}"))
PY
}

record_failure() {
  local dataset="$1"
  local stage="$2"
  local item="$3"
  local note="$4"
  printf "%s\t%s\t%s\t%s\t%s\n" "$(timestamp)" "${dataset}" "${stage}" "${item}" "${note}" >> "${OUT_ROOT}/failures.tsv"
}

wait_for_existing_generation() {
  [[ "${WAIT_FOR_EXISTING}" == "1" ]] || return 0
  while true; do
    local procs
    procs="$(
      ps -eo pid=,args= \
        | awk '/python/ && /-m GraphhopSimhash.generate_real_quant_pools/ {print}' \
        | grep -v "awk " \
        || true
    )"
    if [[ -z "${procs}" ]]; then
      return 0
    fi
    log_msg "[Wait] existing pool generation is running; waiting 10 min before starting another generator"
    echo "${procs}" | sed 's/^/[Wait] /'
    sleep 600
  done
}

ensure_pool() {
  local dataset="$1"
  local tag="$2"
  local path
  path="$(pool_path "${dataset}" "${tag}")"
  if [[ -f "${path}" ]]; then
    log_msg "[Pool] ${dataset} ${tag}: exists"
    return 0
  fi

  wait_for_existing_generation
  local attempt
  for attempt in $(seq 1 "${RETRIES}"); do
    log_msg "[Pool] generating ${dataset} ${tag} attempt=${attempt}/${RETRIES}"
    if "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
      --datasets "${dataset}" \
      --llm_name llama2_7b \
      --configs "${tag}" \
      --batch_size 4 \
      --max_length 512 \
      --awq_calib_samples 128 \
      --awq_seqlen 512 \
      --awq_q_group_size 128 \
      2>&1 | tee "${OUT_ROOT}/logs/${dataset}_${tag}_generate_attempt${attempt}.log"; then
      path="$(pool_path "${dataset}" "${tag}")"
      if [[ -f "${path}" ]]; then
        log_msg "[Pool] ${dataset} ${tag}: generated"
        return 0
      fi
    fi
    log_msg "[Pool] ${dataset} ${tag}: generation attempt failed"
  done

  record_failure "${dataset}" "generate" "${tag}" "pool missing after retries"
  return 1
}

validate_target() {
  local dataset="$1"
  local block="$2"
  local bit="$3"
  local tag="W4BFPA${bit}_B${block}"
  local out_dir="${OUT_ROOT}/boundary/${dataset}/B${block}"
  local log="${out_dir}/bfpa8_b128_vs_p${bit}_b${block}_runs${RUNS}.log"
  mkdir -p "${out_dir}"

  if [[ "${FORCE_VALIDATE}" != "1" ]] && [[ -f "${log}" ]] && grep -q "FINAL PRECISION-DEPTH SUMMARY" "${log}"; then
    log_msg "[Validate] ${dataset} ${tag}: reuse existing ${log}"
    return 0
  fi

  log_msg "[Validate] ${dataset}: ${REFERENCE_TAG} vs ${tag}, runs=${RUNS}"
  if "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets "${dataset}" \
    --runs "${RUNS}" \
    --experiment_suite precision_depth_ablation \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag "${REFERENCE_TAG}" \
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
    --score_low_unique_weight 1 \
    2>&1 | tee "${log}"; then
    return 0
  fi

  record_failure "${dataset}" "validate" "${tag}" "GraphhopSimhash run failed"
  return 1
}

summarize_results() {
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import re

out_root = Path("${OUT_ROOT}")
datasets = "${DATASETS}".split()
blocks = [int(x) for x in "${BLOCKS}".split()]
bits = [int(x) for x in "${BITS}".split()]
runs = "${RUNS}"

display = {
    "cora": "Cora",
    "pubmed": "PubMed",
    "wikics": "Wiki-CS",
    "arxiv": "OGBN-Arxiv",
    "tape_products": "Products-subset",
    "tape_arxiv23": "TAPE-Arxiv23",
}

def parse_log(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if "FINAL PRECISION-DEPTH SUMMARY" not in text:
        return None
    baseline = re.findall(r"Baseline Acc:\\s*([0-9.]+)", text)
    allp = re.findall(r"AllP(?:\\d+)\\s+\\|.*?\\|\\s*([0-9.]+)\\s*\\|\\s*([-0-9.]+)%\\s*\\|\\s*([0-9.]+)", text)
    if not baseline or not allp:
        return None
    acc, drop, avgerr = allp[-1]
    return {
        "fullp8_acc": baseline[-1],
        "target_acc": acc,
        "drop": drop,
        "avgerr": avgerr,
    }

rows = []
for dataset in datasets:
    for block in blocks:
        row = {"dataset": dataset, "display": display.get(dataset, dataset), "block": str(block)}
        fullp8 = None
        complete = True
        for bit in bits:
            log = out_root / "boundary" / dataset / f"B{block}" / f"bfpa8_b128_vs_p{bit}_b{block}_runs{runs}.log"
            parsed = parse_log(log)
            if parsed is None:
                complete = False
                row[f"bfpa{bit}_drop"] = ""
                row[f"bfpa{bit}_acc"] = ""
                row[f"bfpa{bit}_avgerr"] = ""
            else:
                fullp8 = fullp8 or parsed["fullp8_acc"]
                row[f"bfpa{bit}_drop"] = parsed["drop"]
                row[f"bfpa{bit}_acc"] = parsed["target_acc"]
                row[f"bfpa{bit}_avgerr"] = parsed["avgerr"]
        row["fullp8_acc"] = fullp8 or ""
        row["complete"] = "yes" if complete else "no"
        rows.append(row)

summary = out_root / "summary.tsv"
with summary.open("w") as f:
    fields = ["dataset", "block", "complete", "fullp8_acc"] + [f"bfpa{b}_drop" for b in bits] + [f"bfpa{b}_acc" for b in bits] + [f"bfpa{b}_avgerr" for b in bits]
    f.write("\\t".join(fields) + "\\n")
    for row in rows:
        f.write("\\t".join(row.get(k, "") for k in fields) + "\\n")

tex_rows = out_root / "table_rows.tex"
with tex_rows.open("w") as f:
    for row in rows:
        if row["complete"] != "yes":
            continue
        f.write(
            f"{row['display']} & B{row['block']} & {row['fullp8_acc']} & "
            f"{row['bfpa6_drop']}\\\\% & {row['bfpa5_drop']}\\\\% & "
            f"{row['bfpa4_drop']}\\\\% & {row['bfpa3_drop']}\\\\% \\\\\\\\n"
        )

tex_table = out_root / "table_full.tex"
with tex_table.open("w") as f:
    f.write("""\\\\begin{table}[t]
  \\\\centering
  \\\\caption{BFPA precision boundary. Drop is measured against the corresponding W4BFPA8\\\\_B128 encoder target pool, without SimHash or residual frontend reuse.}
  \\\\label{tab:bfpa-boundary}
  \\\\begin{tabular}{lcccccc}
    \\\\toprule
    Dataset & Block & FullP8 Acc. & BFPA6 & BFPA5 & BFPA4 & BFPA3 \\\\\\\\
    \\\\midrule
""")
    if tex_rows.exists():
        f.write(tex_rows.read_text())
    f.write("""    \\\\bottomrule
  \\\\end{tabular}
\\\\end{table}
""")

print(f"[BFPA boundary] wrote {summary}")
print(f"[BFPA boundary] wrote {tex_rows}")
print(f"[BFPA boundary] wrote {tex_table}")
for row in rows:
    print("\\t".join([row["dataset"], "B"+row["block"], row["complete"], row["fullp8_acc"], row.get("bfpa6_drop",""), row.get("bfpa5_drop",""), row.get("bfpa4_drop",""), row.get("bfpa3_drop","")]))
PY
}

if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  summarize_results
  exit 0
fi

log_msg "BFPA block boundary suite starts"
log_msg "datasets=${DATASETS}"
log_msg "blocks=${BLOCKS}"
log_msg "bits=${BITS}"
log_msg "runs=${RUNS}"
log_msg "out=${OUT_ROOT}"

printf "time\tdataset\tstage\titem\tnote\n" > "${OUT_ROOT}/failures.tsv"

for dataset in ${DATASETS}; do
  log_msg "===== dataset=${dataset} ====="
  ensure_pool "${dataset}" "${REFERENCE_TAG}" || continue
  for block in ${BLOCKS}; do
    log_msg "===== dataset=${dataset} block=B${block} ====="
    for bit in ${BITS}; do
      tag="W4BFPA${bit}_B${block}"
      if ensure_pool "${dataset}" "${tag}"; then
        validate_target "${dataset}" "${block}" "${bit}" || true
      fi
      summarize_results || true
    done
  done
done

summarize_results
log_msg "BFPA block boundary suite completed"
