#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zhangshangtong/Transformer/OFA"
REPO_DIR="${ROOT_DIR}/GraphhopSimhash"
PYTHON_BIN="/home/zhangshangtong/.conda/envs/OFA/bin/python"
OUT_DIR="${ROOT_DIR}/output/residual_reuse/pubmed_st_bias_sweep"
SUMMARY_TSV="${OUT_DIR}/summary.tsv"
SUMMARY_MD="${OUT_DIR}/summary.md"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

RUNS="${RUNS:-3}"
THRESHOLDS=(${THRESHOLDS:-20 30 45 60})
BIASES=(${BIASES:-0 1 2})

echo "[PubMedResidualBias] $(date '+%F %T') waiting for existing LLaMA pool generation jobs..."
while true; do
  existing_jobs="$(
    pgrep -af "python -m GraphhopSimhash.generate_real_quant_pools" \
      | rg -v "run_pubmed_st_residual_bias_sweep|pgrep -af" || true
  )"
  if [[ -z "${existing_jobs}" ]]; then
    break
  fi
  echo "${existing_jobs}"
  sleep 300
done

printf "threshold\tconfidence_bias\tdirect_reuse\tdirect_drop\tdirect_acc\tdirect_avgerr\tdirect_hiterr\tresidual_reuse\tresidual_drop\tresidual_acc\tresidual_avgerr\tresidual_hiterr\talpha\ttrain_pairs\tlog\n" > "${SUMMARY_TSV}"

for bias in "${BIASES[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    log="${OUT_DIR}/pubmed_ST_tser311_T${threshold}_bias${bias}.log"
    echo "[PubMedResidualBias] $(date '+%F %T') running T=${threshold}, confidence_bias=${bias}, runs=${RUNS}"
    "${PYTHON_BIN}" -m GraphhopSimhash \
      --datasets pubmed \
      --runs "${RUNS}" \
      --experiment_suite residual_reuse \
      --radius 2 \
      --hash_heads_per_route 4 \
      --main_hash_head_bits 16 16 16 16 \
      --learned_hash_epochs 10 \
      --learned_hash_dim 128 \
      --hamming_only_acceptor \
      --enable_score_gate \
      --allow_rare_fuzzy \
      --score_reuse_threshold "${threshold}" \
      --score_propagation_weight 3 \
      --score_graph_context_weight 1 \
      --score_low_unique_weight 1 \
      --score_pair_confidence_discount "${bias}" \
      --residual_rank 32 \
      --residual_epochs 100 \
      --residual_max_train_pairs 1024 \
      --residual_min_dist 1.0 \
      > "${log}" 2>&1

    "${PYTHON_BIN}" - "${log}" "${SUMMARY_TSV}" "${threshold}" "${bias}" <<'PY'
import re
import sys

log_path, summary_path, threshold, bias = sys.argv[1:5]
text = open(log_path, "r", encoding="utf-8", errors="replace").read()

def parse_row(name):
    pattern = (
        rf"^{name}\s*\|\s*"
        r"(?P<reuse>[0-9.]+%)\s*\|\s*"
        r"(?P<trainpairs>[-0-9.]+)\s*\|\s*"
        r"(?P<acc>[0-9.]+)\s*\|\s*"
        r"(?P<drop>[0-9.]+%)\s*\|\s*"
        r"(?P<avgerr>[0-9.]+)\s*\|\s*"
        r"(?P<hiterr>[0-9.]+)\s*\|\s*"
        r"(?P<alpha>[-0-9.]+)\s*\|"
    )
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise SystemExit(f"Could not parse {name} from {log_path}")
    return matches[-1].groupdict()

direct = parse_row("DirectReuse")
residual = parse_row("ResidualReuse")
with open(summary_path, "a", encoding="utf-8") as out:
    out.write(
        "\t".join(
            [
                str(threshold),
                str(bias),
                direct["reuse"],
                direct["drop"],
                direct["acc"],
                direct["avgerr"],
                direct["hiterr"],
                residual["reuse"],
                residual["drop"],
                residual["acc"],
                residual["avgerr"],
                residual["hiterr"],
                residual["alpha"],
                residual["trainpairs"],
                log_path,
            ]
        )
        + "\n"
    )
PY
  done
done

"${PYTHON_BIN}" - "${SUMMARY_TSV}" "${SUMMARY_MD}" "${REPO_DIR}/RESIDUAL_CORRECTED_REUSE.md" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

summary_tsv = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
doc_path = Path(sys.argv[3])

rows = list(csv.DictReader(summary_tsv.open("r", encoding="utf-8"), delimiter="\t"))
rows.sort(key=lambda r: (int(r["confidence_bias"]), int(r["threshold"])))

lines = []
lines.append("## 7.6 PubMed/ST 3/1/1 Confidence-Bias Sweep")
lines.append("")
lines.append("这组实验固定 PubMed/ST、TSER `3/1/1`、`residual_min_dist=1.0`，扫描复用阈值 `T` 和 `score_pair_confidence_discount`。")
lines.append("")
lines.append("这里的 `confidence_bias` 指的是在计算 `reuse_risk = sensitivity_q * reuse_error_q` 之前，对高置信候选的 `reuse_error_q` 做折扣：")
lines.append("")
lines.append("```text")
lines.append("reuse_error_q <- max(1, reuse_error_q - confidence_bias)")
lines.append("```")
lines.append("")
lines.append("因此它不是 oracle error，而是基于 route support / base support / cosine margin 的在线置信修正。")
lines.append("")
lines.append("| Bias | T | Direct Reuse | Direct Drop | Residual Reuse | Residual Drop | Residual HitErr | Alpha | TrainPairs |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        "| {confidence_bias} | {threshold} | {direct_reuse} | {direct_drop} | "
        "{residual_reuse} | {residual_drop} | {residual_hiterr} | {alpha} | {train_pairs} |".format(**row)
    )
lines.append("")
lines.append("结果日志：")
lines.append("")
lines.append("```text")
lines.append(str(summary_tsv))
lines.append(str(summary_md))
lines.append("output/residual_reuse/pubmed_st_bias_sweep/*.log")
lines.append("```")
lines.append("")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

doc = doc_path.read_text(encoding="utf-8")
start = "<!-- PUBMED_ST_RESIDUAL_BIAS_SWEEP_START -->"
end = "<!-- PUBMED_ST_RESIDUAL_BIAS_SWEEP_END -->"
block = start + "\n" + "\n".join(lines) + "\n" + end
if start in doc and end in doc:
    before = doc.split(start, 1)[0].rstrip()
    after = doc.split(end, 1)[1].lstrip()
    updated = before + "\n\n" + block + "\n\n" + after
else:
    marker = "\n## 8. Current Limitations"
    if marker not in doc:
        updated = doc.rstrip() + "\n\n" + block + "\n"
    else:
        updated = doc.replace(marker, "\n\n" + block + "\n" + marker, 1)
doc_path.write_text(updated, encoding="utf-8")

subprocess.run(["git", "add", "RESIDUAL_CORRECTED_REUSE.md", "run_pubmed_st_residual_bias_sweep.sh"], cwd=doc_path.parent, check=True)
diff_cached = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=doc_path.parent)
if diff_cached.returncode != 0:
    subprocess.run(
        ["git", "commit", "-m", "docs: add pubmed residual confidence bias sweep"],
        cwd=doc_path.parent,
        check=True,
    )
subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=doc_path.parent, check=True)
subprocess.run(["git", "push", "origin", "main"], cwd=doc_path.parent, check=True)
PY

echo "[PubMedResidualBias] $(date '+%F %T') finished and pushed docs"
