#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zhangshangtong/Transformer/OFA"
PYTHON_BIN="/home/zhangshangtong/.conda/envs/OFA/bin/python"
OUT_DIR="${ROOT_DIR}/output/residual_reuse/pubmed_st_diagnosis"
SUMMARY_TSV="${OUT_DIR}/summary.tsv"
SUMMARY_TXT="${OUT_DIR}/summary_aligned.txt"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

RUNS="${RUNS:-3}"
EPOCHS="${EPOCHS:-100}"

BASE_ARGS=(
  -m GraphhopSimhash
  --datasets pubmed
  --runs "${RUNS}"
  --experiment_suite residual_reuse
  --radius 2
  --hash_heads_per_route 4
  --main_hash_head_bits 16 16 16 16
  --learned_hash_epochs 10
  --learned_hash_dim 128
  --hamming_only_acceptor
  --enable_score_gate
  --allow_rare_fuzzy
  --score_propagation_weight 3
  --score_graph_context_weight 1
  --score_low_unique_weight 1
  --residual_rank 32
  --residual_epochs "${EPOCHS}"
  --residual_max_train_pairs 1024
  --residual_min_dist 1.0
)

printf "case\tpurpose\tT\tbias\tsupport_discount\ttrain_split\tmax_pairs\tmin_route_hits\tmin_base_hits\tdirect_reuse\tdirect_drop\tdirect_acc\tdirect_avgerr\tdirect_hiterr\tresidual_reuse\tresidual_drop\tresidual_acc\tresidual_avgerr\tresidual_hiterr\talpha\ttrain_pairs\tlog\n" > "${SUMMARY_TSV}"

run_case() {
  local case_name="$1"
  local purpose="$2"
  local threshold="$3"
  local bias="$4"
  local support_discount="$5"
  local train_split="$6"
  local max_pairs="$7"
  local min_route_hits="$8"
  local min_base_hits="$9"
  shift 9

  local log="${OUT_DIR}/${case_name}.log"
  echo "[Diagnosis] $(date '+%F %T') running ${case_name}: ${purpose}"

  local cmd=("${PYTHON_BIN}" "${BASE_ARGS[@]}"
    --score_reuse_threshold "${threshold}"
    --score_pair_confidence_discount "${bias}"
    --residual_train_split "${train_split}"
    --residual_max_train_pairs "${max_pairs}"
    --residual_min_route_hits "${min_route_hits}"
    --residual_min_base_hits "${min_base_hits}"
  )
  if [[ "${support_discount}" == "off" ]]; then
    cmd+=(--disable_score_support_discount)
  fi
  cmd+=("$@")
  "${cmd[@]}" > "${log}" 2>&1

  "${PYTHON_BIN}" - "${log}" "${SUMMARY_TSV}" "${case_name}" "${purpose}" \
    "${threshold}" "${bias}" "${support_discount}" "${train_split}" \
    "${max_pairs}" "${min_route_hits}" "${min_base_hits}" <<'PY'
import re
import sys

(
    log_path,
    summary_path,
    case_name,
    purpose,
    threshold,
    bias,
    support_discount,
    train_split,
    max_pairs,
    min_route_hits,
    min_base_hits,
) = sys.argv[1:12]
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
                case_name,
                purpose,
                threshold,
                bias,
                support_discount,
                train_split,
                max_pairs,
                min_route_hits,
                min_base_hits,
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
}

# 1/2. Lower thresholds: check whether PubMed needs a stricter reuse gate than T=45/60.
for threshold in 12 15 18 20 25; do
  run_case "lowT_T${threshold}" "low-threshold sweep" "${threshold}" 1 on train_val 1024 1 1
done

# 3. Disable support discount so confidence_bias is not saturated by the route-hit discount.
for bias in 0 1 2; do
  run_case "nosupport_T20_bias${bias}" "confidence bias without support discount" 20 "${bias}" off train_val 1024 1 1
done

# 4. More calibration pairs / wider train split.
run_case "morepairs_T20_trainval2048" "more train-val residual pairs" 20 1 on train_val 2048 1 1
run_case "morepairs_T20_allhits2048" "all-hit residual calibration" 20 1 on all_hits 2048 1 1

# 5. Correct only high-support fuzzy hits.
run_case "support_T20_r2_b2" "residual only high-support hits" 20 1 on train_val 1024 2 2
run_case "support_T20_r3_b2" "residual only very-high-support hits" 20 1 on train_val 1024 3 2
run_case "support_T25_r2_b2" "higher T with high-support correction" 25 1 on train_val 1024 2 2

"${PYTHON_BIN}" - "${SUMMARY_TSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], "r", encoding="utf-8"), delimiter="\t"))
cols = [
    ("case", 28),
    ("purpose", 36),
    ("T", 4),
    ("bias", 4),
    ("support_discount", 8),
    ("train_split", 10),
    ("min_route_hits", 5),
    ("min_base_hits", 5),
    ("direct_reuse", 8),
    ("direct_drop", 8),
    ("residual_drop", 8),
    ("residual_hiterr", 10),
    ("alpha", 6),
    ("train_pairs", 10),
]
lines = []
lines.append("PubMed/ST Residual Reuse Diagnosis")
lines.append(f"Source: {sys.argv[1]}")
lines.append("")
header = " ".join(name[:width].ljust(width) for name, width in cols)
lines.append(header)
lines.append(" ".join("-" * width for _, width in cols))
for row in rows:
    lines.append(" ".join(str(row.get(name, ""))[:width].ljust(width) for name, width in cols))
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "[Diagnosis] $(date '+%F %T') finished"
echo "[Diagnosis] TSV: ${SUMMARY_TSV}"
echo "[Diagnosis] aligned: ${SUMMARY_TXT}"
