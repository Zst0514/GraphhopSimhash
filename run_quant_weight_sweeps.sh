#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OFA_ROOT}"

DATASETS="${DATASETS:-pubmed arxiv}"
RUNS="${RUNS:-10}"
SEED="${SEED:-42}"
MODEL="${MODEL:-llama2_7b}"
FP_TAG="${FP_TAG:-FP16}"
INT8_TAG="${INT8_TAG:-W4A8_LLAMA7B_PTQ_TEST}"
INT4_TAG="${INT4_TAG:-W4A4_LLAMA7B_W4A4O_R2}"
INT8_RATIO="${INT8_RATIO:-0.80}"
ERROR_NORM="${ERROR_NORM:-1.0}"
PRESET="${PRESET:-core}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-saved_exp/graphhop_quant_weight_sweeps/${MODEL}_${PRESET}_${STAMP}}"

mkdir -p "${OUT_DIR}"
MASTER_SUMMARY="${OUT_DIR}/final_summaries.txt"
RESULTS_TSV="${OUT_DIR}/results.tsv"
COMMANDS_TXT="${OUT_DIR}/commands.txt"

cat > "${OUT_DIR}/README.txt" <<EOF
GraphHopSimhash fixed-budget TSER weight sweep

Datasets: ${DATASETS}
Runs: ${RUNS}
Seed: ${SEED}
Model: ${MODEL}
FP tag: ${FP_TAG}
W4A8 tag: ${INT8_TAG}
W4A4 tag: ${INT4_TAG}
W4A8 ratio: ${INT8_RATIO}
Preset: ${PRESET}

Each experiment saves:
  - full log:      <dataset>_<weight_name>.log
  - final summary: <dataset>_<weight_name>.summary.txt

Aggregates:
  - ${MASTER_SUMMARY}
  - ${RESULTS_TSV}
  - ${COMMANDS_TXT}
EOF

printf "dataset\tweight_name\tprop_w\tctx_w\tlow_w\tbaseline\tconfig\tw4a4_pct\tw4a8_pct\tcost\tacc\tdrop_pct\tavg_err\n" > "${RESULTS_TSV}"
: > "${MASTER_SUMMARY}"
: > "${COMMANDS_TXT}"

if [[ "${PRESET}" == "core" ]]; then
  WEIGHT_SPECS=(
    "degree_only:3:0:0"
    "context_add:3:1:0"
    "unique_add:3:0:1"
    "tser_light_3_1_1:3:1:1"
    "prop4_degree:4:0:0"
    "prop4_context:4:1:0"
  )
elif [[ "${PRESET}" == "extended" ]]; then
  WEIGHT_SPECS=(
    "degree_only:3:0:0"
    "context_add:3:1:0"
    "unique_add:3:0:1"
    "tser_light_3_1_1:3:1:1"
    "prop4_degree:4:0:0"
    "prop4_context:4:1:0"
    "prop4_unique:4:0:1"
    "prop4_tser:4:1:1"
    "context_heavy:3:2:0"
    "unique_heavy:3:0:2"
    "old_3_2_2:3:2:2"
    "balanced_2_1_1:2:1:1"
  )
else
  echo "[Error] Unknown PRESET=${PRESET}. Use PRESET=core or PRESET=extended." >&2
  exit 1
fi

require_pool() {
  local dataset="$1"
  local tag="$2"
  local path="cache_data/${dataset}_${MODEL}_oracle_${tag}.pt"
  if [[ ! -f "${path}" ]]; then
    echo "[Error] Missing pool: ${path}" >&2
    echo "        Generate it first or override MODEL/FP_TAG/INT8_TAG/INT4_TAG." >&2
    exit 1
  fi
}

extract_summary() {
  local dataset="$1"
  local weight_name="$2"
  local prop_w="$3"
  local ctx_w="$4"
  local low_w="$5"
  local log_path="$6"
  local summary_path="$7"

  awk 'BEGIN{capture=0} /FINAL REAL QUANT SUMMARY/{capture=1} capture{print}' "${log_path}" > "${summary_path}"
  {
    echo
    echo "################################################################################"
    echo "# dataset=${dataset} weight=${weight_name} weights=${prop_w}/${ctx_w}/${low_w}"
    echo "################################################################################"
    cat "${summary_path}"
  } >> "${MASTER_SUMMARY}"

  local baseline
  baseline="$(awk '/^Baseline Acc:/ {print $3; exit}' "${summary_path}")"
  awk \
    -v dataset="${dataset}" \
    -v weight_name="${weight_name}" \
    -v prop_w="${prop_w}" \
    -v ctx_w="${ctx_w}" \
    -v low_w="${low_w}" \
    -v baseline="${baseline}" \
    -F'|' '
      /^(AllW4A8|AllW4A4|RandomBudget|DegreeBudget|TSERBudget|GraphHopSafeBudget)/ {
        for (i = 1; i <= NF; i++) {
          gsub(/^ +| +$/, "", $i)
        }
        print dataset "\t" weight_name "\t" prop_w "\t" ctx_w "\t" low_w "\t" baseline "\t" \
              $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7
      }
    ' "${summary_path}" >> "${RESULTS_TSV}"
}

for dataset in ${DATASETS}; do
  require_pool "${dataset}" "${FP_TAG}"
  require_pool "${dataset}" "${INT8_TAG}"
  require_pool "${dataset}" "${INT4_TAG}"
done

echo "[Sweep] Output directory: ${OUT_DIR}"
echo "[Sweep] Datasets: ${DATASETS}"
echo "[Sweep] Preset: ${PRESET}"
echo "[Sweep] Weight specs: ${WEIGHT_SPECS[*]}"

for dataset in ${DATASETS}; do
  for spec in "${WEIGHT_SPECS[@]}"; do
    IFS=":" read -r weight_name prop_w ctx_w low_w <<< "${spec}"
    log_path="${OUT_DIR}/${dataset}_${weight_name}.log"
    summary_path="${OUT_DIR}/${dataset}_${weight_name}.summary.txt"

    cmd=(
      python -m GraphhopSimhash
      --datasets "${dataset}"
      --runs "${RUNS}"
      --seed "${SEED}"
      --experiment_suite real_quant_ablation
      --real_quant_policy_suite fixed_aggressive_budget
      --real_quant_model_name "${MODEL}"
      --real_quant_fp_tag "${FP_TAG}"
      --real_quant_int8_tag "${INT8_TAG}"
      --real_quant_int4_tag "${INT4_TAG}"
      --real_quant_error_norm "${ERROR_NORM}"
      --real_quant_int8_ratio "${INT8_RATIO}"
      --score_propagation_weight "${prop_w}"
      --score_graph_context_weight "${ctx_w}"
      --score_low_unique_weight "${low_w}"
    )

    {
      echo
      echo "################################################################################"
      echo "# dataset=${dataset} weight=${weight_name} weights=${prop_w}/${ctx_w}/${low_w}"
      printf "%q " "${cmd[@]}"
      echo
    } | tee -a "${COMMANDS_TXT}"

    "${cmd[@]}" 2>&1 | tee "${log_path}"
    extract_summary "${dataset}" "${weight_name}" "${prop_w}" "${ctx_w}" "${low_w}" "${log_path}" "${summary_path}"
  done
done

echo "[Done] Sweep complete."
echo "[Done] Logs: ${OUT_DIR}"
echo "[Done] Final summaries: ${MASTER_SUMMARY}"
echo "[Done] TSV: ${RESULTS_TSV}"
