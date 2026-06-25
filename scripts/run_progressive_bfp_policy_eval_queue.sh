#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhangshangtong/Transformer/OFA}"
REPO="$ROOT/GraphhopSimhash"
PY="${PY:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS="${DATASETS:-cora pubmed wikics arxiv}"
TASKS="${TASKS:-CN CL PN PL AR WK}"
RATIOS="${RATIOS:-0.10 0.15 0.20 0.25 0.30}"
RUNS="${RUNS:-10}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/progressive_bfp_policy_eval}"
GEN_OUT="$OUT_ROOT/pool_generation"
EVAL_OUT="$OUT_ROOT/eval"
LOG_DIR="$OUT_ROOT/logs"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-120}"

mkdir -p "$GEN_OUT" "$EVAL_OUT" "$LOG_DIR"
cd "$ROOT"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

wait_for_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  while true; do
    free_mb="$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
    )"
    if [[ -z "$free_mb" ]]; then
      return
    fi
    if (( free_mb >= MIN_FREE_GPU_MB )); then
      log "[GPU ready] free=${free_mb}MiB"
      return
    fi
    log "[GPU wait] free=${free_mb}MiB < ${MIN_FREE_GPU_MB}MiB; sleeping ${GPU_WAIT_SEC}s"
    sleep "$GPU_WAIT_SEC" || true
  done
}

ratio_suffix() {
  "$PY" - "$1" <<'PY'
import sys
print(int(round(float(sys.argv[1]) * 100)))
PY
}

pool_path() {
  local dataset="$1"
  local tag="$2"
  printf '%s/cache_data/%s_llama2_7b_oracle_%s.pt' "$ROOT" "$dataset" "$tag"
}

generate_pool() {
  local dataset="$1"
  local tag="$2"
  shift 2
  local log_file="$LOG_DIR/generate_${dataset}_${tag}.log"
  if [[ -f "$(pool_path "$dataset" "$tag")" && "${FORCE_GEN:-0}" != "1" ]]; then
    log "[Pool skip] dataset=$dataset tag=$tag"
    return
  fi
  wait_for_gpu
  log "[Pool gen] dataset=$dataset tag=$tag"
  "$PY" "$REPO/scripts/generate_graph_aware_bfp_dynamic_pool.py" \
    --dataset "$dataset" \
    --llm_name llama2_7b \
    --block_size 256 \
    --base_mantissa 4 \
    --refine_mantissa 6 \
    --awq_calib_samples 128 \
    --awq_seqlen 512 \
    --batch_size "${BATCH_SIZE:-4}" \
    --runs 1 \
    --seed "$SEED" \
    --save_to_cache \
    --cache_tag "$tag" \
    --output_dir "$GEN_OUT" \
    "$@" >"$log_file" 2>&1
  log "[Pool done] dataset=$dataset tag=$tag log=$log_file"
}

eval_tag() {
  local tag="$1"
  local out_dir="$EVAL_OUT/$tag"
  local log_file="$LOG_DIR/eval_${tag}.log"
  if [[ -f "$out_dir/summary.tsv" && "${FORCE_EVAL:-0}" != "1" ]]; then
    log "[Eval skip] tag=$tag"
    return
  fi
  wait_for_gpu
  log "[Eval] tag=$tag tasks=$TASKS runs=$RUNS"
  "$PY" "$REPO/scripts/profile_dual_granularity_bfp_oracle.py" \
    --tasks $TASKS \
    --runs "$RUNS" \
    --seed "$SEED" \
    --dynamic_tag "$tag" \
    --output_dir "$out_dir" >"$log_file" 2>&1
  log "[Eval done] tag=$tag log=$log_file"
}

TAGS=()

# Existing Motivation baselines.  These are skipped if pools already exist.
BASELINE_SPECS=(
  "W4BlockBFPA4to6_B256_random20|--selection_policy random --lift_ratio 0.20"
  "W4BlockBFPA4to6_B256_oracle20|--selection_policy oracle_error --lift_ratio 0.20"
)

for spec in "${BASELINE_SPECS[@]}"; do
  tag="${spec%%|*}"
  args="${spec#*|}"
  TAGS+=("$tag")
  for dataset in $DATASETS; do
    # shellcheck disable=SC2086
    generate_pool "$dataset" "$tag" $args
  done
done

# Selector baselines and proposed Graph x Stress at matched lifted-block ratios.
for ratio in $RATIOS; do
  suffix="$(ratio_suffix "$ratio")"

  stress_tag="W4BlockBFPA4to6_B256_stress${suffix}"
  TAGS+=("$stress_tag")
  for dataset in $DATASETS; do
    generate_pool "$dataset" "$stress_tag" \
      --selection_policy stress \
      --lift_ratio "$ratio"
  done

  graph_tag="W4GraphBFPA4to6_B256_tser_graph${suffix}"
  TAGS+=("$graph_tag")
  for dataset in $DATASETS; do
    generate_pool "$dataset" "$graph_tag" \
      --selection_policy graph_risk \
      --risk_mode tser \
      --lift_ratio "$ratio"
  done

  graphstress_tag="W4GraphBFPA4to6_B256_tser_graphstress${suffix}"
  TAGS+=("$graphstress_tag")
  for dataset in $DATASETS; do
    generate_pool "$dataset" "$graphstress_tag" \
      --selection_policy graph_stress_topk \
      --risk_mode tser \
      --lift_ratio "$ratio"
  done
done

# Existing threshold-style dynamic pools are useful for a policy sensitivity row
# when present; the eval step will fail loudly if a requested dataset is missing.
if [[ "${EVAL_EXISTING_THRESHOLD:-1}" == "1" ]]; then
  TAGS+=("W4GraphBFPA4to6_B256_tser_top25_t0.2")
fi

for tag in "${TAGS[@]}"; do
  eval_tag "$tag"
done

"$PY" "$REPO/scripts/summarize_progressive_bfp_policy_eval.py" \
  --eval_dir "$EVAL_OUT" \
  --output "$OUT_ROOT/policy_summary.tsv" >"$LOG_DIR/summarize.log" 2>&1

log "[All done] output=$OUT_ROOT"
