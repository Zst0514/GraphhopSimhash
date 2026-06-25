#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhangshangtong/Transformer/OFA}"
REPO="$ROOT/GraphhopSimhash"
PY="${PY:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS="${DATASETS:-cora}"
RATIOS="${RATIOS:-0.20}"
ALPHAS="${ALPHAS:-0.25 0.50 1.00 2.00}"
RISK_MODES="${RISK_MODES:-tser degree}"
RUNS="${RUNS:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
AWQ_FORCE_CPU="${AWQ_FORCE_CPU:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/progressive_bfp_boost_tuning}"
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
    if [[ -z "$free_mb" || "$free_mb" -ge "$MIN_FREE_GPU_MB" ]]; then
      log "[GPU ready] free=${free_mb:-unknown}MiB"
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

alpha_suffix() {
  "$PY" - "$1" <<'PY'
import sys
s = f"{float(sys.argv[1]):.2f}".rstrip("0").rstrip(".")
print(s.replace(".", "p"))
PY
}

tasks_for_dataset() {
  case "$1" in
    cora) echo "CN CL" ;;
    pubmed) echo "PN PL" ;;
    arxiv) echo "AR" ;;
    wikics) echo "WK" ;;
    *) echo "Unknown dataset: $1" >&2; return 1 ;;
  esac
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
  awq_device_args=()
  if [[ "$AWQ_FORCE_CPU" == "1" ]]; then
    awq_device_args+=(--awq_force_cpu)
  fi
  "$PY" "$REPO/scripts/generate_graph_aware_bfp_dynamic_pool.py" \
    --dataset "$dataset" \
    --llm_name llama2_7b \
    --block_size 256 \
    --base_mantissa 4 \
    --refine_mantissa 6 \
    --awq_calib_samples 128 \
    --awq_seqlen 512 \
    "${awq_device_args[@]}" \
    --batch_size "$BATCH_SIZE" \
    --runs 1 \
    --seed "$SEED" \
    --save_to_cache \
    --cache_tag "$tag" \
    --output_dir "$GEN_OUT" \
    "$@" >"$log_file" 2>&1
  log "[Pool done] dataset=$dataset tag=$tag log=$log_file"
}

eval_tag_for_dataset() {
  local dataset="$1"
  local tag="$2"
  local tasks
  tasks="$(tasks_for_dataset "$dataset")"
  local out_dir="$EVAL_OUT/$dataset/$tag"
  local log_file="$LOG_DIR/eval_${dataset}_${tag}.log"
  if [[ -f "$out_dir/summary.tsv" && "${FORCE_EVAL:-0}" != "1" ]]; then
    log "[Eval skip] dataset=$dataset tag=$tag"
    return
  fi
  wait_for_gpu
  log "[Eval] dataset=$dataset tasks=$tasks tag=$tag runs=$RUNS"
  "$PY" "$REPO/scripts/profile_dual_granularity_bfp_oracle.py" \
    --tasks $tasks \
    --runs "$RUNS" \
    --seed "$SEED" \
    --dynamic_tag "$tag" \
    --output_dir "$out_dir" >"$log_file" 2>&1
  log "[Eval done] dataset=$dataset tag=$tag log=$log_file"
}

for dataset in $DATASETS; do
  for ratio in $RATIOS; do
    ratio_s="$(ratio_suffix "$ratio")"
    for risk_mode in $RISK_MODES; do
      for alpha in $ALPHAS; do
        alpha_s="$(alpha_suffix "$alpha")"
        tag="W4GraphBFPA4to6_B256_${risk_mode}_stressboost${ratio_s}_a${alpha_s}"
        generate_pool "$dataset" "$tag" \
          --selection_policy stress_graph_boost_topk \
          --risk_mode "$risk_mode" \
          --lift_ratio "$ratio" \
          --risk_boost_alpha "$alpha"
        eval_tag_for_dataset "$dataset" "$tag"
      done
    done
  done
done

log "[All done] output=$OUT_ROOT"
