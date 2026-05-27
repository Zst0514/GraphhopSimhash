#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/llama7b_precision_depth_budget_sweep}"

mkdir -p "$OUT_DIR/pools" "$OUT_DIR/sweeps"

cd "$ROOT_DIR"

DATASETS=(cora pubmed arxiv)
CONFIGS=(W4A6 W4A5)

BUDGET_NAMES=(full p8_10_p6_20 p8_20_p6_30 p8_30_p6_40 p8_40_p6_40 p8_60_p6_20)
BUDGET_HIGH=(0.00 0.10 0.20 0.30 0.40 0.60)
BUDGET_MID=(0.00 0.20 0.30 0.40 0.40 0.20)

TARGETS=(embedding)

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

need_pool() {
  local dataset="$1"
  local tag="$2"
  local path="$ROOT_DIR/cache_data/${dataset}_llama2_7b_oracle_${tag}.pt"
  [[ ! -s "$path" ]]
}

generate_pool() {
  local dataset="$1"
  local tag="$2"
  local log="$OUT_DIR/pools/${dataset}_llama2_7b_${tag}.log"
  if ! need_pool "$dataset" "$tag"; then
    echo "[$(timestamp)] [SkipPool] ${dataset} ${tag} already exists"
    return
  fi

  echo "[$(timestamp)] [Pool] Generating ${dataset} llama2_7b ${tag}"
  "$PYTHON_BIN" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets "$dataset" \
    --llm_name llama2_7b \
    --configs "$tag" \
    --batch_size 4 \
    --awq_calib_samples 128 \
    --awq_seqlen 512 \
    --overwrite 2>&1 | tee "$log"
}

check_required_pools() {
  local dataset="$1"
  local missing=0
  for tag in W4A8 W4A6 W4A5 W4A4; do
    local path="$ROOT_DIR/cache_data/${dataset}_llama2_7b_oracle_${tag}.pt"
    if [[ ! -s "$path" ]]; then
      echo "[$(timestamp)] [MissingPool] $path"
      missing=1
    fi
  done
  return "$missing"
}

run_sweep() {
  local dataset="$1"
  local target="$2"
  local budget_name="$3"
  local high="$4"
  local mid="$5"
  local log="$OUT_DIR/sweeps/${dataset}_${target}_${budget_name}.log"
  local done_file="${log}.done"

  if [[ -s "$done_file" ]]; then
    echo "[$(timestamp)] [SkipSweep] ${dataset} ${target} ${budget_name}"
    return
  fi

  echo "[$(timestamp)] [Sweep] dataset=${dataset} target=${target} high=${high} mid=${mid}"
  "$PYTHON_BIN" -m GraphhopSimhash \
    --datasets "$dataset" \
    --runs 10 \
    --experiment_suite precision_depth_ablation \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_tags W4A6 W4A5 W4A4 \
    --precision_depth_bits 6 5 4 \
    --precision_depth_reference_bits 8 \
    --precision_depth_high_ratio "$high" \
    --precision_depth_mid_ratio "$mid" \
    --precision_depth_cost_scale 0.50 \
    --precision_depth_fixed_cost 0.15 \
    --precision_depth_predictor_calib_samples 512 \
    --precision_depth_predictor_target "$target" 2>&1 | tee "$log"
  touch "$done_file"
}

summarize_results() {
  "$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
sweep_dir = out_dir / "sweeps"
summary_path = out_dir / "summary.tsv"

row_re = re.compile(
    r"^(?P<config>[A-Za-z0-9_]+)\s+\|\s+"
    r"(?P<p8>[-0-9.]+)%\s+\|\s+"
    r"(?P<p6>[-0-9.]+)%\s+\|\s+"
    r"(?P<p5>[-0-9.]+)%\s+\|\s+"
    r"(?P<p4>[-0-9.]+)%\s+\|\s+"
    r"(?P<cost>[-0-9.]+)\s+\|\s+"
    r"(?P<acc>[-0-9.]+)\s+\|\s+"
    r"(?P<drop>[-0-9.]+)%\s+\|\s+"
    r"(?P<avgerr>[-0-9.]+)"
)
base_re = re.compile(r"Baseline Acc:\s+([-0-9.]+)")

rows = []
for log_path in sorted(sweep_dir.glob("*.log")):
    stem = log_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        continue
    dataset = parts[0]
    target = parts[1]
    budget = "_".join(parts[2:])
    text = log_path.read_text(errors="replace")
    baseline = ""
    m = base_re.search(text)
    if m:
        baseline = m.group(1)
    for line in text.splitlines():
        m = row_re.match(line.rstrip())
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "dataset": dataset,
            "target": target,
            "budget": budget,
            "config": d["config"],
            "P8%": d["p8"],
            "P6%": d["p6"],
            "P5%": d["p5"],
            "P4%": d["p4"],
            "cost": d["cost"],
            "acc": d["acc"],
            "drop": d["drop"],
            "avgerr": d["avgerr"],
            "baseline": baseline,
            "log": str(log_path),
        })

fieldnames = [
    "dataset", "target", "budget", "config",
    "P8%", "P6%", "P5%", "P4%", "cost", "acc", "drop", "avgerr", "baseline", "log",
]
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fnum(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")

for dataset in sorted({r["dataset"] for r in rows}):
    for target in sorted({r["target"] for r in rows if r["dataset"] == dataset}):
        subset = [r for r in rows if r["dataset"] == dataset and r["target"] == target]
        compact_path = out_dir / f"compact_{dataset}_{target}.txt"
        interesting = [
            r for r in subset
            if r["config"] in {
                "FullP8", "AllP6", "AllP5", "AllP4",
                "RandomDepthBudget", "DegreeDepthBudget", "TSERDepthBudget",
                "ContextDepthBudget", "LowUniqueDepthBudget", "PredictorDepthBudget",
                "OracleDamageBudget",
            }
        ]
        interesting.sort(key=lambda r: (fnum(r, "cost"), fnum(r, "drop"), r["config"]))
        with compact_path.open("w") as f:
            f.write(f"{dataset.upper()} / LLaMA-7B precision-depth sweep ({target})\n")
            f.write(f"Source: {sweep_dir}\n\n")
            f.write(f"{'budget':<16} {'config':<22} {'P8':>6} {'P6':>6} {'P5':>6} {'P4':>6} {'cost':>7} {'drop':>7} {'acc':>8} {'avgerr':>9}\n")
            f.write("-" * 106 + "\n")
            for r in interesting:
                f.write(
                    f"{r['budget']:<16} {r['config']:<22} "
                    f"{float(r['P8%']):>5.1f}% {float(r['P6%']):>5.1f}% {float(r['P5%']):>5.1f}% {float(r['P4%']):>5.1f}% "
                    f"{float(r['cost']):>7.3f} {float(r['drop']):>6.2f}% {float(r['acc']):>8.4f} {float(r['avgerr']):>9.5f}\n"
                )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset in sorted({r["dataset"] for r in rows}):
        for target in sorted({r["target"] for r in rows if r["dataset"] == dataset}):
            subset = [r for r in rows if r["dataset"] == dataset and r["target"] == target]
            configs = [
                "FullP8", "AllP6", "AllP5", "AllP4",
                "RandomDepthBudget", "DegreeDepthBudget", "TSERDepthBudget",
                "ContextDepthBudget", "LowUniqueDepthBudget", "PredictorDepthBudget",
                "OracleDamageBudget",
            ]
            plt.figure(figsize=(9, 5))
            for config in configs:
                pts = [r for r in subset if r["config"] == config]
                if not pts:
                    continue
                pts.sort(key=lambda r: fnum(r, "cost"))
                plt.plot(
                    [fnum(r, "cost") for r in pts],
                    [fnum(r, "drop") for r in pts],
                    marker="o",
                    label=config,
                )
            plt.xlabel("Cost")
            plt.ylabel("Drop (%)")
            plt.title(f"{dataset.upper()} / LLaMA-7B precision-depth cost-drop ({target})")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"curve_{dataset}_{target}.png", dpi=180)
            plt.close()
except Exception as exc:
    (out_dir / "plot_error.txt").write_text(str(exc))

print(f"[Summary] wrote {summary_path}")
PY
}

echo "[$(timestamp)] [Start] LLaMA-7B W4A6/W4A5 pool generation + precision-depth sweeps"

for dataset in "${DATASETS[@]}"; do
  for tag in "${CONFIGS[@]}"; do
    generate_pool "$dataset" "$tag"
  done

  if ! check_required_pools "$dataset"; then
    echo "[$(timestamp)] [Error] Missing required pools for ${dataset}; skip sweeps"
    continue
  fi

  for target in "${TARGETS[@]}"; do
    for i in "${!BUDGET_NAMES[@]}"; do
      run_sweep "$dataset" "$target" "${BUDGET_NAMES[$i]}" "${BUDGET_HIGH[$i]}" "${BUDGET_MID[$i]}"
      summarize_results
    done
  done
done

summarize_results
echo "[$(timestamp)] [Done] Results in $OUT_DIR"
