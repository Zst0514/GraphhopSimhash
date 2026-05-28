#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhangshangtong/Transformer/OFA}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output/residual_reuse/pubmed_hamming_support_sweep}"
RUNS="${RUNS:-1}"
SEED="${SEED:-42}"

# Stage-1 wide sweep around the previously promising PubMed/ST region.
# Radius is the maximum accepted Hamming distance in CAM lookup.
RADIUS_VALUES=(${RADIUS_VALUES:-1 2 3})
THRESHOLDS=(${THRESHOLDS:-35 38 40})
SPLITS=(${SPLITS:-"5:4" "6:5" "5:3" "4:3"})

mkdir -p "${OUT_DIR}/logs"
cd "${ROOT_DIR}"

echo "[Sweep] output=${OUT_DIR}"
echo "[Sweep] runs=${RUNS} seed=${SEED}"
echo "[Sweep] radius=${RADIUS_VALUES[*]} thresholds=${THRESHOLDS[*]} splits=${SPLITS[*]}"

for radius in "${RADIUS_VALUES[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    for split in "${SPLITS[@]}"; do
      hard="${split%%:*}"
      soft="${split##*:}"
      log="${OUT_DIR}/logs/pubmed_r${radius}_t${threshold}_h${hard}_s${soft}_runs${RUNS}.log"
      echo
      echo "================================================================"
      echo "[Run] radius=${radius} T=${threshold} hard>=${hard} soft>=${soft}"
      echo "[Log] ${log}"
      echo "================================================================"
      "${PYTHON_BIN}" -m GraphhopSimhash \
        --datasets pubmed \
        --runs "${RUNS}" \
        --seed "${SEED}" \
        --experiment_suite residual_reuse \
        --learned_hash_epochs 10 \
        --learned_hash_dim 128 \
        --hamming_only_acceptor \
        --enable_score_gate \
        --score_reuse_threshold "${threshold}" \
        --radius "${radius}" \
        --main_hash_head_bits 16 16 16 16 16 16 16 16 \
        --residual_embedding_source data_x \
        --residual_fit_profile st \
        --residual_hard_min_support_hits "${hard}" \
        --residual_soft_min_support_hits "${soft}" \
        --residual_rank 64 \
        --residual_epochs 200 \
        --residual_max_train_pairs 2048 \
        --residual_min_dist 1.0 \
        --residual_alpha_grid 0 0.0625 0.125 0.25 \
        2>&1 | tee "${log}"
      touch "${log}.done"
    done
  done
done

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
logs = sorted((out_dir / "logs").glob("pubmed_r*_t*_h*_s*_runs*.log"))

row_re = re.compile(
    r"^(DirectReuse|SoftDirectReuse|ResidualReuse)\s+\|\s+"
    r"([0-9.]+%)\s+\|\s+([^|]+)\|\s+([0-9.]+)\s+\|\s+([0-9.]+%)\s+\|\s+"
    r"([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([^|]+)\|"
)
name_re = re.compile(r"pubmed_r(?P<radius>\d+)_t(?P<t>\d+)_h(?P<h>\d+)_s(?P<s>\d+)_runs(?P<runs>\d+)\.log$")

rows = []
for log in logs:
    m = name_re.search(log.name)
    if not m:
        continue
    info = m.groupdict()
    parsed = {}
    dist_hist = ""
    baseline = ""
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("Baseline Acc:"):
            baseline = line.split(":", 1)[1].strip()
        if line.strip().startswith("DistHist:"):
            dist_hist = line.strip().replace("\t", " ")
        rm = row_re.match(line)
        if rm:
            cfg, reuse, train, acc, drop, avgerr, hiterr, alpha = rm.groups()
            parsed[cfg] = {
                "reuse": reuse.strip(),
                "train": train.strip(),
                "acc": acc.strip(),
                "drop": drop.strip(),
                "avgerr": avgerr.strip(),
                "hiterr": hiterr.strip(),
                "alpha": alpha.strip(),
            }
    if "ResidualReuse" not in parsed:
        continue
    soft = parsed.get("SoftDirectReuse", {})
    residual = parsed["ResidualReuse"]
    direct = parsed.get("DirectReuse", {})
    def pct_to_float(x):
        try:
            return float(str(x).strip().rstrip("%"))
        except Exception:
            return float("nan")
    gain = pct_to_float(soft.get("drop", "nan")) - pct_to_float(residual.get("drop", "nan"))
    rows.append({
        **info,
        "baseline": baseline,
        "direct_reuse": direct.get("reuse", ""),
        "direct_drop": direct.get("drop", ""),
        "soft_reuse": soft.get("reuse", ""),
        "soft_drop": soft.get("drop", ""),
        "residual_reuse": residual.get("reuse", ""),
        "residual_drop": residual.get("drop", ""),
        "gain_vs_soft": f"{gain:.2f}%",
        "alpha": residual.get("alpha", ""),
        "train_pairs": residual.get("train", ""),
        "hiterr": residual.get("hiterr", ""),
        "dist_hist": dist_hist,
    })

rows.sort(key=lambda r: (float(r["residual_drop"].rstrip("%")), -float(r["residual_reuse"].rstrip("%"))))

summary = out_dir / "summary.tsv"
with summary.open("w", encoding="utf-8") as f:
    fields = [
        "radius", "T", "hard", "soft", "runs", "baseline",
        "direct_reuse", "direct_drop",
        "soft_reuse", "soft_drop",
        "residual_reuse", "residual_drop", "gain_vs_soft",
        "alpha", "train_pairs", "hiterr", "dist_hist",
    ]
    f.write("\t".join(fields) + "\n")
    for r in rows:
        f.write("\t".join([
            r["radius"], r["t"], r["h"], r["s"], r["runs"], r["baseline"],
            r["direct_reuse"], r["direct_drop"],
            r["soft_reuse"], r["soft_drop"],
            r["residual_reuse"], r["residual_drop"], r["gain_vs_soft"],
            r["alpha"], r["train_pairs"], r["hiterr"], r["dist_hist"],
        ]) + "\n")

compact = out_dir / "summary_compact.txt"
with compact.open("w", encoding="utf-8") as f:
    f.write("PubMed/ST residual reuse Hamming/support sweep\n")
    f.write("Sorted by ResidualReuse drop, then higher reuse.\n\n")
    f.write(f"{'r':>2} {'T':>3} {'h/s':>5} {'Direct':>8} {'Ddrop':>7} {'Soft':>8} {'Sdrop':>7} {'Resid':>8} {'Rdrop':>7} {'Gain':>7} {'Alpha':>8} {'Train':>7}\n")
    f.write("-" * 94 + "\n")
    for r in rows:
        f.write(
            f"{r['radius']:>2} {r['t']:>3} {r['h']+'/'+r['s']:>5} "
            f"{r['direct_reuse']:>8} {r['direct_drop']:>7} "
            f"{r['soft_reuse']:>8} {r['soft_drop']:>7} "
            f"{r['residual_reuse']:>8} {r['residual_drop']:>7} "
            f"{r['gain_vs_soft']:>7} {r['alpha']:>8} {r['train_pairs']:>7}\n"
        )
print(f"[Summary] wrote {summary}")
print(f"[Summary] wrote {compact}")
PY

echo "[Done] ${OUT_DIR}"
