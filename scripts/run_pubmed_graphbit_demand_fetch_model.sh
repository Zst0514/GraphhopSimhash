#!/usr/bin/env bash
set -euo pipefail

# Lightweight PubMed Graph-Bit hardware replay.
# This does not rerun the expensive PubMed GNN/residual experiment. It reuses
# an existing predictor_free_workload.json and applies the demand-fetch model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"

FRONTEND="${FRONTEND:-pubmed_h8_76_T40}"
WORKLOAD="${WORKLOAD:-${OFA_DIR}/output/graphbit_predictor_free/${FRONTEND}/predictor_free_workload.json}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/graphbit_predictor_free/${FRONTEND}/demand_fetch_model}"

NODE_COUNT=19717 \
WORKLOAD="${WORKLOAD}" \
OUT_DIR="${OUT_DIR}" \
bash "${SCRIPT_DIR}/run_graphbit_demand_fetch_model.sh"
