#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP_NAME=${TOP_NAME:-hamming_threshold_compare16}
CLK_PERIOD_NS=${CLK_PERIOD_NS:-2.0}
LIB_DB=${LIB_DB:-/pdk/tsmc28/logic/db/tcbn28hpcplusbwp40p140lvtssg0p9v125c_ccs.db}

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /opt/synopsys:/opt/synopsys \
  -v /opt/synopsys/TSMC28:/pdk/tsmc28 \
  -v "${SCRIPT_DIR}:/work" \
  -w /work \
  -e TOP_NAME="${TOP_NAME}" \
  -e CLK_PERIOD_NS="${CLK_PERIOD_NS}" \
  -e LIB_DB="${LIB_DB}" \
  -e SNPSLMD_LICENSE_FILE="${SNPSLMD_LICENSE_FILE:-52800@172.17.0.1}" \
  dc_final_env:v1 \
  bash -lc '/opt/synopsys/dc_2018/syn/O-2018.06-SP1/bin/dc_shell -64bit -f run_dc.tcl'
