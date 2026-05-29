#!/usr/bin/env bash
set -euo pipefail

# Build the vendored ONNXim simulator used by the Graph-Bit microbenchmarks.
# The upstream project expects Conan v1 style CMake generators, so this script
# keeps all generated files under ONNXim/build and leaves the source tree clean.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ONNXIM_DIR="${ONNXIM_DIR:-${REPO_DIR}/ONNXim}"
BUILD_DIR="${BUILD_DIR:-${ONNXIM_DIR}/build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

cd "${ONNXIM_DIR}"
mkdir -p "${BUILD_DIR}"

if [[ ! -f "${BUILD_DIR}/conanbuildinfo.cmake" ]]; then
  if ! command -v conan >/dev/null 2>&1; then
    echo "[ONNXimBuild] conan is required but was not found in PATH." >&2
    echo "[ONNXimBuild] Install/use a Conan v1 environment, then rerun this script." >&2
    exit 1
  fi
  echo "[ONNXimBuild] running conan install -> ${BUILD_DIR}"
  conan install . --build=missing -if "${BUILD_DIR}"
fi

echo "[ONNXimBuild] configuring ${ONNXIM_DIR}"
cmake -S "${ONNXIM_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCONAN_DISABLE_CHECK_COMPILER=ON \
  -Dprotobuf_BUILD_TESTS=OFF

echo "[ONNXimBuild] building Simulator with ${JOBS} jobs"
cmake --build "${BUILD_DIR}" --target Simulator -j "${JOBS}"

if [[ ! -x "${BUILD_DIR}/bin/Simulator" ]]; then
  echo "[ONNXimBuild] build completed but ${BUILD_DIR}/bin/Simulator is missing." >&2
  exit 1
fi

echo "[ONNXimBuild] ready: ${BUILD_DIR}/bin/Simulator"
