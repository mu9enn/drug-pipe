#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_gpu.template.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_DIR="${ROOT_DIR}/reports"
mkdir -p "${REPORT_DIR}"

if [[ -z "${MOLCLAW_SCP_API_KEY:-}" ]]; then
  echo "[ERROR] MOLCLAW_SCP_API_KEY is empty." >&2
  exit 1
fi

if [[ ",${NO_PROXY}," == *",180.184.86.2,"* ]] || [[ ",${no_proxy}," == *",180.184.86.2,"* ]]; then
  echo "[ERROR] NO_PROXY/no_proxy must NOT include 180.184.86.2" >&2
  exit 1
fi

echo "[INFO] gpu suite started"
echo "[INFO] server_url=${MOLCLAW_SCP_SERVER_URL}"
echo "[INFO] proxy_url=${MOLCLAW_PROXY_URL}"

GPU_NOTEBOOK_JSON="${REPORT_DIR}/report_notebook_gpu.json"
GPU_NOTEBOOK_MD="${REPORT_DIR}/report_notebook_gpu.md"
GPU_EXTRA_JSON="${REPORT_DIR}/report_extra_gpu.json"
GPU_EXTRA_MD="${REPORT_DIR}/report_extra_gpu.md"
GPU_REMAINING_JSON="${REPORT_DIR}/report_remaining_gpu.json"
GPU_REMAINING_MD="${REPORT_DIR}/report_remaining_gpu.md"
RUN_REPORT_GPU_JSON="${REPORT_DIR}/run_report_gpu.json"
RUN_REPORT_GPU_MD="${REPORT_DIR}/run_report_gpu.md"
SUMMARY_DIFF_MD="${REPORT_DIR}/summary_diff.md"
COVERAGE_JSON="${REPORT_DIR}/coverage_report.json"
RUN_REPORT_NO_GPU_JSON="${REPORT_DIR}/run_report_no_gpu.json"
REMAINING_CASES_JSON="${ROOT_DIR}/cases/remaining_tools_auto.json"

echo "[INFO] Connectivity check (initialize via relay proxy)"
curl -sS --max-time 30 \
  -x "${MOLCLAW_PROXY_URL}" \
  -X POST "${MOLCLAW_SCP_SERVER_URL}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "SCP-HUB-API-KEY: ${MOLCLAW_SCP_API_KEY}" \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"gpu-suite","version":"0.1.0"}}}' >/dev/null

echo "[INFO] Run notebook mapped cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/run_molclaw_cases.py" \
  --server-url "${MOLCLAW_SCP_SERVER_URL}" \
  --proxy-url "${MOLCLAW_PROXY_URL}" \
  --worker-mode "gpu" \
  --timeout-sec "${TIMEOUT_SEC:-60}" \
  --case-file "${ROOT_DIR}/cases/notebook_cases_schema_mapped.json" \
  --out-json "${GPU_NOTEBOOK_JSON}" \
  --out-md "${GPU_NOTEBOOK_MD}"

echo "[INFO] Run extra light cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/run_molclaw_cases.py" \
  --server-url "${MOLCLAW_SCP_SERVER_URL}" \
  --proxy-url "${MOLCLAW_PROXY_URL}" \
  --worker-mode "gpu" \
  --timeout-sec "${TIMEOUT_SEC:-60}" \
  --case-file "${ROOT_DIR}/cases/extra_light_cases.json" \
  --out-json "${GPU_EXTRA_JSON}" \
  --out-md "${GPU_EXTRA_MD}"

echo "[INFO] Build remaining-tool cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/build_remaining_cases.py" \
  --schema-json "${ROOT_DIR}/notebook/drugsda_tools_schema.json" \
  --existing-case-files "${ROOT_DIR}/cases/notebook_cases_schema_mapped.json" "${ROOT_DIR}/cases/extra_light_cases.json" \
  --out-case-file "${REMAINING_CASES_JSON}"

echo "[INFO] Run remaining-tool auto cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/run_molclaw_cases.py" \
  --server-url "${MOLCLAW_SCP_SERVER_URL}" \
  --proxy-url "${MOLCLAW_PROXY_URL}" \
  --worker-mode "gpu" \
  --timeout-sec "${TIMEOUT_SEC:-60}" \
  --case-file "${REMAINING_CASES_JSON}" \
  --out-json "${GPU_REMAINING_JSON}" \
  --out-md "${GPU_REMAINING_MD}"

echo "[INFO] Merge gpu reports"
"${PYTHON_BIN}" "${ROOT_DIR}/common/merge_reports.py" \
  --inputs "${GPU_NOTEBOOK_JSON}" "${GPU_EXTRA_JSON}" "${GPU_REMAINING_JSON}" \
  --out-json "${RUN_REPORT_GPU_JSON}" \
  --out-md "${RUN_REPORT_GPU_MD}" \
  --worker-mode "gpu"

if [[ ! -f "${COVERAGE_JSON}" ]]; then
  echo "[WARN] ${COVERAGE_JSON} missing; generate coverage first on no-gpu suite"
fi
if [[ ! -f "${RUN_REPORT_NO_GPU_JSON}" ]]; then
  echo "[WARN] ${RUN_REPORT_NO_GPU_JSON} missing; comparison will be partial"
fi

echo "[INFO] Update summary_diff.md"
"${PYTHON_BIN}" "${ROOT_DIR}/common/compare_reports.py" \
  --coverage-report "${COVERAGE_JSON}" \
  --no-gpu-report "${RUN_REPORT_NO_GPU_JSON}" \
  --gpu-report "${RUN_REPORT_GPU_JSON}" \
  --out-md "${SUMMARY_DIFF_MD}"

echo "[DONE] gpu suite completed"
echo "[DONE] outputs:"
echo "  - ${GPU_REMAINING_JSON}"
echo "  - ${GPU_REMAINING_MD}"
echo "  - ${RUN_REPORT_GPU_JSON}"
echo "  - ${RUN_REPORT_GPU_MD}"
echo "  - ${SUMMARY_DIFF_MD}"
