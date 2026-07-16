#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_no_gpu.template.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_DIR="${ROOT_DIR}/reports"
mkdir -p "${REPORT_DIR}"

if [[ -z "${MOLCLAW_SCP_API_KEY:-}" ]]; then
  echo "[ERROR] MOLCLAW_SCP_API_KEY is empty." >&2
  exit 1
fi

echo "[INFO] no-gpu suite started"
echo "[INFO] server_url=${MOLCLAW_SCP_SERVER_URL}"
echo "[INFO] proxy_url=${MOLCLAW_PROXY_URL}"

COVERAGE_JSON="${REPORT_DIR}/coverage_report.json"
COVERAGE_MD="${REPORT_DIR}/coverage_report.md"

NO_GPU_NOTEBOOK_JSON="${REPORT_DIR}/report_notebook_no_gpu.json"
NO_GPU_NOTEBOOK_MD="${REPORT_DIR}/report_notebook_no_gpu.md"
NO_GPU_EXTRA_JSON="${REPORT_DIR}/report_extra_no_gpu.json"
NO_GPU_EXTRA_MD="${REPORT_DIR}/report_extra_no_gpu.md"
NO_GPU_REMAINING_JSON="${REPORT_DIR}/report_remaining_no_gpu.json"
NO_GPU_REMAINING_MD="${REPORT_DIR}/report_remaining_no_gpu.md"
RUN_REPORT_NO_GPU_JSON="${REPORT_DIR}/run_report_no_gpu.json"
RUN_REPORT_NO_GPU_MD="${REPORT_DIR}/run_report_no_gpu.md"
SUMMARY_DIFF_MD="${REPORT_DIR}/summary_diff.md"
GPU_REPORT_JSON="${REPORT_DIR}/run_report_gpu.json"
REMAINING_CASES_JSON="${ROOT_DIR}/cases/remaining_tools_auto.json"

echo "[INFO] Step 0: coverage audit"
"${PYTHON_BIN}" "${ROOT_DIR}/notebook/audit_notebook_coverage.py" \
  --notebook-dir "${ROOT_DIR}/notebook" \
  --out-json "${COVERAGE_JSON}" \
  --out-md "${COVERAGE_MD}"

echo "[INFO] Step 1: connectivity check (initialize through no-gpu proxy)"
curl -sS --max-time 30 \
  -x "${MOLCLAW_PROXY_URL}" \
  -X POST "${MOLCLAW_SCP_SERVER_URL}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "SCP-HUB-API-KEY: ${MOLCLAW_SCP_API_KEY}" \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"no-gpu-suite","version":"0.1.0"}}}' >/dev/null

echo "[INFO] Step 1: run notebook mapped cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/run_molclaw_cases.py" \
  --server-url "${MOLCLAW_SCP_SERVER_URL}" \
  --proxy-url "${MOLCLAW_PROXY_URL}" \
  --worker-mode "no-gpu" \
  --timeout-sec "${TIMEOUT_SEC:-60}" \
  --case-file "${ROOT_DIR}/cases/notebook_cases_schema_mapped.json" \
  --out-json "${NO_GPU_NOTEBOOK_JSON}" \
  --out-md "${NO_GPU_NOTEBOOK_MD}"

echo "[INFO] Step 1: run extra light cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/run_molclaw_cases.py" \
  --server-url "${MOLCLAW_SCP_SERVER_URL}" \
  --proxy-url "${MOLCLAW_PROXY_URL}" \
  --worker-mode "no-gpu" \
  --timeout-sec "${TIMEOUT_SEC:-60}" \
  --case-file "${ROOT_DIR}/cases/extra_light_cases.json" \
  --out-json "${NO_GPU_EXTRA_JSON}" \
  --out-md "${NO_GPU_EXTRA_MD}"

echo "[INFO] Step 1: build remaining-tool cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/build_remaining_cases.py" \
  --schema-json "${ROOT_DIR}/notebook/drugsda_tools_schema.json" \
  --existing-case-files "${ROOT_DIR}/cases/notebook_cases_schema_mapped.json" "${ROOT_DIR}/cases/extra_light_cases.json" \
  --out-case-file "${REMAINING_CASES_JSON}"

echo "[INFO] Step 1: run remaining-tool auto cases"
"${PYTHON_BIN}" "${ROOT_DIR}/common/run_molclaw_cases.py" \
  --server-url "${MOLCLAW_SCP_SERVER_URL}" \
  --proxy-url "${MOLCLAW_PROXY_URL}" \
  --worker-mode "no-gpu" \
  --timeout-sec "${TIMEOUT_SEC:-60}" \
  --case-file "${REMAINING_CASES_JSON}" \
  --out-json "${NO_GPU_REMAINING_JSON}" \
  --out-md "${NO_GPU_REMAINING_MD}"

echo "[INFO] Merge no-gpu reports"
"${PYTHON_BIN}" "${ROOT_DIR}/common/merge_reports.py" \
  --inputs "${NO_GPU_NOTEBOOK_JSON}" "${NO_GPU_EXTRA_JSON}" "${NO_GPU_REMAINING_JSON}" \
  --out-json "${RUN_REPORT_NO_GPU_JSON}" \
  --out-md "${RUN_REPORT_NO_GPU_MD}" \
  --worker-mode "no-gpu"

echo "[INFO] Update summary_diff.md (gpu report optional)"
"${PYTHON_BIN}" "${ROOT_DIR}/common/compare_reports.py" \
  --coverage-report "${COVERAGE_JSON}" \
  --no-gpu-report "${RUN_REPORT_NO_GPU_JSON}" \
  --gpu-report "${GPU_REPORT_JSON}" \
  --out-md "${SUMMARY_DIFF_MD}"

echo "[DONE] no-gpu suite completed"
echo "[DONE] outputs:"
echo "  - ${COVERAGE_JSON}"
echo "  - ${COVERAGE_MD}"
echo "  - ${NO_GPU_REMAINING_JSON}"
echo "  - ${NO_GPU_REMAINING_MD}"
echo "  - ${RUN_REPORT_NO_GPU_JSON}"
echo "  - ${RUN_REPORT_NO_GPU_MD}"
echo "  - ${SUMMARY_DIFF_MD}"
