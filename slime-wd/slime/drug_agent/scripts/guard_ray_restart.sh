#!/usr/bin/env bash
# Refuse to tear down a Ray cluster which is serving another submitted job.
set -euo pipefail

if [ "${DRUG_AGENT_ALLOW_RAY_RESTART_WITH_ACTIVE_JOBS:-0}" = "1" ]; then
  exit 0
fi

if ! timeout 5 ray status >/dev/null 2>&1; then
  exit 0
fi

active_jobs=$(python3 - <<'PY'
from ray.job_submission import JobSubmissionClient

try:
    jobs = JobSubmissionClient("http://127.0.0.1:8265").list_jobs()
except Exception:
    raise SystemExit(0)

for job in jobs:
    if str(job.status) in {"RUNNING", "PENDING"}:
        print(f"{job.submission_id}\t{job.status}\t{job.entrypoint}")
PY
)

if [ -n "$active_jobs" ]; then
  echo "Refusing to restart Ray while submitted jobs are active:" >&2
  echo "$active_jobs" >&2
  echo "Wait for them to finish, or explicitly set DRUG_AGENT_ALLOW_RAY_RESTART_WITH_ACTIVE_JOBS=1." >&2
  exit 3
fi
