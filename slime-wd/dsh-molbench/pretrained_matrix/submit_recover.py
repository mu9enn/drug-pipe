"""Submit one serial evaluation, rebuilding a failed worker at most twice.

Uses the existing submit_one.sh lifecycle; never restarts the MCP server.
"""
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]


def released(job):
    result = subprocess.run(
        ['rjob', 'get', job, '--namespace', 'ailab-ma4agismall'],
        text=True, capture_output=True, timeout=30,
    )
    text = result.stdout + result.stderr
    return result.returncode == 0 and "'active': 0" in text and bool(re.search(r"\b(Succeeded|Failed|Stopped)\b", text, re.I))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    env = dict(os.environ)
    job, run, infra_name = (env[k] for k in ('JOB_NAME', 'RUN_NAME', 'INFRA_NAME'))
    env.update(EVAL_RECOVERY='1', EVAL_MAX_WORKERS='1')
    run_dir = ROOT / 'outputs/dsh_molbench_evals' / run
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / 'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('recovery_policy') != 'infra_v1':
        raise RuntimeError('Use a new RUN_NAME; this run uses an older recovery policy')
    # A single new-policy model run may own the client at a time.
    with (ROOT / 'outputs/evaluation_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = run_dir / 'worker_attempts.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else []
        if journal and journal[-1]['status'] == 'complete':
            return
        if journal and journal[-1]['status'] == 'started':
            previous = journal[-1]
            infra = Path(previous['infra'])
            if not released(previous['job']):
                raise RuntimeError('Previous worker is not confirmed released; refusing another allocation')
            if (infra / 'evaluation.complete').exists():
                previous['status'] = 'complete'
                save(journal_path, journal)
                return
            previous['status'] = 'failed'
            save(journal_path, journal)
        if journal and not (Path(journal[-1]['infra']) / 'worker_restart_required').exists():
            raise RuntimeError('Previous failure does not qualify for automatic worker recovery')
        for attempt in range(len(journal), 3):
            if attempt:
                time.sleep((60, 180)[attempt - 1])
            current_job = f'{job}-w{attempt + 1}'
            current_infra = f'{infra_name}-w{attempt + 1}'
            infra = ROOT / 'outputs' / current_infra
            env.update(JOB_NAME=current_job, INFRA_NAME=current_infra)
            entry = {'job': current_job, 'infra': str(infra), 'status': 'started'}
            journal.append(entry)
            save(journal_path, journal)
            result = subprocess.run(['bash', str(Path(__file__).with_name('submit_one.sh'))], env=env)
            # submit_one owns cleanup. Do not launch a replacement until release is confirmed.
            for _ in range(20):
                if released(current_job):
                    break
                time.sleep(30)
            else:
                raise RuntimeError('Worker release unconfirmed; queue stopped')
            entry['status'] = 'complete' if result.returncode == 0 else 'failed'
            save(journal_path, journal)
            if result.returncode == 0:
                return
            if not (infra / 'worker_restart_required').exists():
                raise RuntimeError(f'{current_job}: non-retryable failure; see {infra}')
        raise RuntimeError('Worker recovery exhausted (3 allocations); results retained')


if __name__ == '__main__':
    main()
