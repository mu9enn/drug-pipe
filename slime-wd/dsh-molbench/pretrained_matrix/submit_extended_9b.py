"""Submit the three authorized 9B settings on MS-3 / MO-Opt / MO-Edit."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[3]
    release = project.parent / 'drug_wd/drug_pipe_regular_v1_20260908'
    experiment = release / 'experiments/ms3_mo_0909c'
    audit = json.loads((experiment / 'holdout_audit.json').read_text())
    assert audit['training_count'] == 596 and not audit['ms_heldout_task_overlaps']
    assert (experiment / 'offline_scoring.complete').exists(), 'run the offline scorer regression gate first'
    wd = project / 'slime-wd'
    score_python = wd / 'outputs/dsh_eval_runtime/mo-score-venv/bin/python'
    original = wd / 'data/Qwen3.5-9B'
    sft = release / 'experiments/q35-9b-596-0909a/hf'
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        assert (original / name).read_bytes() == (sft / name).read_bytes(), name
    settings = [('orig-l1', original, 'qwen3.5-9b-original-local', 'l1-flat'),
                ('orig-hier', original, 'qwen3.5-9b-original-local', 'legacy-hierarchy'),
                ('sft-l1', sft, 'qwen3.5-9b-sft-local', 'l1-flat')]
    manifest = {
        'suites': {'ms3': 25, 'mo-opt': 39, 'mo-edit': 39},
        'gpu_count_per_job': 2, 'tp_size': 2, 'seed': 42,
        'context_length': 262144, 'max_output_tokens': 16384, 'temperature': 0,
        'native_thinking': True, 'max_workers': 2, 'task_timeout_sec': 14400,
        'infrastructure_retries': 2, 'sft_hierarchy': False,
        'answer_policies': ['strict', 'fence_only', 'unique_json_object'],
        'holdout_audit': audit, 'jobs': [],
        'scoring_environment': str(score_python),
        'chemcotbench_commit': subprocess.check_output(
            ['git', '-C', str(wd / 'molbench/ChemCoTBench'), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_hashes': {},
    }
    sources = [project / 'data-pipe/pipeline/output_contracts.py',
               project / 'data-pipe/pipeline/mo_benchmark_release.py',
               wd / 'dsh-molbench/run_dsh_molbench.py', wd / 'dsh-molbench/score_format_sensitivity.py',
               wd / 'dsh-molbench/pretrained_matrix/run_worker.sh',
               wd / 'dsh-molbench/pretrained_matrix/submit_one.sh',
               wd / 'molbench/aligned/manifest.json', wd / 'molbench/aligned-mo/manifest.json']
    for path in sources:
        manifest['source_hashes'][str(path.relative_to(project))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for label, model, model_id, variant in settings:
        job = f'av4-ext-{label}-0909c'
        run = f'aligned-v4-9b-{label}-ms3-mo-full-0909c'
        infra = wd / 'outputs' / f'infra-{run}'
        assert not infra.exists(), f'run already exists: {infra}'
        command = ['env', f'JOB_NAME={job}', f'MODEL_DIR={model}', f'MODEL_ID={model_id}',
                   f'MODEL_NAME={model_id}', 'GPU_COUNT=2', 'TP_SIZE=2', 'EVAL_SEED=42',
                   'SUITES=ms3 mo-opt mo-edit', f'WORKSPACE_VARIANT={variant}',
                   f'RUN_NAME={run}', f'INFRA_NAME=infra-{run}', f'SCORE_PYTHON={score_python}',
                   'bash', str(wd / 'dsh-molbench/pretrained_matrix/submit_one.sh')]
        script = experiment / f'{job}.sh'
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n')
        manifest['jobs'].append({'name': job, 'model': str(model), 'variant': variant,
                                 'run_name': run, 'entrypoint': 'pretrained_matrix/run_worker.sh',
                                 'login_orchestrator': str(script)})
    target = experiment / 'experiment_manifest.json'
    with target.open('x') as f:
        f.write(json.dumps(manifest, indent=2) + '\n')
    for job in manifest['jobs']:
        subprocess.run(['tmux', 'new-session', '-d', '-s', job['name'],
                        shlex.join(['bash', job['login_orchestrator']])], check=True)
    print(target)


if __name__ == '__main__':
    main()
