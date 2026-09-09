"""Score saved final messages under predeclared wrapper policies; never change raw runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import run_dsh_molbench as runner


def extract(text: str, policy: str) -> str | None:
    if policy == 'fence_only':
        blocks = re.findall(r'```(?:json)?[ \t]*\r?\n(.*?)```', text, re.S | re.I)
        return blocks[0].strip() if len(blocks) == 1 else None
    objects = []
    decoder = json.JSONDecoder()
    offset = 0
    while (start := text.find('{', offset)) >= 0:
        try:
            _, end = decoder.raw_decode(text[start:])
        except ValueError:
            offset = start + 1
            continue
        objects.append(text[start:start + end])
        offset = start + end
    return objects[0] if len(objects) == 1 else None


def score(run_dir: Path, root: Path) -> dict:
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    samples = runner.load_samples(root, set(manifest['suites']), manifest['limit_per_suite'])
    output = {}
    # Known same-source/QED-objective overlap with three E2E SFT trajectories.
    overlap_id = '48b3a4aa-d573-44f9-8fda-dfc4757a3aaa'
    for policy in ('strict', 'fence_only', 'unique_json_object'):
        target = run_dir / 'format_sensitivity' / policy
        recovered = []
        for sample in samples:
            record = dict(runner.load_record(run_dir, sample) or {
                'task_id': sample.task_id, 'status': 'missing', 'final_text': '',
            })
            text = record.get('final_text') or ''
            if policy != 'strict' and record['status'] == 'completed' and not runner.project_prediction(sample, text)[1]:
                candidate = extract(text, policy)
                if candidate is not None and runner.project_prediction(sample, candidate)[1]:
                    record['final_text'] = candidate
                    recovered.append(sample.task_id)
            runner.write_json(target / 'results' / sample.task_id / 'record.json', record)
        summary = runner.materialize_scores(target, root, samples)
        output[policy] = {'metrics': summary['metrics'], 'recovered_ids': recovered,
                          'sample_count': len(samples), 'valid_output_count': summary['valid_output_count']}
        clean_mo = [s for s in samples if s.suite == 'mo-opt' and s.source_id != overlap_id]
        if clean_mo:
            clean_root = target / 'mo_opt_task_disjoint'
            for sample in clean_mo:
                runner.write_json(clean_root / 'results' / sample.task_id / 'record.json', runner.load_record(target, sample))
            output[policy]['mo_opt_task_disjoint'] = runner.materialize_scores(clean_root, root, clean_mo)['metrics']
    runner.write_json(run_dir / 'format_sensitivity/comparison.json', {
        'source_run': str(run_dir), 'policies': output,
        'note': 'Final-message-only extraction. Same scientific scorer and full denominators within each declared subset.',
        'mo_opt_task_disjoint_excluded_source_ids': [overlap_id],
    })
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--molbench-root', type=Path, default=runner.DEFAULT_MOLBENCH_ROOT)
    args = parser.parse_args()
    score(args.run_dir, args.molbench_root)
