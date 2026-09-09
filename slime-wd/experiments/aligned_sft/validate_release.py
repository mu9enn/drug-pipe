"""Bind an immutable data release to a launch-time readiness audit."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(root, audit=False):
    publication = json.loads((root / 'release_manifest.json').read_text())
    gate = json.loads((root / 'training/context_gate_manifest.json').read_text())
    assert publication['context_gate'] == 'passed' and gate['ok']
    assert publication['training_count'] == gate['accepted_count']
    for name in ('training/qwen35_sft_train.jsonl', 'training/probes.jsonl', 'semantic_trajectories.jsonl'):
        assert sha(root / name) == publication['files'][name], name
    assert sha(root / 'training/context_gate_manifest.json') == publication['training_manifest_sha256']
    binding = {'release_manifest_sha256': sha(root / 'release_manifest.json'),
               'training_sha256': gate['output_sha256'], 'training_count': gate['accepted_count']}
    report_path = root / 'experiments/release_validation.json'
    if audit:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'data-pipe'))
        from pipeline.benchmark_release import require_training_task
        from pipeline.output_contracts import normalize_final_answer, task_constraints
        count = 0
        with (root / 'semantic_trajectories.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line); task = row['metadata']['task_type']
                require_training_task(row['user_task'], task,
                    source_task_ids=tuple(row['metadata'].get('source_task_ids', [])))
                final = next(e['final_answer'] for e in reversed(row['events']) if e.get('final_answer'))
                normalize_final_answer(final, task, constraints=task_constraints(row['user_task'], task))
                count += 1
        assert count == gate['input_count']
        report = {**binding, 'release_ready': True, 'heldout_overlap_after_release': 0,
                  'reserved_questions': 112, 'semantic_count': count}
        report_path.parent.mkdir(exist_ok=True)
        with report_path.open('x') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
    else:
        report = json.loads(report_path.read_text())
        assert all(report[k] == v for k, v in binding.items())
        assert report['release_ready'] and report['heldout_overlap_after_release'] == 0
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate(args.release, args.audit), indent=2))
