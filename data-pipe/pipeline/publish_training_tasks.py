"""Publish generated CSV tasks with one output contract and held-out filtering."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from pipeline.output_contracts import normalize_task_prompt, CONTRACT_VERSION
from pipeline.benchmark_release import task_group, heldout_groups, heldout_matches, digest


def publish_csv(path: Path, task: str) -> dict:
    raw = path.read_bytes()
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    column = 'questions' if task == 'vs' else ('prompt' if task == 'pf' else 'question')
    if column not in (fields or []):
        raise ValueError(f'missing task column {column}')
    accepted, excluded = [], []
    groups = heldout_groups()
    for index, row in enumerate(rows, 1):
        row[column] = normalize_task_prompt(row[column], task)
        overlaps = heldout_matches(row[column], task, source_task_ids=tuple(filter(None, [row.get('source_task_id')])))
        if overlaps:
            excluded.append({'source_row': index, 'overlap_ids': overlaps})
        else:
            accepted.append(row)
    temporary = path.with_suffix('.aligned.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(accepted)
    temporary.replace(path)
    report = {'contract_version': CONTRACT_VERSION, 'source_sha256': digest(raw), 'output_sha256': digest(path.read_bytes()),
              'input_count': len(rows), 'accepted_count': len(accepted), 'excluded': excluded}
    path.with_suffix('.contract.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);p.add_argument('--task',required=True,choices=['ac','pf','vs','kg','e2e']);a=p.parse_args()
    print(json.dumps(publish_csv(a.path,a.task),indent=2))
