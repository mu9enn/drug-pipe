"""Frozen MO subsets; preserve scientific questions and upstream answer field names."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.benchmark_release import ROOT, UPSTREAM_COMMIT, digest
from pipeline.output_contracts import normalize_task_prompt

FILES = {
    'mo-edit': {
        'add': (10, '6f91c85ab9d8125c487ae47d9acc006e3a1baba07a5dbdfc2b4902e2f6906cd0'),
        'delete': (9, 'e6921cb4820dc55478a391741e232157291f628b25b4fda35cd1bec804eaa1d4'),
        'sub': (20, '3f846a4f8aad991d3197532c6addccf5d210b726f8c230ba2d5f6cb9b0ff3fed'),
    },
    'mo-opt': {
        'logp': (11, 'e919e2eb75cd2e173f6a19453b9dbc0185953ed6f0ec79d309218931a25186cd'),
        'qed': (20, '4ca1f659538d08e3d22dd05db9e3f8c75815fa10110798fa5974751e683439bd'),
        'solubility': (8, '0d327c9b83c6ebd8d284b86c301b2bac1a94d68e8f33bcaafc8cd92f7c3ee4f9'),
    },
}


def source_molecule(query: str, suite: str) -> str:
    pattern = (r'Input Molecule:\s*(.*?),\s*Functional Group' if suite == 'mo-edit'
               else r'Source Molecule:\s*(.*?)\.\s*$')
    match = re.search(pattern, query, re.S)
    if not match:
        raise ValueError('MO source molecule not found')
    return match[1].strip()


def publish(root: Path = ROOT) -> dict:
    manifest = {'upstream_commit': UPSTREAM_COMMIT, 'suites': {}}
    for suite, files in FILES.items():
        entries = []
        for subtask, (count, expected_hash) in files.items():
            relative = Path('data/molbench-mo') / f'molbench-{suite}' / f'{subtask}.json'
            raw = (root / 'upstream' / UPSTREAM_COMMIT / relative).read_bytes()
            if digest(raw) != expected_hash:
                raise ValueError(f'{suite}/{subtask}: upstream hash mismatch')
            rows = json.loads(raw)
            if len(rows) != count:
                raise ValueError(f'{suite}/{subtask}: row count mismatch')
            for row in rows:
                source_molecule(row['query'], suite)
                row['query'] = normalize_task_prompt(row['query'], suite)
            target = root / 'aligned-mo' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
            entries.append({'subtask': subtask, 'count': count, 'path': str(relative),
                            'source_sha256': expected_hash, 'output_sha256': digest(target.read_bytes())})
        manifest['suites'][suite] = entries
    (root / 'aligned-mo/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_rows(root: Path, suites: set[str], limit: int):
    manifest = json.loads((root / 'aligned-mo/manifest.json').read_text())
    for suite in FILES:
        if suite not in suites:
            continue
        index = 0
        for entry in manifest['suites'][suite]:
            path = root / 'aligned-mo' / entry['path']
            if digest(path.read_bytes()) != entry['output_sha256']:
                raise ValueError(f'{suite}: published question hash mismatch')
            source_raw = (root / 'upstream' / UPSTREAM_COMMIT / entry['path']).read_bytes()
            if digest(source_raw) != FILES[suite][entry['subtask']][1]:
                raise ValueError('MO upstream snapshot changed')
            original = json.loads(source_raw)
            rows = json.loads(path.read_text())
            count, source_hash = FILES[suite][entry['subtask']]
            if len(rows) != count or entry['source_sha256'] != source_hash:
                raise ValueError('MO frozen source identity mismatch')
            for row, source in zip(rows, original, strict=True):
                expected = dict(source, query=normalize_task_prompt(source['query'], suite))
                if row != expected:
                    raise ValueError('MO publication changed non-format data')
                index += 1
                if limit and index > limit:
                    continue
                yield suite, index, row, source_molecule(source['query'], suite)


if __name__ == '__main__':
    print(json.dumps(publish(), indent=2))
