"""Pinned MolBench questions and train/test task-group isolation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from functools import lru_cache
from pipeline.output_contracts import CONTRACT_VERSION, CONTRACTS, normalize_task_prompt, task_constraints

UPSTREAM_COMMIT = '180abb7679ffc5b1ca9974e7390e141d8098f642'
ROOT = Path(__file__).resolve().parents[2] / 'slime-wd/molbench'
SPECS = {'ms1': ('molbench-ms-1', 'prompt', 'pf', 50),
         'ms2': ('molbench-ms-2', 'question', 'ac', 37),
         'ms3': ('molbench-ms-3', 'questions', 'vs', 25)}
SOURCE_HASHES = {'ms1': 'd13d55edcbdcfa42a7930ec3842bbf4b8af79071b2fa6282789f62d2326e02b1',
                 'ms2': '00fa9acc709fcd185029f8b7e0da22a702a18f35d8e472734f1389980781ae2d',
                 'ms3': '1ae7f0e4818a648068826234fcc7937a4cd17e24d938bde5a0ee36d51d803bb4'}

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(root: Path = ROOT) -> dict:
    """Preserve upstream bytes and change only each question's output instructions."""
    manifest = {'upstream_commit': UPSTREAM_COMMIT, 'contract_version': CONTRACT_VERSION, 'suites': {}}
    for suite, (folder, column, task, count) in SPECS.items():
        source = root / 'upstream' / UPSTREAM_COMMIT / 'data' / folder / f'{folder}.csv'
        raw = source.read_bytes()
        if digest(raw) != SOURCE_HASHES[suite]:
            raise ValueError(f'{suite}: upstream source hash mismatch')
        snapshot = root / 'upstream' / UPSTREAM_COMMIT / 'data' / folder / source.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes() != raw:
            raise ValueError('immutable upstream snapshot differs')
        if not snapshot.exists():
            snapshot.write_bytes(raw)
        with source.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
        if len(rows) != count:
            raise ValueError(f'{suite}: count mismatch')
        constraints = []
        for index, row in enumerate(rows, 1):
            original = row[column]
            c = task_constraints(original, task)
            row[column] = normalize_task_prompt(original, task)
            if task_constraints(row[column], task) != c:
                raise ValueError(f'{suite}/{index}: scientific requirements changed')
            constraints.append({'id': f'molbench_{suite}_{index:03d}', 'exact_count': c.exact_count,
                                'complete_ranking': c.complete_ranking})
        output = root / 'aligned' / 'data' / folder / source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        manifest['suites'][suite] = {'count': count, 'source_sha256': digest(raw),
                                    'output_sha256': digest(output.read_bytes()), 'constraints': constraints}
    target = root / 'aligned/manifest.json'
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


@lru_cache(maxsize=32768)
def molecule_key(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'invalid candidate SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def task_group(task: str, task_type: str) -> str:
    c = task_constraints(task, task_type)
    if task_type == 'ac':
        target = re.search(r'For the target (.*?), you are given', task, re.S)
        if target is None:
            raise ValueError('AC task lacks target identity')
        identity = re.sub(r'\s+', ' ', target[1]).strip().casefold()
    elif task_type == 'vs':
        q = json.loads(task)
        identity = str(q.get('target_chembl_id') or q.get('target_name') or '').strip().casefold()
        if not identity:
            raise ValueError('VS task lacks target identity')
    else:
        identity = ''
    if task_type in {'ac', 'pf', 'vs'}:
        return digest(json.dumps([task_type, identity, sorted({molecule_key(s) for s in c.candidates})]).encode())
    return digest(re.sub(r'\s+', ' ', task).strip().encode())


@lru_cache(maxsize=4)
def heldout_groups(root: Path = ROOT) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for suite, (folder, column, task, count) in SPECS.items():
        source = root / 'upstream' / UPSTREAM_COMMIT / 'data' / folder / f'{folder}.csv'
        if digest(source.read_bytes()) != SOURCE_HASHES[suite]:
            raise ValueError('held-out source changed')
        with source.open(encoding='utf-8-sig', newline='') as f:
            for index, row in enumerate(csv.DictReader(f), 1):
                groups.setdefault(task_group(row[column], task), []).append(f'molbench_{suite}_{index:03d}')
    return groups


def scientific_question_key(task: str) -> str:
    try:
        payload = json.loads(task)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        payload.pop('output_format', None)
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    else:
        text = task
        for contract in CONTRACTS.values():
            text = text.replace(contract, '')
        text = re.sub(r'\s*Selection requirement:\s*The selected_smiles list must contain exactly one candidate\.', '', text)
        text = re.sub(r'(?im)^[ \t]*If none satisfy, output an empty line\.[ \t]*\n?', '', text)
        text = re.sub(r'\s*(?:Output format:\s*)?(?:Only output the corresponding SMILES\.|Print(?: each satisfying| ONLY the selected) SMILES(?: on its own line,)? and nothing else\.)\s*$', '', text, flags=re.I)
        text = re.sub(r'\s*Output format:\s*$', '', text)
    return digest(re.sub(r'\s+', ' ', text).strip().encode())


@lru_cache(maxsize=4)
def heldout_question_keys(root: Path = ROOT) -> dict[str, list[str]]:
    output = {}
    for suite, (folder, column, task, count) in SPECS.items():
        with (root/'upstream'/UPSTREAM_COMMIT/'data'/folder/f'{folder}.csv').open(encoding='utf-8-sig', newline='') as f:
            for index, row in enumerate(csv.DictReader(f), 1):
                key = scientific_question_key(row[column])
                output.setdefault(key, []).append(f'molbench_{suite}_{index:03d}')
    return output


def heldout_matches(task: str, task_type: str, *, source_task_ids: tuple[str, ...] = ()) -> list[str]:
    groups = heldout_groups()
    matches = set(groups.get(task_group(task, task_type), []))
    matches.update(heldout_question_keys().get(scientific_question_key(task), []))
    known_ids = {task_id for ids in groups.values() for task_id in ids}
    matches.update(known_ids.intersection(source_task_ids))
    return sorted(matches)


def require_training_task(task: str, task_type: str, *, source_task_ids: tuple[str, ...] = ()) -> None:
    if normalize_task_prompt(task, task_type) != task:
        raise ValueError('task must be published with the canonical output contract before collection')
    overlap = heldout_matches(task, task_type, source_task_ids=source_task_ids)
    if overlap:
        raise ValueError(f'training task overlaps held-out benchmark: {overlap}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(publish(args.root), indent=2))
