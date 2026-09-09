"""Publish a fresh SFT release with the existing whole-trajectory token gate."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from pipeline.benchmark_release import digest
from pipeline.cleaning.materialize_sft import materialize_sft
from pipeline.cleaning.gate_release import gate


def publish(input_path: Path, output: Path, model: Path, audits=(), *, deployment_tool_set=None, system_prompt_file=None, user_prompt_prefix_file=None, reviews_path=None):
    # A release is immutable. Intermediate cleaning directories remain resumable.
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[3]
    from pipeline.cleaning.answer_recovery import strip_review_text
    reviews = {r['id']: r for r in map(json.loads, reviews_path.open())} if reviews_path else {}
    sanitation = []
    with input_path.open() as source, (output / 'semantic_trajectories.jsonl').open('w') as target:
        for line in source:
            row = json.loads(line)
            cleaned = strip_review_text(row, reviews.get(row['id']))
            target.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
            if cleaned != row:
                sanitation.append({'id': row['id'], 'operation': 'remove_processing_metadata',
                    'source_sha256': digest(line.encode()), 'review_id': row['id'] if row['id'] in reviews else None})
    audit_root = output / 'audit'; audit_root.mkdir()
    (audit_root / 'sanitation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sanitation))
    for path in [*audits, *([reviews_path] if reviews_path else [])]:
        shutil.copy2(path, audit_root / path.name)
    materialize_sft(output / 'semantic_trajectories.jsonl', output / 'sft',
        deployment_tool_set=deployment_tool_set or project / 'data-pipe/configs/dsh_molclaw_tool_set.json',
        system_prompt=(system_prompt_file or project / 'data-pipe/pipeline/cleaning/prompts/qwen35_system.md').read_text(),
        user_prompt_prefix=(user_prompt_prefix_file or project / 'workdir-skills/molclaw-l1-workspace/prompt_prefix.md').read_text())
    manifest = {'schema_version': 'drug_pipe_regular_release_v1', 'source': str(input_path.resolve()),
        'context_gate': 'pending', 'jobs': 'paused', 'files': {
            str(path.relative_to(output)): digest(path.read_bytes())
            for path in output.rglob('*') if path.is_file()}}
    (output / 'release_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return gate(output, model)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--tokenizer', required=True, type=Path)
    p.add_argument('--audit', action='append', type=Path, default=[])
    p.add_argument('--deployment-tool-set', type=Path)
    p.add_argument('--system-prompt-file', type=Path)
    p.add_argument('--user-prompt-prefix-file', type=Path)
    p.add_argument('--reviews', type=Path, help='Historical sidecar: exact appendices to remove, never new labels')
    a = p.parse_args()
    result = publish(a.input, a.output_root, a.tokenizer, a.audit, deployment_tool_set=a.deployment_tool_set,
                     system_prompt_file=a.system_prompt_file, user_prompt_prefix_file=a.user_prompt_prefix_file, reviews_path=a.reviews)
    print(json.dumps({k: v for k, v in result.items() if k != 'record_lengths'}, indent=2))
