"""Publish an auditable, held-out-disjoint training corpus without guessing labels."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from pipeline.benchmark_release import heldout_groups, heldout_matches, task_group, molecule_key, digest
from pipeline.output_contracts import ANSWER_KEYS, CONTRACT_VERSION, normalize_task_prompt, normalize_final_answer, task_constraints, strict_json_loads
from pipeline.claude_agent.session_capture import extract_assistant_text
from pipeline.cleaning.materialize_sft import materialize_sft
from pipeline.cleaning.skill_native_augmentation import skill_catalog, render_catalog, render_skill_result


def unique_candidate_mapping(values: list[str], candidates: tuple[str, ...]) -> list[str]:
    mapping: dict[str, list[str]] = {}
    for candidate in candidates:
        mapping.setdefault(molecule_key(candidate), []).append(candidate)
    result = []
    for value in values:
        if value in candidates:
            result.append(value)
            continue
        choices = mapping.get(molecule_key(value), [])
        if len(choices) != 1:
            raise ValueError('candidate identity cannot be restored uniquely')
        result.append(choices[0])
    return result


def repair_answer(old: str, task: str, question: str, final_text: str) -> tuple[str, str]:
    c = task_constraints(question, task)
    try:
        return normalize_final_answer(old, task, constraints=c), 'unchanged'
    except ValueError:
        pass
    payload = json.loads(normalize_final_answer(old, task))
    key = ANSWER_KEYS[task]
    if task in {'ac', 'pf', 'vs'}:
        values = [payload[key]] if task == 'ac' else payload[key]
        try:
            mapped = unique_candidate_mapping(values, c.candidates)
            candidate = {key: mapped[0] if task == 'ac' else mapped, 'evidence': payload['evidence']}
            return normalize_final_answer(json.dumps(candidate), task, constraints=c), 'unique_isomeric_candidate_mapping'
        except ValueError:
            pass
    # Never recover a label from a report explicitly consulting an answer key.
    if re.search(r'ground.?truth|expected answer|question\.json.*answer', final_text, re.I | re.S):
        raise ValueError('raw final refers to answer key; no label recovery permitted')
    surfaces = [('raw_final_json', final_text)]
    # Historical migration only: an explicit final answer block is evidence, not a live parser fallback.
    blocks = re.findall(r'<(?:answer|solution)>(.*?)</(?:answer|solution)>', final_text, re.S | re.I)
    if len(blocks) == 1:
        surfaces.append(('raw_final_answer_block', blocks[0].strip()))
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', final_text.strip(), re.S)
    if fenced:
        surfaces.append(('raw_final_json_fence', fenced[1]))
    for source, surface in surfaces:
        try:
            obj = strict_json_loads(surface)
        except ValueError:
            continue
        if isinstance(obj, dict):
            if set(obj) != {key, 'evidence'}:
                continue
            proposal = obj
        elif task in {'pf', 'vs'} and isinstance(obj, list):
            proposal = {key: obj, 'evidence': payload['evidence']}
        elif task == 'ac' and isinstance(obj, str):
            proposal = {key: obj, 'evidence': payload['evidence']}
        else:
            continue
        try:
            return normalize_final_answer(json.dumps(proposal), task, constraints=c), source
        except ValueError:
            continue
    if task == 'ac':
        # Require one whole quoted candidate, not a substring of another SMILES.
        quoted = re.findall(r'`([^`\n]+)`', final_text)
        selected = {v for v in quoted if v in c.candidates}
        if final_text.strip() in c.candidates:
            selected.add(final_text.strip())
        if len(selected) == 1:
            chosen = next(iter(selected))
            location = final_text.find(chosen)
            context = final_text[max(0, location - 250):location + len(chosen) + 250]
            explicit = (final_text.strip() == chosen or
                        re.search(r'\b(?:answer|selected|choose|chosen)\b', context, re.I) or
                        (re.search(r'Based on my computational analysis', context, re.I) and
                         re.search(r'(?:higher|lower) binding affinity', context, re.I)))
            if not explicit:
                raise ValueError('unique mention is not an explicit final selection')
            proposal = {key: next(iter(selected)), 'evidence': payload['evidence']}
            return normalize_final_answer(json.dumps(proposal), task, constraints=c), 'raw_final_unique_explicit_candidate'
    raise ValueError('no unique complete answer in original final response')


def refresh_skill_projection(row: dict, skills: Path) -> None:
    augmentation = row['metadata'].get('skill_native_augmentation')
    if augmentation is None:
        raise ValueError('release requires an explicit native skill projection')
    augmentation['catalog_message'] = render_catalog(skill_catalog(skills))
    removed = set()
    removed_events = []
    for event in row['events']:
        if event['type'] != 'assistant_decision':
            continue
        kept = []
        for call in event.get('tool_calls', []):
            command = str(call.get('arguments', {}).get('command', '')).strip()
            if call['name'].lower() == 'bash' and re.fullmatch(r"(?:ls -la \.claude/?|find \.claude -type d \| head -20)", command):
                removed.add(call['source_tool_use_id'])
                removed_events.append(copy.deepcopy(call))
            else:
                kept.append(call)
        event['tool_calls'] = kept
    row['events'] = [e for e in row['events'] if not (
        e['type'] == 'tool_observation' and e['source_tool_use_id'] in removed
        or e['type'] == 'assistant_decision' and not e['tool_calls'] and e.get('final_answer') is None)]
    augmentation.setdefault('removed_teacher_discovery_calls', []).extend(removed_events)
    calls = {call['source_tool_use_id']: call for event in row['events'] for call in event.get('tool_calls', [])}
    refreshed = 0
    for event in row['events']:
        if event['type'] == 'tool_observation' and event['name'] == 'skill' and not event.get('is_error') and event.get('status') != 'error':
            call = calls[event['source_tool_use_id']]
            event['content'] = render_skill_result(call['arguments']['name'], skills)
            refreshed += 1
    augmentation['runtime_projection_version'] = 'recorded_dsh_skill_body_v1'
    augmentation['refreshed_results'] = refreshed


def publish(source: Path, corpus_root: Path, output: Path, tools: Path, system: Path, prefix: Path, reviewed_reconstructions: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f'new release directory required: {output}')
    reviews = {}
    if reviewed_reconstructions is not None:
        entries = [json.loads(line) for line in reviewed_reconstructions.open()]
        reviews = {entry['id']: entry for entry in entries}
        if len(reviews) != len(entries):
            raise ValueError('duplicate reviewed reconstruction id')
    provenance = {r['canonical_id']: r for line in (corpus_root/'provenance/local_605_manifest.jsonl').open() if (r := json.loads(line))}
    heldout = heldout_groups()
    accepted, audits = [], []
    for line in source.open():
        row = json.loads(line); out = copy.deepcopy(row); task = row['metadata']['task_type']
        audit = {'id': row['id'], 'task_type': task, 'source_semantic_sha256': digest(line.encode()), 'status': 'quarantined'}
        finals = [e for e in out['events'] if e.get('final_answer') is not None]
        audit['old_answer'] = finals[0]['final_answer'] if len(finals) == 1 else None
        try:
            if len(finals) != 1:
                raise ValueError('expected one terminal answer')
            out['user_task'] = normalize_task_prompt(row['user_task'], task)
            overlap = heldout_matches(out['user_task'], task, source_task_ids=tuple(row['metadata'].get('source_task_ids', [])))
            audit['overlap_ids'] = overlap
            if overlap:
                audit.update(status='excluded_overlap', reason='heldout_task_group')
                audits.append(audit); continue
            binding = provenance[row['id']]
            raw = corpus_root/'raw'/binding['local_raw_path'].split('/raw/', 1)[1]
            audit.update(raw_path=str(raw), raw_sha256=digest(raw.read_bytes()))
            if audit['raw_sha256'] != binding['sha256'] or row['metadata']['source_session_sha256'] != binding['sha256']:
                raise ValueError('raw provenance hash mismatch')
            raw_final = extract_assistant_text(raw, final_only=True)
            audit['raw_final'] = raw_final
            audit['raw_final_sha256'] = digest(raw_final.encode())
            if row['id'] in reviews:
                from pipeline.cleaning.recover_historical_answers import validate_review
                review = reviews.pop(row['id'])
                answer = validate_review(review, row, raw)
                method = review['method']
                audit['review'] = review
            else:
                if not raw_final.strip() and task in {'ac', 'vs'}:
                    raise ValueError('interrupted trajectory without a terminal scientific conclusion')
                answer, method = repair_answer(finals[0]['final_answer'], task, out['user_task'], raw_final)
            finals[0]['final_answer'] = answer
            from pipeline.cleaning.answer_recovery import strip_review_text
            out = strip_review_text(out, audit.get('review'))
            out['metadata']['answer_contract_version'] = CONTRACT_VERSION
            out['metadata']['answer_alignment'] = {'method': method, 'raw_sha256': audit['raw_sha256'], 'raw_final_sha256': audit['raw_final_sha256']}
            refresh_skill_projection(out, prefix.parent / '.agents/skills')
            audit.update(status='accepted', method=method, new_answer=answer)
            accepted.append(out)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            audit['reason'] = str(exc)
        audits.append(audit)
    if reviews:
        raise ValueError(f'unused reviewed ids: {sorted(reviews)}')
    output.mkdir(parents=True)
    def jsonl(name, rows):
        with (output/name).open('w') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    jsonl('semantic_trajectories.jsonl', accepted)
    jsonl('answer_audit.jsonl', audits)
    jsonl('quarantined.jsonl', [r for r in audits if r['status'] != 'accepted'])
    materialize_sft(output/'semantic_trajectories.jsonl', output/'sft', deployment_tool_set=tools,
                    system_prompt=system.read_text(), user_prompt_prefix=prefix.read_text())
    manifest = {'contract_version': CONTRACT_VERSION, 'source': str(source), 'source_sha256': digest(source.read_bytes()),
                'counts': dict(Counter(a['status'] for a in audits)), 'repair_methods': dict(Counter(a.get('method') for a in audits if a['status']=='accepted')),
                'heldout_group_count': len(heldout), 'heldout_overlap_after_release': 0,
                'system_sha256': digest(system.read_bytes()), 'prefix_sha256': digest(prefix.read_bytes()),
                'tools_sha256': digest(tools.read_bytes()), 'context_gate': 'pending',
                'files': {str(p.relative_to(output)): digest(p.read_bytes()) for p in output.rglob('*.jsonl')}}
    if reviewed_reconstructions is not None:
        import shutil
        shutil.copyfile(reviewed_reconstructions, output/'reviewed_reconstructions.jsonl')
        manifest['files']['reviewed_reconstructions.jsonl'] = digest(reviewed_reconstructions.read_bytes())
        manifest['reconstruction_policy'] = 'User-authorized trajectory-evidence reconstruction; unresolved ranking tail explicitly disclosed; no ground-truth labels'
    (output/'release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ['source', 'corpus-root', 'output', 'tools', 'system', 'prefix']:
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--reviewed-reconstructions', type=Path)
    a = p.parse_args()
    print(json.dumps(publish(a.source,a.corpus_root,a.output,a.tools,a.system,a.prefix,a.reviewed_reconstructions),indent=2))
