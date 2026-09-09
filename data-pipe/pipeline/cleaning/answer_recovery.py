"""Training-only evidence recovery. The live answer contract remains strict."""
from __future__ import annotations
import copy
import json
import math
import re
from pipeline.output_contracts import ANSWER_KEYS, normalize_final_answer, task_constraints
from pipeline.cleaning.recover_historical_answers import identity, rank_predictions

# These are processing metadata, not scientific evidence. Never render them.
AUDIT_EVIDENCE_KEYS = {'identity_matched_affinity_count', 'unresolved_order',
                       'recovery_level', 'review_status', 'repair_method', 'audit_status'}


def final_event(row):
    return next(e for e in reversed(row['events']) if e.get('final_answer') is not None)


def answer_key_findings(row):
    """Flag explicit label consultation, not ordinary discussion of expected results."""
    findings = []
    for e in row['events']:
        text = json.dumps(e, ensure_ascii=False)
        if re.search(r'question\.json.{0,120}(?:expected.answer|answer.{0,30}(?:field|shows|contains))|(?:read|consulted|looked up|copied)\s+(?:the\s+)?(?:reference|expected|standard).answer', text, re.I):
            findings.append(e.get('source_message_id', e.get('source_tool_use_id', 'final')))
        for call in e.get('tool_calls', []):
            if str(call['name']).lower() in {'read', 'bash', 'grep'}:
                args = str(call['arguments'])
                if re.search(r'(?:cat|jq|read|grep|open|file_path).{0,120}(?:private_labels|reference_answer|ground_truth)', args, re.I):
                    findings.append(call['source_tool_use_id'])
    question_reads = {call['source_tool_use_id'] for e in row['events'] for call in e.get('tool_calls', [])
                      if str(call['name']).lower() in {'read', 'bash', 'grep'}
                      and 'question.json' in str(call['arguments'])}
    for e in row['events']:
        if e.get('source_tool_use_id') in question_reads:
            content = e.get('content')
            if isinstance(content, str):
                try: content = json.loads(content)
                except ValueError: continue
            if isinstance(content, dict) and any(content.get(k) not in (None, '', [], {}) for k in ('answer', 'expected_answer', 'ground_truth')):
                findings.append(e['source_tool_use_id'])
    return list(dict.fromkeys(findings))


def strip_review_text(row, review=None):
    """Remove a known migration appendix; do not regex-rewrite scientific prose."""
    out = copy.deepcopy(row)
    final = final_event(out)
    if review and review.get('reasoning_appendix'):
        appendix = review['reasoning_appendix']
        if final.get('reasoning', '').endswith(appendix):
            final['reasoning'] = final['reasoning'][:-len(appendix)].rstrip()
        if review.get('task_type') == 'vs':
            final['reasoning'] += '\n\n' + ranking_reasoning(bool(review.get('details', {}).get('observed_scores')))
    try:
        answer = json.loads(final['final_answer'])
    except (ValueError, TypeError):
        return out
    if isinstance(answer, dict) and isinstance(answer.get('evidence'), list):
        evidence = []
        for item in answer['evidence']:
            if isinstance(item, dict) and (AUDIT_EVIDENCE_KEYS.intersection(item)
                    or item.get('basis') in ('trajectory_predictions_and_report', 'historical_reconstruction')):
                # Uncertainty is a scientific limitation; processing counters are not.
                unresolved = item.get('unresolved_candidates', [])
                if unresolved:
                    evidence.append({'limitation': 'No usable prediction for these candidates; their final priority is unresolved.',
                                     'smiles': unresolved})
            else:
                evidence.append(item)
        answer['evidence'] = evidence
        final['final_answer'] = json.dumps(answer, ensure_ascii=False, separators=(',', ':'))
    return out


def ranking_reasoning(numeric):
    return (('More negative docking energies receive higher priority. ' if numeric else '') +
            'For candidates without comparable numerical predictions, retain the relative ordering supported by the analysis. '
            'Candidates without a supported position come last with unresolved priority; lexical order only breaks this tie and does not imply measured affinity.')


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values(): yield from _strings(v)
    elif isinstance(value, list):
        for v in value: yield from _strings(v)


def evidence_surfaces(row):
    """Only content already visible in this trajectory; never external answer files."""
    for event in row['events']:
        location = event.get('source_message_id', event.get('source_tool_use_id'))
        if event['type'] == 'tool_observation':
            yield location, event['content']
        else:
            yield location, event.get('reasoning', '')
            if event.get('final_answer'): yield location, event['final_answer']
            for call in event.get('tool_calls', []):
                if str(call['name']).lower() in {'write', 'edit'}:
                    for field in ('content', 'new_string', 'text', 'newText'):
                        if field in call['arguments']: yield location, call['arguments'][field]


def recover_answer(row):
    """Return a proposed row plus audit; unresolved rows continue to LLM cleaning."""
    out = copy.deepcopy(row)
    task = out['metadata']['task_type']; final = final_event(out)
    constraints = task_constraints(out['user_task'], task)
    audit = {'id': out['id'], 'status': 'pending', 'evidence_locations': []}
    if answer_key_findings(out):
        audit.update(status='quarantined', reason='explicit_answer_key_consultation')
        return out, audit
    try:
        final['final_answer'] = normalize_final_answer(final['final_answer'], task, constraints=constraints)
        return out, {**audit, 'status': 'unchanged'}
    except ValueError:
        pass
    key = ANSWER_KEYS[task]; candidates = constraints.candidates
    proposals = []
    for location, surface in evidence_surfaces(out):
        for text in _strings(surface):
            # Candidate lists are considered only in conclusions / result reports.
            for m in re.finditer(r'[\[{]', text):
                try: obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
                except ValueError: continue
                if isinstance(obj, dict) and key in obj: vals = obj[key]
                elif isinstance(obj, list) and all(isinstance(v, str) for v in obj) and re.search(r'answer|rank|result|selected', text[:m.start()][-300:], re.I): vals = obj
                else: continue
                values = [vals] if isinstance(vals, str) else vals
                if not isinstance(values, list): continue
                mapped = [identity(v, candidates) for v in values if isinstance(v, str)] if candidates else values
                if task == 'vs': mapped = list(dict.fromkeys(v for v in mapped if v is not None))
                if (mapped or (task == 'pf' and values == [])) and all(v is not None for v in mapped): proposals.append((location, mapped))
            if task == 'ac':
                for line in text.splitlines():
                    if not re.search(r'answer|chosen|selected|predicted to have', line, re.I): continue
                    label = re.search(r'\bMolecule\s+([AB])\b', line, re.I)
                    matches = [c for c in candidates if c in line]
                    if len(matches) == 1: proposals.append((location, matches))
                    elif label and len(candidates) == 2: proposals.append((location, [candidates[ord(label[1].upper())-65]]))
    if task == 'vs':
        observations = {e['source_tool_use_id']: e for e in out['events'] if e['type']=='tool_observation'}
        scores = {}
        for e in out['events']:
            for call in e.get('tool_calls', []):
                if 'molecule_docking_quickvina_fullprocess' not in call['name']: continue
                value = call['arguments'].get('smiles')
                candidate = identity(value, candidates) if isinstance(value, str) else None
                obs = observations.get(call['source_tool_use_id'], {})
                content = obs.get('content', {})
                if isinstance(content, str):
                    try: content = json.loads(content)
                    except ValueError: continue
                score = content.get('docking_affinity_value') if isinstance(content, dict) else None
                if (candidate and type(score) in (int, float) and math.isfinite(score)
                        and not obs.get('is_error') and obs.get('status') != 'error'):
                    scores[candidate] = score; audit['evidence_locations'].append(call['source_tool_use_id'])
        historical = proposals[-1][1] if proposals else []
        if scores or historical:
            vals, missing = rank_predictions(candidates, scores, historical)
            proposals = [(proposals[-1][0] if proposals else final['source_message_id'], vals)]
            audit.update(level='unresolved_tail' if missing else 'evidence_reconstruction', unresolved_candidates=missing)
            final['reasoning'] = final.get('reasoning', '') + '\n\n' + ranking_reasoning(bool(scores))
    for location, values in reversed(proposals):
        proposal = {key: values[0] if task == 'ac' and len(values)==1 else values, 'evidence': []}
        try: answer = normalize_final_answer(json.dumps(proposal), task, constraints=constraints)
        except ValueError: continue
        final['final_answer'] = answer
        audit.update(status='recovered', level=audit.get('level','explicit_selection'))
        audit['evidence_locations'].append(location)
        return out, audit
    return out, audit


def apply_answer_patch(row, patch):
    """A semantic proposal, never a tool-history edit. Audit stays outside the row."""
    refs = patch.get('evidence_locations', [])
    valid_refs = {loc for loc, _ in evidence_surfaces(row)}
    if not refs or any(ref not in valid_refs for ref in refs):
        raise ValueError('answer recovery must cite existing trajectory evidence')
    out = copy.deepcopy(row); task=out['metadata']['task_type']
    final=final_event(out)
    try:
        normalize_final_answer(final['final_answer'], task, constraints=task_constraints(out['user_task'], task))
    except ValueError:
        pass
    else:
        raise ValueError('answer recovery is only for a noncompliant final answer')
    final['final_answer'] = normalize_final_answer(json.dumps(patch['answer']), task,
        constraints=task_constraints(out['user_task'], task))
    if patch.get('reasoning'): final['reasoning'] = patch['reasoning']
    return strip_review_text(out)
