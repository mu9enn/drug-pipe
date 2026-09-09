"""Explicit, audited historical reconstruction; never used by the live answer parser.

Rank known historical predictions first. Unscored candidates are an uncertainty
bucket at the end, not invented affinity measurements. Tie order is lexical.
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
from pipeline.benchmark_release import molecule_key, digest
from pipeline.output_contracts import ANSWER_KEYS, normalize_final_answer, task_constraints


def identity(value, candidates):
    if value in candidates:
        return value
    try:
        key = molecule_key(value)
        matches = [c for c in candidates if molecule_key(c) == key]
    except ValueError:
        return None
    return matches[0] if len(matches) == 1 else None


def rank_predictions(candidates, scores, historical_order):
    if len(set(candidates)) != len(candidates):
        raise ValueError('duplicate source candidates')
    if not scores and not historical_order:
        raise ValueError('no scientific ranking evidence; interrupted trajectory')
    if any(c not in candidates for c in scores) or any(c not in candidates for c in historical_order):
        raise ValueError('candidate outside task')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values()):
        raise ValueError('non-finite score')
    # Preserve the established prediction order when numeric evidence is unavailable.
    positions = {c: i for i, c in enumerate(dict.fromkeys(historical_order))}
    if scores:
        known = sorted(scores, key=lambda c: (scores[c], positions.get(c, len(candidates)), c))
        unscored = [c for c in historical_order if c not in scores]
    else:
        known, unscored = [], list(dict.fromkeys(historical_order))
    order = list(dict.fromkeys(known + unscored))
    missing = sorted(set(candidates) - set(order))
    return order + missing, missing


def validate_review(review, row, raw):
    if digest(raw.read_bytes()) != review['raw_sha256']:
        raise ValueError('review provenance mismatch')
    if review['id'] != row['id'] or digest(row['user_task'].encode()) != review['question_sha256']:
        raise ValueError('review task mismatch')
    if not review['evidence_locations'] or not review['reasoning_appendix']:
        raise ValueError('review requires evidence and explicit reconstruction reasoning')
    return normalize_final_answer(json.dumps(review['answer']), row['metadata']['task_type'],
                                  constraints=task_constraints(row['user_task'], row['metadata']['task_type']))
