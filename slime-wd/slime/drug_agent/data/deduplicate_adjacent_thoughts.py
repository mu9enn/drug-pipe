from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.utils import read_jsonl, write_json, write_jsonl


THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
INFORMATION_ATOM_RE = re.compile(
    r"<artifact:[^>]+>|-?\d+(?:\.\d+)?(?:e[-+]?\d+)?|\b[A-Z][A-Z0-9_-]{1,}\b"
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _choose_representative(first: str, second: str) -> tuple[str, str]:
    """Keep the more informative paraphrase without synthesizing new science."""
    first_normalized = _normalize(first)
    second_normalized = _normalize(second)
    if first_normalized == second_normalized:
        return first.strip(), "exact_delete_later"
    first_score = (len(set(INFORMATION_ATOM_RE.findall(first))), len(first_normalized))
    second_score = (len(set(INFORMATION_ATOM_RE.findall(second))), len(second_normalized))
    if second_score > first_score:
        return second.strip(), "near_duplicate_keep_more_informative"
    return first.strip(), "near_duplicate_keep_more_informative"


def deduplicate_content(
    content: str,
    *,
    threshold: float = 0.85,
) -> tuple[str, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    while True:
        matches = list(THOUGHT_RE.finditer(content))
        replacement: tuple[int, int, str, dict[str, Any]] | None = None
        for index, (first_match, second_match) in enumerate(zip(matches, matches[1:])):
            # Only merge literally adjacent thought blocks. A tool call or any
            # other protocol content between them is a real decision boundary.
            if content[first_match.end() : second_match.start()].strip():
                continue
            first = first_match.group(1).strip()
            second = second_match.group(1).strip()
            first_normalized = _normalize(first)
            second_normalized = _normalize(second)
            if not first_normalized or not second_normalized:
                continue
            ratio = difflib.SequenceMatcher(
                None, first_normalized, second_normalized, autojunk=True
            ).ratio()
            if ratio < threshold:
                continue
            kept, strategy = _choose_representative(first, second)
            action = {
                "thought_index": index,
                "similarity": round(ratio, 6),
                "strategy": strategy,
                "first_sha256": _sha256_text(first),
                "second_sha256": _sha256_text(second),
                "kept_sha256": _sha256_text(kept),
                "first": first,
                "second": second,
                "kept": kept,
            }
            replacement = (
                first_match.start(),
                second_match.end(),
                f"<thought>{kept}</thought>",
                action,
            )
            break
        if replacement is None:
            break
        start, end, rendered, action = replacement
        content = content[:start] + rendered + content[end:]
        actions.append(action)
    return content, actions


def deduplicate_records(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    report_path: Path,
    *,
    threshold: float = 0.85,
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    output: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        migrated = dict(record)
        messages = []
        record_actions: list[dict[str, Any]] = []
        for message_index, message in enumerate(record.get("messages") or []):
            item = dict(message)
            if item.get("role") == "assistant" and isinstance(item.get("content"), str):
                item["content"], actions = deduplicate_content(
                    item["content"], threshold=threshold
                )
                for action in actions:
                    action["message_index"] = message_index
                    counts[action["strategy"]] += 1
                record_actions.extend(actions)
            messages.append(item)
        migrated["messages"] = messages
        output.append(migrated)
        audits.append({
            "source_id": record.get("id"),
            "status": "deduplicated" if record_actions else "unchanged",
            "actions": record_actions,
        })
        counts["records_deduplicated" if record_actions else "records_unchanged"] += 1

    write_jsonl(output_path, output)
    write_jsonl(audit_path, audits)
    report = {
        "schema_version": "drug_agent_adjacent_thought_dedup_v1",
        "input": str(input_path),
        "output": str(output_path),
        "record_count": len(records),
        "threshold": threshold,
        "counts": dict(counts),
        "audit": str(audit_path),
    }
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically collapse adjacent near-duplicate thought blocks"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    report = deduplicate_records(
        Path(args.input),
        Path(args.output),
        Path(args.audit),
        Path(args.report),
        threshold=args.threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
