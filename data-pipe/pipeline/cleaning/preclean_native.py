from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.cleaning.io import base_manifest, write_json, write_jsonl, write_pretty_json
from pipeline.cleaning.semantic_builder import (
    SemanticConstructionError,
    _group_assistant_responses,
    _index_results,
    _parse_content,
    _unwrap_final,
)
from pipeline.cleaning.trace_parser import (
    RolloutSample,
    discover_rollout_samples,
    discover_run_dirs,
    load_session_events,
    question_text,
    safe_load_json,
)


SCHEMA_VERSION = "claude_raw_qwen35_native_projection_v1"


def _tool_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def raw_events_to_qwen35_native(
    events: list[dict[str, Any]],
    *,
    record_id: str,
    user_task: str,
    source_session: str,
) -> dict[str, Any]:
    """Project Claude events into native message structure without cleaning.

    This is an audit snapshot, not a training source. It intentionally retains
    runtime calls, machine paths, and complete observations so the following
    deterministic cleaning stage remains directly inspectable.
    """
    responses = _group_assistant_responses(events)
    results = _index_results(events)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_task}]
    assistant_provenance: list[dict[str, Any]] = []

    for response_index, response in enumerate(responses):
        thinking = [
            str(item.get("thinking") or "").strip()
            for item in response.content
            if item.get("type") == "thinking" and str(item.get("thinking") or "").strip()
        ]
        visible = [
            str(item.get("text") or "").strip()
            for item in response.content
            if item.get("type") == "text" and str(item.get("text") or "").strip()
        ]
        raw_calls = [item for item in response.content if item.get("type") == "tool_use"]
        assistant: dict[str, Any] = {
            "role": "assistant",
            "reasoning_content": "\n\n".join(thinking),
            "content": "\n\n".join(visible),
        }
        if not raw_calls and response_index == len(responses) - 1:
            assistant["content"] = _unwrap_final(assistant["content"])
        if raw_calls:
            assistant["tool_calls"] = [
                {
                    "id": str(call.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(call.get("name") or ""),
                        "arguments": call.get("input") if isinstance(call.get("input"), dict) else {},
                    },
                }
                for call in raw_calls
            ]
        messages.append(assistant)
        assistant_provenance.append(
            {"message_index": len(messages) - 1, "source_message_id": response.message_id}
        )

        for call in raw_calls:
            call_id = str(call.get("id") or "")
            if not call_id or call_id not in results:
                raise SemanticConstructionError(f"raw call {call_id or '<missing>'} has no tool_result")
            result = results[call_id]
            messages.append(
                {
                    "role": "tool",
                    "name": str(call.get("name") or ""),
                    "tool_call_id": call_id,
                    "content": _tool_content(_parse_content(result.get("content"))),
                }
            )

    if not responses:
        raise SemanticConstructionError("trajectory has no assistant response")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "messages": messages,
        "cleaning_state": {
            "teacher_skill_filter_applied": False,
            "path_sanitization_applied": False,
            "observation_compaction_applied": False,
            "reasoning_cleaning_applied": False,
        },
        "provenance": {
            "source_session": source_session,
            "assistant_messages": assistant_provenance,
        },
    }


def project_sample(sample: RolloutSample) -> tuple[dict[str, Any], dict[str, Any]]:
    question = safe_load_json(sample.sample_dir / "question.json") or safe_load_json(sample.row_dir / "question.json")
    session_path = sample.sample_dir / "complete_session.jsonl"
    events, malformed, runner_error = load_session_events(session_path)
    sample_key = f"{sample.row_number}:{sample.dataset_index}:{sample.rollout_index}:{session_path.resolve()}"
    record_id = f"native_raw_{hashlib.sha256(sample_key.encode()).hexdigest()[:16]}"
    audit = {
        "id": record_id,
        "source_session": str(session_path.resolve()),
        "session_event_count": len(events),
        "malformed_session_line_count": malformed,
        "runner_error_last_line": runner_error,
    }
    if not session_path.is_file() or malformed or runner_error:
        return {}, {**audit, "status": "rejected"}
    try:
        record = raw_events_to_qwen35_native(
            events,
            record_id=record_id,
            user_task=question_text(question),
            source_session=str(session_path.resolve()),
        )
    except SemanticConstructionError as exc:
        return {}, {**audit, "status": "rejected", "error": str(exc)}
    return record, {**audit, "status": "projected", "message_count": len(record["messages"])}


def project_results(results_root: Path, output_root: Path) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    run_dirs = discover_run_dirs(results_root)
    if not run_dirs:
        raise FileNotFoundError(f"no run_config.json found under {results_root}")
    projected: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for sample in discover_rollout_samples(run_dir):
            record, audit = project_sample(sample)
            audits.append(audit)
            if record:
                projected.append(record)
    write_jsonl(output_root / "qwen35_native_raw.jsonl", projected)
    write_pretty_json(output_root / "qwen35_native_raw.pretty.json", projected)
    write_jsonl(output_root / "projection_audit.jsonl", audits)
    manifest = {
        **base_manifest(
            step="claude_raw_to_qwen35_native_projection",
            source=results_root,
            repo_root=Path(__file__).resolve().parents[3],
        ),
        "output_schema_version": SCHEMA_VERSION,
        "input_count": len(audits),
        "projected_count": len(projected),
        "rejected_count": len(audits) - len(projected),
        "training_source": False,
        "cleaning_applied": False,
    }
    write_json(output_root / "projection_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Project raw Claude events into uncleaned Qwen3.5 native messages.")
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(project_results(args.results_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
