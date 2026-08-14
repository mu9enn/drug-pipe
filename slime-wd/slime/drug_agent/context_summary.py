"""Grounded LLM summaries for oversized, history-only ReAct decision prefixes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from drug_agent.protocol.react_protocol import parse_react_sequence


SUMMARY_SCHEMA_VERSION = "react_context_summary_v1"
SUMMARY_PROMPT_VERSION = "react-step-context-summarization-v1"
PROTOCOL_TAG_RE = re.compile(r"</?(?:thought|tool_call|observation|final_answer)(?:\s[^>]*)?>", re.I)

DRUG_PIPE_ROOT = Path(__file__).resolve().parents[3]
DATA_PIPE_ROOT = DRUG_PIPE_ROOT / "data-pipe"
SCENE_DIR = DRUG_PIPE_ROOT / "workdir-skills/react-step-context-summarization"
SKILL_DIR = SCENE_DIR / ".claude/skills/summarize-react-step-context"
SCHEMA_PATH = SKILL_DIR / "references/output_schema.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _scalar_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _scalar_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _scalar_values(item)
    elif value is not None:
        yield value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_source_inventory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for relative_index, message in enumerate(messages):
        source_index = int(message.get("_source_message_index", relative_index))
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        parsed = parse_react_sequence(content, role=role)
        blocks = []
        for block in ((parsed.get("blocks") or []) if parsed.get("ok") else []):
            item: dict[str, Any] = {"kind": block.get("kind")}
            if block.get("kind") == "thought":
                item["text"] = block.get("text")
            elif block.get("kind") == "tool_call":
                item["tool_name"] = block.get("tool_name")
                item["arguments"] = copy.deepcopy(block.get("arguments") or {})
            elif block.get("kind") == "observation":
                item["tool_name"] = block.get("tool_name")
                item["payload"] = copy.deepcopy(block.get("payload") or {})
            blocks.append(item)
        inventory.append(
            {
                "source_message_index": source_index,
                "role": role,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "blocks": blocks,
            }
        )
    return inventory


def validate_context_summary(
    summary: Any,
    *,
    source_context_sha256: str,
    source_inventory: list[dict[str, Any]],
) -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    findings = []
    for error in Draft202012Validator(schema).iter_errors(summary):
        location = "/".join(str(part) for part in error.absolute_path) or "$"
        findings.append(f"schema:{location}:{error.message}")
    if findings or not isinstance(summary, dict):
        return findings
    if summary.get("source_context_sha256") != source_context_sha256:
        findings.append("source_context_sha256_mismatch")
    by_index = {int(item["source_message_index"]): item for item in source_inventory}
    previous_max = -1
    for event_index, event in enumerate(summary.get("events") or []):
        indices = event.get("source_message_indices") or []
        if indices != sorted(indices) or (indices and indices[0] < previous_max):
            findings.append(f"event_order_invalid:{event_index}")
        if indices:
            previous_max = max(indices)
        if PROTOCOL_TAG_RE.search(str(event.get("rationale") or "")):
            findings.append(f"rationale_contains_protocol_tag:{event_index}")
        for call in event.get("tool_calls") or []:
            source_index = int(call.get("source_message_index", -1))
            source = by_index.get(source_index, {})
            allowed = {
                str(block.get("tool_name") or "")
                for block in source.get("blocks") or []
                if block.get("kind") == "tool_call"
            }
            if str(call.get("tool_name") or "") not in allowed:
                findings.append(f"ungrounded_tool_call:{event_index}:{source_index}")
            source_text = _canonical_json(source)
            for value in _scalar_values(call.get("arguments") or {}):
                if _canonical_json(value) not in source_text:
                    findings.append(f"ungrounded_argument_value:{event_index}:{source_index}:{value}")
        for observation in event.get("observations") or []:
            source_index = int(observation.get("source_message_index", -1))
            source = by_index.get(source_index, {})
            allowed = {
                str(block.get("tool_name") or "")
                for block in source.get("blocks") or []
                if block.get("kind") == "observation"
            }
            if str(observation.get("tool_name") or "") not in allowed:
                findings.append(f"ungrounded_observation:{event_index}:{source_index}")
            source_text = _canonical_json(source)
            payloads = [
                block.get("payload") or {}
                for block in source.get("blocks") or []
                if block.get("kind") == "observation"
                and str(block.get("tool_name") or "") == str(observation.get("tool_name") or "")
            ]
            failure_tokens = {"error", "failed", "failure", "timeout", "timed_out", "cancelled"}
            explicit_failure = any(
                payload.get("ok") is False
                or str(payload.get("status") or "").strip().lower() in failure_tokens
                for payload in payloads
            )
            explicit_success = any(
                payload.get("ok") is True
                or str(payload.get("status") or "").strip().lower() in {"ok", "success", "succeeded", "completed"}
                for payload in payloads
            )
            expected_status = "failure" if explicit_failure else ("success" if explicit_success else "unknown")
            if observation.get("status") != expected_status:
                findings.append(
                    f"ungrounded_status:{event_index}:{source_index}:{observation.get('status')}:{expected_status}"
                )
            exact_values = [
                *(observation.get("artifacts") or []),
                *(observation.get("paths") or []),
                *(observation.get("ids") or []),
            ]
            if observation.get("error") is not None:
                exact_values.append(observation.get("error"))
            for value in exact_values:
                if str(value) not in source_text:
                    findings.append(f"ungrounded_exact_value:{event_index}:{source_index}:{value}")
    return list(dict.fromkeys(findings))


class ClaudeContextSummarizer:
    """Run the isolated workdir skill with hash caching and bounded retries."""

    def __init__(
        self,
        *,
        cache_root: Path,
        claude_bin: str = "claude",
        timeout_sec: float = 600.0,
        max_attempts: int = 3,
        max_chunk_chars: int = 240_000,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.cache_root = cache_root
        self.claude_bin = claude_bin
        self.timeout_sec = timeout_sec
        self.max_attempts = max_attempts
        self.max_chunk_chars = max_chunk_chars
        self._schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        skill_material = "".join(
            path.read_text(encoding="utf-8")
            for path in sorted(SCENE_DIR.rglob("*"))
            if path.is_file()
        )
        self.skill_sha256 = hashlib.sha256(skill_material.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunks(messages: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
        # Preserve assistant plus following observation messages as one unit.
        units: list[list[dict[str, Any]]] = []
        if all("role" in message for message in messages):
            unit: list[dict[str, Any]] = []
            for message in messages:
                if message.get("role") == "assistant" and unit:
                    units.append(unit)
                    unit = []
                unit.append(message)
            if unit:
                units.append(unit)
        else:
            units = [[message] for message in messages]
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_chars = 0
        for unit in units:
            size = len(_canonical_json(unit))
            if current and current_chars + size > max_chars:
                chunks.append(current)
                current = []
                current_chars = 0
            current.extend(unit)
            current_chars += size
        if current:
            chunks.append(current)
        return chunks

    def _invoke(
        self,
        *,
        payload: Any,
        source_context_sha256: str,
        source_inventory: list[dict[str, Any]],
        mode: str,
        output_max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_key = hashlib.sha256(
            _canonical_json(
                {
                    "payload": payload,
                    "source_context_sha256": source_context_sha256,
                    "skill_sha256": self.skill_sha256,
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    "prompt_version": SUMMARY_PROMPT_VERSION,
                    "mode": mode,
                    "output_max_tokens": output_max_tokens,
                }
            ).encode("utf-8")
        ).hexdigest()
        entry_root = self.cache_root / "entries" / cache_key
        cached_path = entry_root / "context_summary.json"
        if cached_path.is_file():
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            findings = validate_context_summary(
                cached,
                source_context_sha256=source_context_sha256,
                source_inventory=source_inventory,
            )
            if not findings:
                return cached, {"cache_hit": True, "cache_key": cache_key, "attempts": 0}

        if str(DATA_PIPE_ROOT) not in sys.path:
            sys.path.insert(0, str(DATA_PIPE_ROOT))
        from pipeline.claude_agent.session_capture import run_stream_json, select_attempt

        system_prompt = (SCENE_DIR / "system_prompt.md").read_text(encoding="utf-8").strip()
        user_prompt = (SCENE_DIR / "user_prompt.md").read_text(encoding="utf-8").strip()
        attempt_audits = []
        for attempt_number in range(1, self.max_attempts + 1):
            workdir = entry_root / f"attempt-{attempt_number}"
            workdir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SCENE_DIR / ".claude", workdir / ".claude", dirs_exist_ok=True)
            _write_json(
                workdir / "context_request.json",
                {
                    "schema_version": "react_context_summary_request_v1",
                    "mode": mode,
                    "source_context_sha256": source_context_sha256,
                    "current_target_withheld": True,
                    "output_max_tokens": output_max_tokens,
                },
            )
            _write_json(workdir / "omitted_history.json", payload)
            _write_json(workdir / "source_inventory.json", source_inventory)
            result_path = workdir / "context_summary.json"
            if result_path.exists():
                result_path.unlink()
            command = [
                self.claude_bin,
                "--print", "--verbose", "--output-format", "stream-json",
                "--no-session-persistence", "--permission-mode", "bypassPermissions",
                "--tools", "Read,Write,Skill", "--allowedTools", "Read,Write,Skill",
                "--system-prompt", system_prompt, "-p", user_prompt,
            ]
            attempt = run_stream_json(command, cwd=workdir, archive_root=workdir, timeout_sec=self.timeout_sec)
            selected = select_attempt(attempt, workdir / "complete_session.jsonl")
            audit = {
                "attempt": attempt_number,
                "return_code": attempt.get("return_code"),
                "timed_out": attempt.get("timed_out"),
                "raw_session_valid": selected.get("raw_session_valid"),
                "session_sha256": selected.get("sha256"),
                "findings": [],
            }
            if not result_path.is_file():
                audit["findings"].append("missing_context_summary")
            else:
                try:
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    audit["findings"].append(f"json_decode:{exc.msg}")
                else:
                    audit["findings"].extend(
                        validate_context_summary(
                            summary,
                            source_context_sha256=source_context_sha256,
                            source_inventory=source_inventory,
                        )
                    )
                    if (
                        attempt.get("return_code") == 0
                        and not attempt.get("timed_out")
                        and selected.get("raw_session_valid")
                        and not audit["findings"]
                    ):
                        entry_root.mkdir(parents=True, exist_ok=True)
                        _write_json(cached_path, summary)
                        attempt_audits.append(audit)
                        _write_json(entry_root / "audit.json", {"attempts": attempt_audits})
                        return summary, {
                            "cache_hit": False,
                            "cache_key": cache_key,
                            "attempts": attempt_number,
                            "attempt_audits": attempt_audits,
                        }
            attempt_audits.append(audit)
        raise RuntimeError(f"context summarization failed after {self.max_attempts} attempts: {attempt_audits}")

    def summarize(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenizer: Any | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source_context_sha256 = _sha256_value(messages)
        inventory = build_source_inventory(messages)
        chunks = self._chunks(messages, self.max_chunk_chars)
        summaries = []
        audits = []
        for chunk in chunks:
            summary, audit = self._invoke(
                payload=chunk,
                source_context_sha256=source_context_sha256,
                source_inventory=inventory,
                mode="map",
                output_max_tokens=max_tokens,
            )
            summaries.append(summary)
            audits.append(audit)
        rounds = 0
        while len(summaries) > 1:
            rounds += 1
            next_summaries = []
            for group in self._chunks(summaries, self.max_chunk_chars):
                summary, audit = self._invoke(
                    payload={"summary_chunks": group},
                    source_context_sha256=source_context_sha256,
                    source_inventory=inventory,
                    mode="reduce",
                    output_max_tokens=max_tokens,
                )
                next_summaries.append(summary)
                audits.append(audit)
            if len(next_summaries) >= len(summaries):
                # A single reduce over all remaining summaries guarantees progress.
                summary, audit = self._invoke(
                    payload={"summary_chunks": summaries},
                    source_context_sha256=source_context_sha256,
                    source_inventory=inventory,
                    mode="reduce",
                    output_max_tokens=max_tokens,
                )
                next_summaries = [summary]
                audits.append(audit)
            summaries = next_summaries
        if not summaries:
            summaries = [{
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "source_context_sha256": source_context_sha256,
                "events": [],
                "unresolved_state": [],
            }]
        budget_reductions = 0
        if tokenizer is not None and max_tokens is not None:
            while len(tokenizer.encode(_canonical_json(summaries[0]), add_special_tokens=False)) > max_tokens:
                budget_reductions += 1
                if budget_reductions > self.max_attempts:
                    raise RuntimeError(
                        f"semantic summary still exceeds {max_tokens} tokens after {self.max_attempts} budget reductions"
                    )
                summary, audit = self._invoke(
                    payload={"summary_chunks": summaries},
                    source_context_sha256=source_context_sha256,
                    source_inventory=inventory,
                    mode=f"reduce_budget_{budget_reductions}",
                    output_max_tokens=max_tokens,
                )
                summaries = [summary]
                audits.append(audit)
        return summaries[0], {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "source_context_sha256": source_context_sha256,
            "skill_sha256": self.skill_sha256,
            "prompt_version": SUMMARY_PROMPT_VERSION,
            "map_chunks": len(chunks),
            "reduce_rounds": rounds,
            "budget_reductions": budget_reductions,
            "calls": audits,
        }
