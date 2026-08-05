#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PIPE_DIR = PROJECT_DIR.parent
REPO_ROOT = DATA_PIPE_DIR.parent
if str(DATA_PIPE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPE_DIR))

from pipeline.claude_agent.session_capture import run_stream_json, select_attempt
from pipeline.cleaning.artifacts import ABSOLUTE_PATH_RE, RELATIVE_PATH_RE
from pipeline.cleaning.invariants import (
    FINAL_RE,
    THOUGHT_RE,
    TOOL_CALL_RE,
    compare_immutable_facts,
    protocol_parts,
)
from pipeline.cleaning.models import react_schema_findings


SCHEMA_VERSION = "canonical_reclean_run_v1"
REVIEW_SCHEMA_VERSION = "canonical_reclean_review_v1"
CHUNK_SCHEMA_VERSION = "canonical_reclean_chunk_notes_v1"
PROTOCOL_TAG_RE = re.compile(
    r"</?(?:thought|tool_call|observation|final_answer)(?:\s[^>]*)?>", re.I
)
CONTAMINATION_RE = re.compile(
    r"(?i)(?:the current rewritten thinking appears to be incomplete|"
    r"I cannot complete this thought)"
)
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
COORD_KEYS = ("message_index", "segment_type", "segment_index")
CONTEXT_EXCERPT_MAX_CHARS = 8000


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class FatalProviderError(RuntimeError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


@dataclass(frozen=True)
class Coordinate:
    message_index: int
    segment_type: str
    segment_index: int

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.message_index, self.segment_type, self.segment_index)


@dataclass
class Segment:
    coordinate: Coordinate
    text: str
    task_prompt: str
    preceding_observation: str
    immutable_terminal: str

    def request_value(self) -> dict[str, Any]:
        return {
            "coordinate": asdict(self.coordinate),
            "text": self.text,
            "preceding_observation": context_excerpt(self.preceding_observation),
            "immutable_terminal": context_excerpt(self.immutable_terminal),
        }

    @property
    def request_characters(self) -> int:
        return sum(
            len(value)
            for value in (
                self.text,
                context_excerpt(self.preceding_observation)["text"],
                context_excerpt(self.immutable_terminal)["text"],
            )
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "record"


def context_excerpt(text: str, max_chars: int = CONTEXT_EXCERPT_MAX_CHARS) -> dict[str, Any]:
    if len(text) <= max_chars:
        rendered = text
        truncated = False
    else:
        half = max_chars // 2
        omitted = len(text) - max_chars
        rendered = (
            text[:half]
            + f"\n\n[... {omitted} source characters omitted from review context ...]\n\n"
            + text[-half:]
        )
        truncated = True
    return {
        "text": rendered,
        "source_characters": len(text),
        "source_sha256": sha256_bytes(text.encode("utf-8")),
        "truncated": truncated,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("key", "token", "secret", "password", "auth")):
                result[key] = "***"
            else:
                result[key] = redact_config(item)
        return result
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def provider_snapshot() -> dict[str, Any]:
    database = Path.home() / ".cc-switch" / "cc-switch.db"
    if database.is_file():
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT id, name, settings_config, meta FROM providers "
                "WHERE app_type='claude' AND is_current=1"
            ).fetchone()
        finally:
            connection.close()
        if row:
            provider_id, name, settings_text, meta_text = row
            try:
                settings = json.loads(settings_text or "{}")
            except json.JSONDecodeError:
                settings = {"unparsed": True}
            try:
                meta = json.loads(meta_text or "{}")
            except json.JSONDecodeError:
                meta = {"unparsed": True}
            sanitized = redact_config(settings)
            public_env = {
                key: value
                for key, value in (sanitized.get("env") or {}).items()
                if key in {
                    "ANTHROPIC_BASE_URL",
                    "ANTHROPIC_MODEL",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL",
                    "ANTHROPIC_REASONING_MODEL",
                }
            }
            identity = {
                "source": "cc-switch",
                "provider_id": provider_id,
                "provider_name": name,
                "public_env": public_env,
                "meta": redact_config(meta),
            }
            identity["fingerprint"] = sha256_bytes(stable_json(identity).encode("utf-8"))
            return identity
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.is_file():
        raise RuntimeError("cannot identify current Claude provider: cc-switch DB and settings missing")
    settings = redact_config(json.loads(settings_path.read_text(encoding="utf-8")))
    identity = {
        "source": "claude_settings",
        "provider_id": "unknown",
        "provider_name": "unknown",
        "public_env": settings.get("env") or {},
    }
    identity["fingerprint"] = sha256_bytes(stable_json(identity).encode("utf-8"))
    return identity


def assert_provider(expected: dict[str, Any]) -> None:
    current = provider_snapshot()
    if current.get("fingerprint") != expected.get("fingerprint"):
        raise RuntimeError(
            "cc-switch Claude provider changed during reclean run: "
            f"{expected.get('provider_id')} -> {current.get('provider_id')}"
        )


def load_tokenizer(path: Path) -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for exact Qwen token gates; install "
            "data-pipe/canonical-reclean/requirements.txt"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )


def token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def repeat_ngram_fraction(ids: list[int], n: int = 32) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[index : index + n]) for index in range(len(ids) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def max_consecutive_equal_paragraphs(text: str) -> int:
    paragraphs = [re.sub(r"\s+", " ", item).strip().casefold() for item in PARAGRAPH_RE.split(text)]
    paragraphs = [item for item in paragraphs if item]
    maximum = current = 0
    previous = None
    for paragraph in paragraphs:
        if paragraph == previous:
            current += 1
        else:
            current = 1
            previous = paragraph
        maximum = max(maximum, current)
    return maximum


def first_task_prompt(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            if "<observation " not in content:
                return content
    return ""


def editable_segments(record: dict[str, Any]) -> list[Segment]:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    task_prompt = first_task_prompt(record)
    segments: list[Segment] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        immutable_terminal = THOUGHT_RE.sub("", content).strip()
        previous = ""
        if message_index > 0 and isinstance(messages[message_index - 1], dict):
            prior = messages[message_index - 1]
            if prior.get("role") == "user" and "<observation " in str(prior.get("content") or ""):
                previous = str(prior.get("content") or "")
        for segment_index, match in enumerate(THOUGHT_RE.finditer(content)):
            segments.append(
                Segment(
                    Coordinate(message_index, "thought", segment_index),
                    match.group(1),
                    task_prompt,
                    previous,
                    immutable_terminal,
                )
            )
        final_matches = list(FINAL_RE.finditer(content))
        if len(final_matches) == 1:
            try:
                payload = json.loads(final_matches[0].group(1))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("summary"), str):
                segments.append(
                    Segment(
                        Coordinate(message_index, "final_summary", 0),
                        payload["summary"],
                        task_prompt,
                        previous,
                        immutable_terminal,
                    )
                )
    return segments


def split_paragraph_chunks(text: str, max_chars: int, overlap_chars: int = 4000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"(\n\s*\n+)", text)
    units: list[str] = []
    for index in range(0, len(paragraphs), 2):
        unit = paragraphs[index]
        if index + 1 < len(paragraphs):
            unit += paragraphs[index + 1]
        if len(unit) <= max_chars:
            units.append(unit)
        else:
            units.extend(unit[start : start + max_chars] for start in range(0, len(unit), max_chars))
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > max_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:]
            current = overlap + unit
        else:
            current += unit
    if current:
        chunks.append(current)
    return chunks


def batches(segments: list[Segment], max_chars: int) -> tuple[list[list[Segment]], list[Segment]]:
    normal: list[list[Segment]] = []
    oversized: list[Segment] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        item_size = segment.request_characters
        if item_size > max_chars:
            if current:
                normal.append(current)
                current, size = [], 0
            oversized.append(segment)
            continue
        if current and size + item_size > max_chars:
            normal.append(current)
            current, size = [], 0
        current.append(segment)
        size += item_size
    if current:
        normal.append(current)
    return normal, oversized


def command_for(prompt_path: Path, claude_bin: str) -> list[str]:
    return [
        claude_bin,
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--safe-mode",
        "--no-session-persistence",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "Read,Write",
        "--allowedTools",
        "Read,Write",
        "--disable-slash-commands",
        "-p",
        prompt_path.read_text(encoding="utf-8"),
    ]


def inspect_provider_terminal_failure(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    fatal_statuses = {401, 403}
    fatal_markers = (
        "failed to authenticate",
        "authentication_failed",
        "insufficient quota",
        "quota exceeded",
        "额度不足",
    )
    detected: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        status = event.get("api_error_status")
        serialized = stable_json(event).casefold()
        if status in fatal_statuses or any(marker in serialized for marker in fatal_markers):
            detected = {
                "reason": "fatal_provider_auth_or_quota_error",
                "api_error_status": status,
                "event_type": event.get("type"),
                "event_subtype": event.get("subtype"),
            }
            if status in fatal_statuses:
                return detected
    return detected


def invoke_claude(
    *,
    workdir: Path,
    request: dict[str, Any],
    output_name: str,
    prompt_name: str,
    claude_bin: str,
    timeout_sec: float,
    provider: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    workdir.mkdir(parents=True, exist_ok=True)
    write_json(workdir / "request.json", request)
    output_path = workdir / output_name
    if output_path.exists():
        output_path.unlink()
    assert_provider(provider)
    attempt = run_stream_json(
        command_for(PROJECT_DIR / "prompts" / prompt_name, claude_bin),
        cwd=workdir,
        archive_root=workdir,
        timeout_sec=timeout_sec,
    )
    selected = select_attempt(attempt, workdir / "complete_session.jsonl")
    metadata = {
        **attempt,
        "selected_session_file": str(workdir / "complete_session.jsonl"),
        "selected_session_sha256": selected.get("sha256"),
        "provider_fingerprint": provider.get("fingerprint"),
        "output_file": str(output_path),
    }
    fatal = inspect_provider_terminal_failure(Path(str(attempt["session_file"])))
    if fatal:
        metadata["failure"] = fatal["reason"]
        metadata["fatal_provider_error"] = fatal
        raise FatalProviderError(fatal["reason"], metadata)
    if attempt.get("timed_out"):
        metadata["failure"] = "claude_timeout"
        return None, metadata
    if attempt.get("return_code") != 0:
        metadata["failure"] = f"claude_exit_code:{attempt.get('return_code')}"
        return None, metadata
    if not selected.get("raw_session_valid"):
        metadata["failure"] = "raw_session_invalid"
        return None, metadata
    if not output_path.is_file():
        metadata["failure"] = f"missing_{output_name}"
        return None, metadata
    try:
        value = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata["failure"] = f"output_json_invalid:{exc.msg}"
        return None, metadata
    if not isinstance(value, dict):
        metadata["failure"] = "output_not_object"
        return None, metadata
    return value, metadata


def validate_review(
    value: dict[str, Any], record_id: str, expected: list[Segment]
) -> tuple[dict[tuple[int, str, int], dict[str, Any]], list[str]]:
    findings: list[str] = []
    if value.get("schema_version") != REVIEW_SCHEMA_VERSION:
        findings.append("review_schema_version_invalid")
    if value.get("record_id") != record_id:
        findings.append("review_record_id_mismatch")
    reviews = value.get("reviews")
    if not isinstance(reviews, list):
        return {}, [*findings, "reviews_not_array"]
    expected_keys = [segment.coordinate.key for segment in expected]
    by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            findings.append(f"review_{index}_not_object")
            continue
        coordinate_value = review.get("coordinate") if isinstance(review.get("coordinate"), dict) else review
        try:
            key = (
                int(coordinate_value["message_index"]),
                str(coordinate_value["segment_type"]),
                int(coordinate_value["segment_index"]),
            )
        except (KeyError, TypeError, ValueError):
            findings.append(f"review_{index}_coordinate_invalid")
            continue
        if key in by_key:
            findings.append(f"duplicate_coordinate:{key}")
            continue
        action = review.get("action")
        if action not in {"keep", "replace", "delete"}:
            findings.append(f"invalid_action:{key}:{action}")
            continue
        if key[1] == "final_summary" and action == "delete":
            findings.append(f"final_summary_delete_forbidden:{key}")
            continue
        replacement = review.get("replacement")
        if action == "replace" and (not isinstance(replacement, str) or not replacement.strip()):
            findings.append(f"replacement_missing:{key}")
        if action != "replace" and replacement not in (None, ""):
            findings.append(f"unexpected_replacement:{key}")
        if isinstance(replacement, str):
            if PROTOCOL_TAG_RE.search(replacement):
                findings.append(f"replacement_contains_protocol_tag:{key}")
            if ABSOLUTE_PATH_RE.search(replacement) or RELATIVE_PATH_RE.search(replacement):
                findings.append(f"replacement_contains_path:{key}")
        by_key[key] = {
            **review,
            "message_index": key[0],
            "segment_type": key[1],
            "segment_index": key[2],
        }
    if set(by_key) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(by_key))
        extra = sorted(set(by_key) - set(expected_keys))
        findings.extend(f"missing_review:{key}" for key in missing)
        findings.extend(f"unexpected_review:{key}" for key in extra)
    return by_key, findings


def validate_chunk_notes(
    value: dict[str, Any], record_id: str, segment: Segment, chunk_index: int
) -> list[str]:
    findings: list[str] = []
    if value.get("schema_version") != CHUNK_SCHEMA_VERSION:
        findings.append("chunk_schema_version_invalid")
    if value.get("record_id") != record_id:
        findings.append("chunk_record_id_mismatch")
    coordinate = value.get("coordinate")
    if not isinstance(coordinate, dict) or tuple(coordinate.get(k) for k in COORD_KEYS) != segment.coordinate.key:
        findings.append("chunk_coordinate_mismatch")
    if value.get("chunk_index") != chunk_index:
        findings.append("chunk_index_mismatch")
    if not isinstance(value.get("unique_content"), str) or not value["unique_content"].strip():
        findings.append("chunk_unique_content_missing")
    return findings


def apply_reviews(
    source: dict[str, Any], reviews: dict[tuple[int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    candidate = copy.deepcopy(source)
    messages = candidate["messages"]
    by_message: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    for (message_index, segment_type, segment_index), review in reviews.items():
        by_message.setdefault(message_index, {})[(segment_type, segment_index)] = review
    for message_index, message_reviews in by_message.items():
        content = str(messages[message_index]["content"])
        current = -1

        def replace_thought(match: re.Match[str]) -> str:
            nonlocal current
            current += 1
            review = message_reviews.get(("thought", current))
            if not review or review["action"] == "keep":
                return match.group(0)
            if review["action"] == "delete":
                return ""
            return f"<thought>{str(review['replacement']).strip()}</thought>"

        content = THOUGHT_RE.sub(replace_thought, content)
        summary_review = message_reviews.get(("final_summary", 0))
        if summary_review and summary_review["action"] != "keep":
            matches = list(FINAL_RE.finditer(content))
            if len(matches) != 1:
                raise ValueError(f"final summary target invalid at message {message_index}")
            payload = json.loads(matches[0].group(1))
            if summary_review["action"] == "delete":
                payload.pop("summary", None)
            else:
                payload["summary"] = str(summary_review["replacement"]).strip()
            rendered = f"<final_answer>{stable_json(payload)}</final_answer>"
            content = content[: matches[0].start()] + rendered + content[matches[0].end() :]
        messages[message_index]["content"] = content.strip()
    return candidate


def load_catalog_names(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    tools = value.get("tools") if isinstance(value, dict) else value
    if not isinstance(tools, list):
        raise ValueError(f"tool catalog does not contain tools array: {path}")
    names = {str(tool.get("name")) for tool in tools if isinstance(tool, dict) and tool.get("name")}
    if not names:
        raise ValueError(f"tool catalog is empty: {path}")
    return names


def quality_report(
    source: dict[str, Any],
    candidate: dict[str, Any],
    tokenizer: Tokenizer,
    catalog_names: set[str],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    schema = react_schema_findings(candidate)
    findings.extend({"reason": item} for item in schema)
    immutable = compare_immutable_facts(source, candidate)
    findings.extend({"reason": item} for item in immutable)
    unknown_tools = sorted(
        {
            str(call.get("tool_name") or "")
            for call in protocol_parts(candidate)["calls"]
            if str(call.get("tool_name") or "") not in catalog_names
        }
    )
    if unknown_tools:
        findings.append({"reason": "unknown_tool_names", "values": unknown_tools})
    decision_stats: list[dict[str, Any]] = []
    for message_index, message in enumerate(candidate.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        decision_tokens = len(token_ids(tokenizer, content))
        stat = {"message_index": message_index, "decision_tokens": decision_tokens, "thoughts": []}
        if decision_tokens > 16384:
            findings.append(
                {"reason": "decision_over_16384_tokens", "message_index": message_index, "value": decision_tokens}
            )
        for segment_index, match in enumerate(THOUGHT_RE.finditer(content)):
            text = match.group(1)
            ids = token_ids(tokenizer, text)
            repeat_fraction = repeat_ngram_fraction(ids)
            consecutive = max_consecutive_equal_paragraphs(text)
            thought_stat = {
                "segment_index": segment_index,
                "tokens": len(ids),
                "repeat_32gram_fraction": round(repeat_fraction, 6),
                "max_consecutive_equal_paragraphs": consecutive,
                "contamination": bool(CONTAMINATION_RE.search(text)),
            }
            stat["thoughts"].append(thought_stat)
            coordinate = {"message_index": message_index, "segment_type": "thought", "segment_index": segment_index}
            if len(ids) > 8192:
                findings.append({"reason": "thought_over_8192_tokens", **coordinate, "value": len(ids)})
            if repeat_fraction > 0.20:
                findings.append({"reason": "thought_repeat_32gram_over_0.20", **coordinate, "value": repeat_fraction})
            if consecutive >= 3:
                findings.append({"reason": "three_consecutive_equal_paragraphs", **coordinate, "value": consecutive})
            if CONTAMINATION_RE.search(text):
                findings.append({"reason": "historical_cleaning_contamination", **coordinate})
        decision_stats.append(stat)
    return {
        "valid": not findings,
        "findings": findings,
        "immutable_findings": immutable,
        "unknown_tools": unknown_tools,
        "decision_stats": decision_stats,
    }


def aggregate_stats(records: list[dict[str, Any]], tokenizer: Tokenizer) -> dict[str, Any]:
    decisions: list[int] = []
    thoughts: list[int] = []
    repeats: list[float] = []
    for record in records:
        for message in record.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            decisions.append(len(token_ids(tokenizer, content)))
            for match in THOUGHT_RE.finditer(content):
                ids = token_ids(tokenizer, match.group(1))
                thoughts.append(len(ids))
                repeats.append(repeat_ngram_fraction(ids))

    def distribution(values: list[float | int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
        ordered = sorted(values)
        def percentile(q: float) -> float | int:
            return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]
        return {
            "count": len(values),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": ordered[-1],
        }
    return {
        "records": len(records),
        "assistant_decision_tokens": {
            **distribution(decisions),
            "over_4096": sum(value > 4096 for value in decisions),
            "over_8192": sum(value > 8192 for value in decisions),
            "over_16384": sum(value > 16384 for value in decisions),
        },
        "thought_tokens": distribution(thoughts),
        "thought_repeat_32gram_fraction": distribution(repeats),
    }


class RecleanRunner:
    def __init__(self, args: argparse.Namespace, tokenizer: Tokenizer, provider: dict[str, Any]):
        self.args = args
        self.tokenizer = tokenizer
        self.provider = provider
        self.catalog_names = load_catalog_names(args.tool_catalog)
        self.prompt_hashes = {
            path.name: sha256_file(path) for path in sorted((PROJECT_DIR / "prompts").glob("*.md"))
        }
        self.print_lock = threading.Lock()

    def log(self, message: str) -> None:
        with self.print_lock:
            print(message, flush=True)

    def normal_review(
        self,
        record_id: str,
        segments: list[Segment],
        workdir: Path,
        previous_findings: list[dict[str, Any]],
    ) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[str, Any]]:
        request = {
            "schema_version": "canonical_reclean_request_v1",
            "record_id": record_id,
            "task_prompt": segments[0].task_prompt if segments else "",
            "segments": [segment.request_value() for segment in segments],
            "previous_findings": previous_findings,
        }
        value, metadata = invoke_claude(
            workdir=workdir,
            request=request,
            output_name="review.json",
            prompt_name="review.md",
            claude_bin=self.args.claude_bin,
            timeout_sec=self.args.timeout_sec,
            provider=self.provider,
        )
        if value is None:
            return {}, {"valid": False, "findings": [metadata.get("failure")], "invocation": metadata}
        reviews, findings = validate_review(value, record_id, segments)
        return reviews, {"valid": not findings, "findings": findings, "invocation": metadata, "review": value}

    def oversized_review(
        self,
        record_id: str,
        segment: Segment,
        workdir: Path,
        previous_findings: list[dict[str, Any]],
    ) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[str, Any]]:
        chunk_values = split_paragraph_chunks(segment.text, self.args.chunk_max_chars)
        notes: list[dict[str, Any]] = []
        invocations: list[dict[str, Any]] = []
        findings: list[str] = []
        for chunk_index, chunk in enumerate(chunk_values):
            request = {
                "schema_version": "canonical_reclean_chunk_request_v1",
                "record_id": record_id,
                "coordinate": asdict(segment.coordinate),
                "chunk_index": chunk_index,
                "chunk_count": len(chunk_values),
                "text": chunk,
                "task_prompt": segment.task_prompt,
                "preceding_observation": context_excerpt(segment.preceding_observation),
                "immutable_terminal": context_excerpt(segment.immutable_terminal),
                "previous_findings": previous_findings,
            }
            value, metadata = invoke_claude(
                workdir=workdir / f"map_{chunk_index:04d}",
                request=request,
                output_name="chunk_notes.json",
                prompt_name="map.md",
                claude_bin=self.args.claude_bin,
                timeout_sec=self.args.timeout_sec,
                provider=self.provider,
            )
            invocations.append(metadata)
            if value is None:
                findings.append(str(metadata.get("failure")))
                continue
            chunk_findings = validate_chunk_notes(value, record_id, segment, chunk_index)
            findings.extend(chunk_findings)
            if not chunk_findings:
                notes.append(value)
        if findings:
            return {}, {"valid": False, "findings": findings, "invocations": invocations}
        reduce_request = {
            "schema_version": "canonical_reclean_reduce_request_v1",
            "record_id": record_id,
            "coordinate": asdict(segment.coordinate),
            "task_prompt": segment.task_prompt,
            "preceding_observation": context_excerpt(segment.preceding_observation),
            "immutable_terminal": context_excerpt(segment.immutable_terminal),
            "chunk_notes": notes,
            "previous_findings": previous_findings,
        }
        value, metadata = invoke_claude(
            workdir=workdir / "reduce",
            request=reduce_request,
            output_name="review.json",
            prompt_name="reduce.md",
            claude_bin=self.args.claude_bin,
            timeout_sec=self.args.timeout_sec,
            provider=self.provider,
        )
        invocations.append(metadata)
        if value is None:
            return {}, {"valid": False, "findings": [metadata.get("failure")], "invocations": invocations}
        reviews, review_findings = validate_review(value, record_id, [segment])
        return reviews, {
            "valid": not review_findings,
            "findings": review_findings,
            "invocations": invocations,
            "review": value,
        }

    def process_record(self, source: dict[str, Any], position: int) -> dict[str, Any]:
        record_id = str(source.get("id") or "")
        record_dir = self.args.run_root / "work" / f"{position:04d}_{safe_name(record_id)}"
        result_path = record_dir / "result.json"
        if self.args.resume and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("status") == "ready"
                and result.get("input_sha256") == sha256_bytes(stable_json(source).encode("utf-8"))
            ):
                self.log(f"[resume] {position + 1}: {record_id}: {result.get('status')}")
                return result
        source_schema = react_schema_findings(source)
        if source_schema:
            result = {
                "id": record_id,
                "position": position,
                "status": "unresolved",
                "reason": "invalid_source_schema",
                "findings": source_schema,
                "source": source,
                "input_sha256": sha256_bytes(stable_json(source).encode("utf-8")),
            }
            write_json(result_path, result)
            return result
        candidate = copy.deepcopy(source)
        previous_findings: list[dict[str, Any]] = []
        pass_audits: list[dict[str, Any]] = []
        for pass_index in range(1, self.args.max_attempts + 1):
            segments = editable_segments(candidate)
            normal, oversized = batches(segments, self.args.batch_max_chars)
            reviews: dict[tuple[int, str, int], dict[str, Any]] = {}
            unit_audits: list[dict[str, Any]] = []
            pass_dir = record_dir / f"pass_{pass_index:02d}"
            failed = False
            for batch_index, group in enumerate(normal):
                partial, audit = self.normal_review(
                    record_id,
                    group,
                    pass_dir / f"batch_{batch_index:04d}",
                    previous_findings,
                )
                unit_audits.append({"kind": "batch", "index": batch_index, **audit})
                if not audit["valid"]:
                    failed = True
                reviews.update(partial)
            for oversized_index, segment in enumerate(oversized):
                partial, audit = self.oversized_review(
                    record_id,
                    segment,
                    pass_dir / f"oversized_{oversized_index:04d}",
                    previous_findings,
                )
                unit_audits.append({"kind": "oversized", "index": oversized_index, **audit})
                if not audit["valid"]:
                    failed = True
                reviews.update(partial)
            if failed:
                previous_findings = [
                    {"reason": str(finding)}
                    for audit in unit_audits
                    for finding in audit.get("findings") or []
                ]
                pass_audits.append(
                    {"pass": pass_index, "status": "provider_or_patch_failed", "units": unit_audits}
                )
                continue
            expected_keys = {segment.coordinate.key for segment in segments}
            if set(reviews) != expected_keys:
                missing = sorted(expected_keys - set(reviews))
                previous_findings = [{"reason": "review_coverage_incomplete", "missing": missing}]
                pass_audits.append(
                    {"pass": pass_index, "status": "review_coverage_incomplete", "missing": missing, "units": unit_audits}
                )
                continue
            try:
                next_candidate = apply_reviews(candidate, reviews)
            except Exception as exc:
                previous_findings = [{"reason": "patch_apply_failed", "detail": str(exc)}]
                pass_audits.append(
                    {"pass": pass_index, "status": "patch_apply_failed", "detail": str(exc), "units": unit_audits}
                )
                continue
            report = quality_report(source, next_candidate, self.tokenizer, self.catalog_names)
            pass_audits.append(
                {"pass": pass_index, "status": "accepted" if report["valid"] else "quality_failed", "quality": report, "units": unit_audits}
            )
            candidate = next_candidate
            if report["valid"]:
                result = {
                    "id": record_id,
                    "position": position,
                    "status": "ready",
                    "attempts_used": pass_index,
                    "input_sha256": sha256_bytes(stable_json(source).encode("utf-8")),
                    "output_sha256": sha256_bytes(stable_json(candidate).encode("utf-8")),
                    "record": candidate,
                    "audit": {"passes": pass_audits, "quality": report},
                }
                write_json(result_path, result)
                self.log(f"[ready] {position + 1}: {record_id} (pass {pass_index})")
                return result
            previous_findings = report["findings"]
        result = {
            "id": record_id,
            "position": position,
            "status": "unresolved",
            "attempts_used": self.args.max_attempts,
            "reason": "max_attempts_exhausted",
            "findings": previous_findings,
            "input_sha256": sha256_bytes(stable_json(source).encode("utf-8")),
            "source": source,
            "last_candidate": candidate,
            "audit": {"passes": pass_audits},
        }
        write_json(result_path, result)
        self.log(f"[unresolved] {position + 1}: {record_id}")
        return result

    def run(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any] | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            futures = {
                executor.submit(self.process_record, record, position): position
                for position, record in enumerate(records)
            }
            try:
                for future in as_completed(futures):
                    position = futures[future]
                    ordered[position] = future.result()
            except FatalProviderError:
                for future in futures:
                    future.cancel()
                raise
        return [item for item in ordered if item is not None]


def git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def validate_run_config(path: Path, expected: dict[str, Any], resume: bool) -> None:
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"run root already initialized: {path.parent}")
        keys = (
            "input_sha256",
            "tool_catalog_sha256",
            "tokenizer",
            "prompt_hashes",
            "provider_fingerprint",
            "record_id",
            "limit",
        )
        differences = [key for key in keys if actual.get(key) != expected.get(key)]
        if differences:
            raise RuntimeError(f"resume configuration changed: {', '.join(differences)}")
        return
    if resume:
        raise FileNotFoundError(f"resume run config missing: {path}")
    write_json(path, expected)


def write_review_samples(
    root: Path,
    source_records: list[dict[str, Any]],
    ready_results: list[dict[str, Any]],
    *,
    count: int = 10,
) -> None:
    source_by_id = {str(record.get("id") or ""): record for record in source_records}
    ranked: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for result in ready_results:
        source = source_by_id[result["id"]]
        candidate = result["record"]
        before_chars = sum(
            len(str(message.get("content") or ""))
            for message in source.get("messages") or []
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
        after_chars = sum(
            len(str(message.get("content") or ""))
            for message in candidate.get("messages") or []
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
        ranked.append((before_chars - after_chars, source, candidate))
    for rank, (reduction, source, candidate) in enumerate(sorted(ranked, reverse=True, key=lambda item: item[0])[:count], start=1):
        changed_messages = []
        for index, (before, after) in enumerate(zip(source.get("messages") or [], candidate.get("messages") or [])):
            if not isinstance(before, dict) or not isinstance(after, dict):
                continue
            if before.get("role") != "assistant" or before.get("content") == after.get("content"):
                continue
            changed_messages.append(
                {
                    "message_index": index,
                    "before": before.get("content"),
                    "after": after.get("content"),
                }
            )
        write_json(
            root / f"{rank:02d}_{safe_name(str(source.get('id') or 'record'))}.json",
            {
                "id": source.get("id"),
                "assistant_character_reduction": reduction,
                "changed_messages": changed_messages,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude-assisted restricted reclean of canonical ReAct prose")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tool-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--record-id", default="")
    parser.add_argument("--batch-max-chars", type=int, default=48000)
    parser.add_argument("--chunk-max-chars", type=int, default=36000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("input", "tool_catalog", "tokenizer", "output", "run_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.max_workers < 1 or args.max_attempts < 1:
        raise ValueError("max-workers and max-attempts must be positive")
    if args.batch_max_chars <= args.chunk_max_chars:
        raise ValueError("batch-max-chars must be greater than chunk-max-chars")
    if not args.input.is_file() or not args.tool_catalog.is_file():
        raise FileNotFoundError("input or tool catalog not found")
    provider = provider_snapshot()
    prompt_hashes = {
        path.name: sha256_file(path) for path in sorted((PROJECT_DIR / "prompts").glob("*.md"))
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "tool_catalog": str(args.tool_catalog),
        "tool_catalog_sha256": sha256_file(args.tool_catalog),
        "tokenizer": str(args.tokenizer),
        "prompt_hashes": prompt_hashes,
        "provider": provider,
        "provider_fingerprint": provider["fingerprint"],
        "max_workers": args.max_workers,
        "timeout_sec": args.timeout_sec,
        "max_attempts": args.max_attempts,
        "batch_max_chars": args.batch_max_chars,
        "chunk_max_chars": args.chunk_max_chars,
        "code_commit": git_commit(),
        "record_id": args.record_id,
        "limit": args.limit,
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    validate_run_config(args.run_root / "run_config.json", config, args.resume)
    records = read_jsonl(args.input)
    ids = [str(record.get("id") or "") for record in records]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("input record IDs must be non-empty and unique")
    selected = records
    if args.record_id:
        selected = [record for record in selected if str(record.get("id") or "") == args.record_id]
        if not selected:
            raise ValueError(f"record ID not found: {args.record_id}")
    if args.limit > 0:
        selected = selected[: args.limit]
    tokenizer = load_tokenizer(args.tokenizer)
    runner = RecleanRunner(args, tokenizer, provider)
    before = aggregate_stats(selected, tokenizer)
    try:
        results = runner.run(selected)
    except FatalProviderError as exc:
        failure = {
            "schema_version": "canonical_reclean_fatal_error_v1",
            "failed_at": utc_now(),
            "reason": str(exc),
            "provider": provider,
            "invocation": exc.metadata,
            "action": (
                "Restore quota for the recorded provider and resume this run. "
                "If selecting another cc-switch provider, start a new run so providers are not mixed."
            ),
        }
        write_json(args.run_root / "fatal_error.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    ready_results = sorted((item for item in results if item["status"] == "ready"), key=lambda item: item["position"])
    unresolved_results = sorted((item for item in results if item["status"] != "ready"), key=lambda item: item["position"])
    ready_records = [item["record"] for item in ready_results]
    audits = [
        {
            "id": item["id"],
            "position": item["position"],
            "status": item["status"],
            "attempts_used": item.get("attempts_used"),
            "input_sha256": item.get("input_sha256"),
            "output_sha256": item.get("output_sha256"),
            "findings": item.get("findings") or [],
            "detail": item.get("audit") or {},
            "workdir": str(args.run_root / "work" / f"{item['position']:04d}_{safe_name(item['id'])}"),
        }
        for item in sorted(results, key=lambda value: value["position"])
    ]
    after = aggregate_stats(ready_records, tokenizer)
    stats = {"schema_version": "canonical_reclean_before_after_v1", "before": before, "after": after}
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "react_trajectories": args.output / "react_trajectories.jsonl",
        "reclean_audit": args.output / "reclean_audit.jsonl",
        "unresolved": args.output / "unresolved.jsonl",
        "before_after_stats": args.output / "before_after_stats.json",
        "manifest": args.output / "manifest.json",
    }
    write_jsonl(outputs["react_trajectories"], ready_records)
    write_jsonl(outputs["reclean_audit"], audits)
    write_jsonl(
        outputs["unresolved"],
        [
            {
                "id": item["id"],
                "position": item["position"],
                "reason": item.get("reason"),
                "findings": item.get("findings") or [],
                "source": item.get("source"),
                "last_candidate": item.get("last_candidate"),
            }
            for item in unresolved_results
        ],
    )
    write_json(outputs["before_after_stats"], stats)
    write_review_samples(args.run_root / "review_samples", selected, ready_results)
    manifest = {
        **config,
        "completed_at": utc_now(),
        "run_root": str(args.run_root),
        "output_root": str(args.output),
        "input_count": len(selected),
        "reviewed_count": len(results),
        "ready_count": len(ready_records),
        "unresolved_count": len(unresolved_results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest["output_sha256"] = {
        name: sha256_file(path)
        for name, path in outputs.items()
        if name != "manifest" and path.is_file()
    }
    write_json(outputs["manifest"], manifest)
    write_jsonl(args.run_root / "progress.jsonl", audits)
    write_jsonl(args.run_root / "reclean_audit.jsonl", audits)
    shutil.copyfile(outputs["unresolved"], args.run_root / "unresolved.jsonl")
    shutil.copyfile(outputs["before_after_stats"], args.run_root / "before_after_stats.json")
    write_json(args.run_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not unresolved_results else 2


if __name__ == "__main__":
    raise SystemExit(main())
