from __future__ import annotations

import json
import re
import shlex
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from pipeline.cleaning.artifacts import inspect_observation_status, sanitize_artifact_paths
except ImportError:
    from cleaning.artifacts import inspect_observation_status, sanitize_artifact_paths


MOLCLAW_PREFIX = "mcp__molclaw-scp__"
LOCAL_TOOL_NAMES = frozenset({"Read", "Write", "Edit", "Bash", "Grep", "Glob", "Skill"})
SAFE_BASH_COMMANDS = frozenset(
    {"pwd", "ls", "find", "cat", "head", "tail", "grep", "wc", "stat", "mkdir", "cp", "base64", "realpath", "readlink", "test", "echo", "cd"}
)
UNSAFE_BASH_RE = re.compile(
    r"(?:`|\$\(|(?:^|[\s;&|])(curl|wget|python\d*|perl|ruby|node|sudo|rm|chmod|chown|kill|pkill|"
    r"apt(?:-get)?|yum|dnf|pip\d*|conda|bash|sh)(?=$|[\s;&|]))",
    re.IGNORECASE,
)
SKILL_LEVEL_RE = re.compile(r"(?:^|/)(?:\.claude/skills/)?(L[23]_[^/]+|LR_research)(?:/|$)", re.IGNORECASE)
L1_PATH_RE = re.compile(
    r"(?:^|/)(?:\.claude/skills/)?L1_tools/([^/\s\"']+)(?:/([^\s\"']*))?",
    re.IGNORECASE,
)
TEACHER_SIDECAR_NAMES = frozenset(
    {
        "claude.md",
        "complete_session.jsonl",
        "parsed_answer.json",
        "prompt.txt",
        "question.json",
        "run_meta.json",
    }
)
QUICKVINA_TOOL = "molecule_docking_quickvina_fullprocess"
HIGHER_ORDER_RANKING_MARKERS = ("equiscore", "consensus")
CANONICAL_SYSTEM_PROMPT = """You are a scientific agent operating under the canonical ReAct protocol.
Use only recorded MolClaw calls and supported local file/skill calls from the recorded execution.
Write scientific reasoning inside
<thought>...</thought>, calls inside <tool_call>...</tool_call>, recorded results inside
<observation>...</observation>, and the grounded result inside <final_answer>...</final_answer>.
Never invent a tool result. Observations must come from execution history, and the final answer
must be supported by the task result and recorded observations."""
MOLCLAW_ONLY_SYSTEM_PROMPT = CANONICAL_SYSTEM_PROMPT.replace(
    "recorded MolClaw calls and supported local file/skill calls",
    "recorded MolClaw calls",
)
IMPORTANT_OBSERVATION_KEYS = (
    "status",
    "ok",
    "error",
    "message",
    "msg",
    "summary",
    "score",
    "scores",
    "value",
    "count",
    "pocket_count",
    "top_pocket",
    "ranking",
    "result",
    "results",
    "artifact",
    "output_path",
    "output_file",
)
OUTPUT_ARTIFACT_KEYS = ("output_file", "output_path", "artifact")
DELIVERABLE_FILE_RE = re.compile(
    r"(?i)(?:<artifact:[^>]+>|(?<![\w.-])[\w.-]+\.(?:pdb|cif|mmcif|sdf|mol|mol2|pdbqt|csv|tsv|json|npy|npz|pt|pkl))"
)
ARTIFACT_REF_RE = re.compile(r"<artifact:[^>]+>")


def bare_tool_name(raw_name: str) -> str | None:
    name = str(raw_name or "").strip()
    if name.startswith(MOLCLAW_PREFIX):
        return name[len(MOLCLAW_PREFIX) :].strip() or None
    return None


def _l1_skill_names() -> frozenset[str]:
    skill_root = Path(__file__).resolve().parents[2] / "skills" / "skills_full" / ".claude" / "skills" / "L1_tools"
    if not skill_root.is_dir():
        return frozenset()
    return frozenset(path.name for path in skill_root.iterdir() if path.is_dir())


L1_SKILL_NAMES = _l1_skill_names()


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _walk_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _walk_strings(item)]
    return []


def _canonicalize_l1_paths(value: Any) -> Any:
    if isinstance(value, str):
        match = L1_PATH_RE.search(value)
        if match:
            suffix = f"/{match.group(2)}" if match.group(2) else ""
            token_start = max(
                value.rfind(" ", 0, match.start()),
                value.rfind("\t", 0, match.start()),
                value.rfind('"', 0, match.start()),
                value.rfind("'", 0, match.start()),
            ) + 1
            return (
                value[:token_start]
                + f"skills/L1_tools/{match.group(1)}{suffix}"
                + value[match.end() :]
            )
        return value
    if isinstance(value, dict):
        return {key: _canonicalize_l1_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_l1_paths(item) for item in value]
    return value


def _skill_name(arguments: dict[str, Any]) -> str:
    for key in ("skill", "name", "skill_name"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().split("/")[-1]
    return ""


def _has_disallowed_skill_level(arguments: dict[str, Any]) -> bool:
    return any(SKILL_LEVEL_RE.search(text.replace("\\", "/")) for text in _walk_strings(arguments))


def _teacher_local_access_reason(arguments: dict[str, Any]) -> str | None:
    for text in _walk_strings(arguments):
        normalized = text.replace("\\", "/").lower()
        tokens = re.split(r"[\s|;&<>\"']+", normalized)
        for token in tokens:
            if not token:
                continue
            if Path(token).name in TEACHER_SIDECAR_NAMES:
                return "teacher_runtime_sidecar_access"
        if (
            re.search(r"(?:^|/)(?:\.claude/)?skills(?:/|$)", normalized)
            and "l1_tools" not in normalized
        ):
            return "non_l1_skill_catalog_access"
    return None


def _bash_is_safe(arguments: dict[str, Any]) -> tuple[bool, str]:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return False, "bash_missing_command"
    if _has_disallowed_skill_level(arguments):
        return False, "l2_l3_skill_access"
    if UNSAFE_BASH_RE.search(command) or any(operator in command for operator in (";", "&&", "||", "\n")):
        return False, "unsafe_bash_command"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False, "unparseable_bash_command"
    expecting_command = True
    for token in tokens:
        if token == "|":
            expecting_command = True
            continue
        if token in {"<", ">", ">>"}:
            continue
        if expecting_command:
            if token not in SAFE_BASH_COMMANDS:
                return False, f"unsupported_bash_command:{token}"
            expecting_command = False
        normalized = token.replace("\\", "/")
        if ".." in Path(normalized).parts:
            return False, "bash_path_traversal"
        if normalized.startswith(("/etc/", "/proc/", "/sys/", "/dev/", "/boot/", "/usr/", "/var/")):
            return False, "bash_path_outside_workspace"
        if token == "find" or token.startswith("-"):
            if token in {"-exec", "-execdir", "-delete", "-ok", "-okdir"}:
                return False, f"unsafe_find_option:{token}"
    return True, ""


def _classify_tool(
    raw_name: str,
    arguments: dict[str, Any],
    *,
    only_molclaw_tool: bool,
) -> tuple[str | None, str | None, str | None]:
    molclaw_name = bare_tool_name(raw_name)
    if molclaw_name:
        return molclaw_name, "molclaw", None
    if raw_name not in LOCAL_TOOL_NAMES:
        return None, None, "unsupported_teacher_tool"
    if only_molclaw_tool:
        return None, None, "only_molclaw_tool"
    if _has_disallowed_skill_level(arguments):
        return None, None, "l2_l3_skill_access"
    teacher_access_reason = _teacher_local_access_reason(arguments)
    if teacher_access_reason:
        return None, None, teacher_access_reason
    if raw_name == "Skill" and _skill_name(arguments) not in L1_SKILL_NAMES:
        return None, None, "non_l1_skill"
    if raw_name == "Bash":
        safe, reason = _bash_is_safe(arguments)
        if not safe:
            return None, None, reason
    return raw_name, "local", None


def _parse_content(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _compact_observation(value: Any, max_chars: int) -> tuple[Any, bool, int]:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    original_chars = len(serialized)
    if original_chars <= max_chars:
        return value, False, original_chars
    if isinstance(value, dict):
        summary = {key: value[key] for key in IMPORTANT_OBSERVATION_KEYS if key in value}
        omitted = sorted(str(key) for key in value if key not in summary)
        compacted: dict[str, Any] = {
            "compacted": True,
            "original_size_chars": original_chars,
            "summary": summary,
        }
        if omitted:
            compacted["omitted_keys"] = omitted[:50]
        return compacted, True, original_chars
    if isinstance(value, list):
        return {
            "compacted": True,
            "original_size_chars": original_chars,
            "item_count": len(value),
            "items_preview": value[:3],
        }, True, original_chars
    return {
        "compacted": True,
        "original_size_chars": original_chars,
        "text_preview": str(value)[: min(max_chars, 1000)],
    }, True, original_chars


def _content_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, str) and content.strip():
        return [{"type": "text", "text": content.strip()}]
    return []


def paired_tool_result_ids(events: list[dict[str, Any]]) -> set[str]:
    """Return raw tool-use IDs that have a recorded tool-result event."""
    return {
        str(item.get("tool_use_id") or "")
        for event in events
        if str(event.get("type") or "").strip().lower() == "user"
        for item in _content_items(event)
        if str(item.get("type") or "") == "tool_result"
        and str(item.get("tool_use_id") or "")
    }


def retainable_molclaw_call_count(events: list[dict[str, Any]]) -> int:
    """Count MolClaw calls that can be retained as a call/observation pair."""
    observed_ids = paired_tool_result_ids(events)
    return sum(
        1
        for event in events
        if str(event.get("type") or "").strip().lower() == "assistant"
        for item in _content_items(event)
        if str(item.get("type") or "") == "tool_use"
        and bare_tool_name(str(item.get("name") or ""))
        and str(item.get("id") or "") in observed_ids
    )


def _raw_system_prompt(events: list[dict[str, Any]]) -> str:
    for event in events:
        if str(event.get("type") or "").lower() != "system":
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(
                str(item.get("text") or item.get("thinking") or "").strip()
                for item in content
                if isinstance(item, dict)
            ).strip()
        else:
            text = ""
        if all(tag in text for tag in ("<thought>", "<tool_call>", "<observation>", "<final_answer>")):
            return text
    return ""


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _answer_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        for key in ("ranking", "ranked_smiles", "selected_smiles", "answer_smiles", "answer", "result"):
            if key in value:
                values = _answer_values(value[key])
                if values:
                    return values
    return []


def _evidence_summary(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for observation in observations[-8:]:
        content = observation.get("content")
        item: dict[str, Any] = {
            "tool_name": observation["tool_name"],
            "status": observation["status"],
        }
        if isinstance(content, dict):
            key_values = {
                key: content[key]
                for key in ("summary", "message", "msg", "score", "scores", "value", "count", "artifact", "output_path")
                if key in content
            }
            if key_values:
                item["key_values"] = key_values
        evidence.append(item)
    return evidence


def _artifact_refs(value: Any) -> list[str]:
    return ARTIFACT_REF_RE.findall(_serialize(value))


def _values_for_key(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for item_key, item in value.items():
            if str(item_key) == key:
                values.append(item)
            values.extend(_values_for_key(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_values_for_key(item, key))
    return values


def _primary_output_artifact(observations: list[dict[str, Any]]) -> str | None:
    preferred: list[str] = []
    fallback: list[str] = []
    for observation in observations:
        if observation.get("is_error"):
            continue
        content = observation.get("content")
        fallback.extend(_artifact_refs(content))
        for key in OUTPUT_ARTIFACT_KEYS:
            for value in _values_for_key(content, key):
                preferred.extend(_artifact_refs(value))
    candidates = preferred or fallback
    return candidates[-1] if candidates else None


def _is_file_delivery_answer(value: Any) -> bool:
    return bool(DELIVERABLE_FILE_RE.search(_serialize(value)))


def _replace_ranking(value: Any, ranking: list[str]) -> Any:
    if isinstance(value, list):
        return list(ranking)
    if not isinstance(value, dict):
        return value
    updated = deepcopy(value)
    for key in ("ranking", "ranked_smiles", "answer", "result"):
        if key in updated and _answer_values(updated[key]):
            updated[key] = list(ranking)
            break
    if "selected_smiles" in updated:
        updated["selected_smiles"] = ranking[0] if ranking else ""
    return updated


def _quickvina_context(arguments: dict[str, Any]) -> str:
    context = {
        key: value
        for key, value in arguments.items()
        if key not in {"smiles", "ligand_smiles", "molecule", "compound"}
    }
    return _serialize(context)


def _repair_vs_ranking(
    final_answer: Any,
    quickvina_results: list[dict[str, Any]],
    *,
    higher_order_ranking_seen: bool,
) -> tuple[Any, dict[str, Any]]:
    ranking = _answer_values(final_answer)
    audit: dict[str, Any] = {
        "status": "not_needed",
        "original_ranking": ranking,
    }
    if len(ranking) < 2:
        audit["reason"] = "ranking_has_fewer_than_two_entries"
        return final_answer, audit
    if higher_order_ranking_seen:
        audit.update(status="skipped", reason="higher_order_ranking_evidence_present")
        return final_answer, audit

    by_smiles: dict[str, list[dict[str, Any]]] = {}
    for result in quickvina_results:
        smiles = result.get("smiles")
        score = result.get("score")
        if isinstance(smiles, str) and smiles and isinstance(score, (int, float)):
            by_smiles.setdefault(smiles, []).append(result)

    selected: list[dict[str, Any]] = []
    for smiles in ranking:
        records = by_smiles.get(smiles, [])
        scores = {float(record["score"]) for record in records}
        contexts = {str(record["context"]) for record in records}
        if not records:
            audit.update(status="skipped", reason=f"missing_successful_quickvina_score:{smiles}")
            return final_answer, audit
        if len(scores) != 1:
            audit.update(status="skipped", reason=f"conflicting_quickvina_scores:{smiles}")
            return final_answer, audit
        if len(contexts) != 1:
            audit.update(status="skipped", reason=f"conflicting_quickvina_contexts:{smiles}")
            return final_answer, audit
        selected.append(records[-1])

    if len({str(record["context"]) for record in selected}) != 1:
        audit.update(status="skipped", reason="ranking_uses_different_quickvina_contexts")
        return final_answer, audit

    scores = {smiles: float(record["score"]) for smiles, record in zip(ranking, selected)}
    repaired = sorted(ranking, key=lambda smiles: scores[smiles])
    audit["scores"] = scores
    audit["repaired_ranking"] = repaired
    if repaired == ranking:
        audit["reason"] = "already_matches_quickvina_scores"
        return final_answer, audit
    audit.update(status="repaired", reason="sorted_by_quickvina_affinity_ascending")
    return _replace_ranking(final_answer, repaired), audit


def build_final_payload(
    task: str,
    final_answer: Any,
    final_text: str,
    observations: list[dict[str, Any]],
    *,
    primary_artifact_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = final_answer
    if value in (None, "", []):
        value = final_text.strip()
    values = _answer_values(value)
    summary = final_text.strip()
    evidence = _evidence_summary(observations)
    if task == "vs":
        payload: dict[str, Any] = {
            "task_type": "vs",
            "ranked_smiles": values,
            "selected_smiles": values[0] if values else "",
        }
    elif task == "ac":
        payload = {"task_type": "ac", "answer_smiles": values[0] if values else ""}
    elif task == "pf":
        payload = {"task_type": "pf", "selected_smiles": values}
    else:
        artifact_observations = (
            observations if primary_artifact_observations is None else primary_artifact_observations
        )
        primary_artifact = _primary_output_artifact(artifact_observations)
        result = (
            primary_artifact
            if task in {"kg", "e2e"} and primary_artifact and _is_file_delivery_answer(value)
            else value
        )
        payload = {"task_type": task, "result": result}
    if summary:
        payload["summary"] = summary
    if evidence:
        payload["evidence"] = evidence
    return payload


def reconstruct_react_messages(
    events: list[dict[str, Any]],
    *,
    question_text: str,
    final_answer: Any,
    task: str = "kg",
    max_observation_chars: int = 6000,
    only_molclaw_tool: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path_map: dict[str, str] = {}
    paired_result_ids = paired_tool_result_ids(events)
    system_prompt = _raw_system_prompt(events) or (
        MOLCLAW_ONLY_SYSTEM_PROMPT if only_molclaw_tool else CANONICAL_SYSTEM_PROMPT
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sanitize_artifact_paths(system_prompt, path_map), "step_loss_mask": 0},
        {"role": "user", "content": sanitize_artifact_paths(question_text, path_map), "step_loss_mask": 0},
    ]
    retained_calls: dict[str, dict[str, Any]] = {}
    retained_call_order: dict[str, int] = {}
    observed_calls: set[str] = set()
    tool_hist: Counter[str] = Counter()
    local_tool_hist: Counter[str] = Counter()
    observations_for_final: list[dict[str, Any]] = []
    molclaw_observations_for_final: list[dict[str, Any]] = []
    raw_tool_names: dict[str, str] = {}
    dropped_non_molclaw_calls = 0
    dropped_tool_hist: Counter[str] = Counter()
    dropped_tool_calls: list[dict[str, Any]] = []
    dropped_call_ids: set[str] = set()
    dropped_observation_count = 0
    orphan_observations = 0
    reordered_observations = 0
    compacted_observations = 0
    error_status_conflicts: list[dict[str, Any]] = []
    assistant_text_after_last_observation: list[str] = []
    only_molclaw_repair_hints: list[dict[str, Any]] = []
    quickvina_results: list[dict[str, Any]] = []
    higher_order_ranking_seen = False
    buffered_observations: list[dict[str, Any]] = []

    def append_turn(role: str, content: str, loss_mask: int) -> None:
        text = content.strip()
        if not text:
            return
        if len(messages) > 2 and messages[-1]["role"] == role:
            messages[-1]["content"] = f"{messages[-1]['content']}\n{text}"
        else:
            messages.append({"role": role, "content": text, "step_loss_mask": loss_mask})

    def flush_observations() -> None:
        nonlocal reordered_observations, assistant_text_after_last_observation
        if not buffered_observations:
            return
        original_ids = [entry["call_id"] for entry in buffered_observations]
        ordered = sorted(buffered_observations, key=lambda entry: entry["call_order"])
        ordered_ids = [entry["call_id"] for entry in ordered]
        reordered_observations += sum(
            before != after for before, after in zip(original_ids, ordered_ids)
        )
        for entry in ordered:
            observation = entry["observation"]
            observations_for_final.append(observation)
            if entry["tool_kind"] == "molclaw":
                molclaw_observations_for_final.append(observation)
        append_turn(
            "user",
            "\n".join(entry["rendered"] for entry in ordered),
            0,
        )
        buffered_observations.clear()
        assistant_text_after_last_observation = []

    for event_index, event in enumerate(events):
        event_type = str(event.get("type") or "").lower()
        if event_type == "assistant":
            flush_observations()
            rendered: list[str] = []
            event_texts: list[str] = []
            for item in _content_items(event):
                item_type = str(item.get("type") or "")
                if item_type in {"thinking", "text"}:
                    text = item.get("thinking") if item_type == "thinking" else item.get("text")
                    if isinstance(text, str) and text.strip():
                        clean = sanitize_artifact_paths(text.strip(), path_map)
                        event_texts.append(clean)
                        rendered.append(f"<thought>{clean}</thought>")
                elif item_type == "tool_use":
                    raw_name = str(item.get("name") or "")
                    call_id = str(item.get("id") or f"event-{event_index}-call-{len(retained_calls)}")
                    arguments = item.get("input") if isinstance(item.get("input"), dict) else {}
                    name, tool_kind, drop_reason = _classify_tool(
                        raw_name,
                        arguments,
                        only_molclaw_tool=only_molclaw_tool,
                    )
                    if not name:
                        dropped_non_molclaw_calls += 1
                        dropped_call_ids.add(call_id)
                        dropped_tool_hist[drop_reason or "unsupported"] += 1
                        dropped_tool_calls.append(
                            {
                                "event_index": event_index,
                                "tool_use_id": call_id,
                                "raw_tool_name": raw_name,
                                "reason": drop_reason or "unsupported",
                            }
                        )
                        if only_molclaw_tool and drop_reason == "only_molclaw_tool" and event_texts:
                            only_molclaw_repair_hints.append(
                                {
                                    "event_index": event_index,
                                    "removed_tool": raw_name,
                                    "reason": "remove narration that only describes the removed local tool; preserve scientific reasoning",
                                }
                            )
                        continue
                    if call_id not in paired_result_ids:
                        dropped_call_ids.add(call_id)
                        dropped_tool_hist["missing_paired_observation"] += 1
                        dropped_tool_calls.append(
                            {
                                "event_index": event_index,
                                "tool_use_id": call_id,
                                "raw_tool_name": raw_name,
                                "reason": "missing_paired_observation",
                            }
                        )
                        if tool_kind == "local":
                            dropped_non_molclaw_calls += 1
                        continue
                    arguments = _canonicalize_l1_paths(arguments)
                    arguments = sanitize_artifact_paths(arguments, path_map)
                    retained_calls[call_id] = {
                        "name": name,
                        "raw_name": raw_name,
                        "tool_kind": tool_kind,
                        "arguments": arguments,
                        "is_l1_read": raw_name == "Read" and any(
                            L1_PATH_RE.search(text.replace("\\", "/"))
                            for text in _walk_strings(item.get("input") or {})
                        ),
                    }
                    retained_call_order[call_id] = len(retained_call_order)
                    raw_tool_names[raw_name] = name
                    tool_hist[name] += 1
                    if tool_kind == "local":
                        local_tool_hist[name] += 1
                    rendered.append(
                        f"<tool_call>{_serialize({'tool_name': name, 'arguments': arguments})}</tool_call>"
                    )
            if rendered:
                append_turn("assistant", "\n".join(rendered), 1)
            if event_texts and retained_calls:
                assistant_text_after_last_observation.extend(event_texts)
        elif event_type == "user":
            result_items = [
                item
                for item in _content_items(event)
                if str(item.get("type") or "") == "tool_result"
            ]
            for item in result_items:
                call_id = str(item.get("tool_use_id") or "")
                call = retained_calls.get(call_id)
                if not call:
                    if call_id in dropped_call_ids:
                        dropped_observation_count += 1
                    else:
                        orphan_observations += 1
                    continue
                parsed = sanitize_artifact_paths(_parse_content(item.get("content")), path_map)
                raw_is_error = bool(item.get("is_error"))
                status_inspection = inspect_observation_status(parsed, event_is_error=raw_is_error)
                if status_inspection["conflict"]:
                    error_status_conflicts.append(
                        {
                            "event_index": event_index,
                            "tool_use_id": call_id,
                            "tool_name": call["name"],
                            "event_is_error": raw_is_error,
                            "content_status": status_inspection["content_status"],
                        }
                    )
                if call["is_l1_read"]:
                    compacted, changed, original_chars = parsed, False, len(_serialize(parsed))
                else:
                    compacted, changed, original_chars = _compact_observation(parsed, max_observation_chars)
                compacted_observations += int(changed)
                observation = {
                    "tool_name": call["name"],
                    "status": status_inspection["status"],
                    "is_error": status_inspection["is_error"],
                    "content": compacted,
                }
                if changed:
                    observation["compaction"] = {
                        "method": "structured_summary",
                        "original_size_chars": original_chars,
                    }
                if call["tool_kind"] == "molclaw":
                    if call["name"] == QUICKVINA_TOOL and not observation["is_error"]:
                        score_values = _values_for_key(parsed, "docking_affinity_value")
                        smiles = call["arguments"].get("smiles")
                        try:
                            score = float(score_values[-1]) if score_values and score_values[-1] is not None else None
                        except (TypeError, ValueError):
                            score = None
                        if isinstance(smiles, str) and score is not None:
                            quickvina_results.append(
                                {
                                    "tool_use_id": call_id,
                                    "smiles": smiles,
                                    "score": score,
                                    "context": _quickvina_context(call["arguments"]),
                                }
                            )
                    if any(marker in call["name"].lower() for marker in HIGHER_ORDER_RANKING_MARKERS):
                        if not observation["is_error"]:
                            higher_order_ranking_seen = True
                observed_calls.add(call_id)
                buffered_observations.append(
                    {
                        "call_id": call_id,
                        "call_order": retained_call_order[call_id],
                        "tool_kind": call["tool_kind"],
                        "observation": observation,
                        "rendered": (
                            f'<observation tool_name="{call["name"]}">'
                            f"{_serialize(observation)}</observation>"
                        ),
                    }
                )
        elif event_type == "result":
            flush_observations()
            result_text = str(event.get("result") or "").strip()
            if result_text:
                assistant_text_after_last_observation.append(sanitize_artifact_paths(result_text, path_map))

    flush_observations()
    final_text = "\n\n".join(assistant_text_after_last_observation).strip()
    resolved_final_answer = final_answer
    vs_ranking_repair: dict[str, Any] = {"status": "not_applicable", "reason": "task_is_not_vs"}
    if task == "vs":
        resolved_final_answer, vs_ranking_repair = _repair_vs_ranking(
            final_answer,
            quickvina_results,
            higher_order_ranking_seen=higher_order_ranking_seen,
        )
    payload = sanitize_artifact_paths(
        build_final_payload(
            task,
            resolved_final_answer,
            final_text,
            observations_for_final,
            primary_artifact_observations=molclaw_observations_for_final,
        ),
        path_map,
    )
    messages.append(
        {
            "role": "assistant",
            "content": f"<final_answer>{_serialize(payload)}</final_answer>",
            "step_loss_mask": 1,
        }
    )
    messages = sanitize_artifact_paths(messages, path_map)
    if resolved_final_answer in (None, "", []):
        resolved_final_answer = payload.get("result") or payload.get("ranked_smiles") or payload.get("answer_smiles")
    return messages, {
        "molclaw_usage_count": sum(
            1 for call in retained_calls.values() if call["tool_kind"] == "molclaw"
        ),
        "molclaw_usage_computation_count": 1,
        "tool_name_hist": dict(tool_hist),
        "local_tool_name_hist": dict(local_tool_hist),
        "retained_local_tool_call_count": sum(local_tool_hist.values()),
        "observed_tool_call_count": len(observed_calls),
        "missing_observation_count": len(set(retained_calls) - observed_calls),
        "dropped_non_molclaw_call_count": dropped_non_molclaw_calls,
        "dropped_tool_reason_hist": dict(dropped_tool_hist),
        "dropped_tool_calls": dropped_tool_calls,
        "dropped_observation_count": dropped_observation_count,
        "orphan_observation_count": orphan_observations,
        "reordered_observation_count": reordered_observations,
        "compacted_observation_count": compacted_observations,
        "error_status_conflicts": error_status_conflicts,
        "raw_tool_name_map": raw_tool_names,
        "artifact_mappings": path_map,
        "used_raw_system_prompt": bool(_raw_system_prompt(events)),
        "only_molclaw_tool": only_molclaw_tool,
        "only_molclaw_repair_hints": only_molclaw_repair_hints,
        "dropped_local_tool_context": only_molclaw_repair_hints,
        "vs_ranking_repair": vs_ranking_repair,
        "resolved_final_answer": resolved_final_answer,
        "final_payload": payload,
    }
