#!/usr/bin/env python3
"""Build a v7 source release with one unambiguous path/resource contract."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from drug_agent.protocol.toolrl_turn import serialize_decision, split_assistant_segments

LEGACY_RE = re.compile(r"<artifact:([A-Za-z0-9._-]+)/([^>]+)>")
RESOURCE_RE = re.compile(r"resource://[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+")
MACHINE_PATH_RE = re.compile(r"(?<![A-Za-z0-9._-])/(?:tmp|root|home)/(?:[A-Za-z0-9._@+:-]+/?)+")
LOCAL_TOOLS = {"Read", "Write", "Edit", "Bash", "Grep", "Glob"}
PATH_KEYS = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "Grep": ("path",),
    "Glob": ("path",),
}
PUNCTUATION = {"|", ">", ">>"}


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    return [
        call
        for segment in split_assistant_segments(content)
        for call in segment["tool_calls"]
        if isinstance(call, dict)
    ]


def _safe_tail(value: str) -> str:
    path = PurePosixPath(value.strip().lstrip("/"))
    if not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe path reference: {value!r}")
    return path.as_posix()


def workspace_path(value: str) -> str:
    value = value.strip()
    if value in {"workspace", "."}:
        return "workspace"
    if value.startswith("workspace/"):
        return f"workspace/{_safe_tail(value[len('workspace/'):])}"
    if value.startswith("skills/"):
        return value
    if value.startswith("L1_tools/"):
        return f"skills/{value}"
    return f"workspace/{_safe_tail(value)}"


def normalize_refs(text: str) -> str:
    def replace_legacy(match: re.Match[str]) -> str:
        namespace, name = match.groups()
        if namespace == "local":
            return workspace_path(name)
        return f"resource://{namespace}/{_safe_tail(name)}"

    def replace_machine(match: re.Match[str]) -> str:
        name = PurePosixPath(match.group(0).rstrip("/")).name or "file"
        return f"resource://unavailable/{name}"

    return MACHINE_PATH_RE.sub(replace_machine, LEGACY_RE.sub(replace_legacy, text))


def _looks_like_path(token: str) -> bool:
    return (
        token in {".", ".."}
        or token.startswith(("./", "../", "workspace/", "skills/", "L1_tools/"))
        or "/" in token
        or any(char in token for char in "*?[]")
        or bool(Path(token).suffix)
    )


def _normalize_shell_path(token: str) -> str:
    if token.startswith("skills/"):
        return token
    if token.startswith("L1_tools/"):
        return f"skills/{token}"
    return workspace_path(token)


def normalize_bash(command: str) -> str:
    if LEGACY_RE.search(command) or RESOURCE_RE.search(command):
        raise ValueError("Bash command contains a resource reference")
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = list(lexer)
    normalized: list[str] = []
    command_name: str | None = None
    positional = 0
    skip_next = False
    redirect_next = False
    for token in tokens:
        if token in PUNCTUATION:
            normalized.append(token)
            command_name = None if token == "|" else command_name
            redirect_next = token in {">", ">>"}
            positional = 0 if token == "|" else positional
            continue
        if redirect_next:
            normalized.append(_normalize_shell_path(token))
            redirect_next = False
            continue
        if command_name is None:
            command_name = token
            normalized.append(token)
            continue
        if skip_next:
            normalized.append(token)
            skip_next = False
            continue
        if token in {"-n", "-c", "-w", "--max-count", "--include", "--exclude"}:
            normalized.append(token)
            skip_next = True
            continue
        if token.startswith("-"):
            normalized.append(token)
            continue
        use_path = False
        if command_name in {"ls", "cat", "wc", "mkdir", "cp", "mv", "rm", "realpath", "readlink", "test", "cd", "find", "stat", "base64"}:
            use_path = _looks_like_path(token) or command_name in {"cp", "mv", "cd", "find"}
        elif command_name in {"head", "tail"}:
            use_path = _looks_like_path(token) and not token.isdigit()
        elif command_name in {"grep", "rg"}:
            use_path = positional >= 1 and _looks_like_path(token)
        normalized.append(_normalize_shell_path(token) if use_path else token)
        positional += 1
    return " ".join(token if token in PUNCTUATION else shlex.quote(token) for token in normalized)


def _local_call_violation(call: dict[str, Any]) -> str | None:
    name = call.get("tool_name") or call.get("name")
    args = call.get("arguments") or {}
    if name not in LOCAL_TOOLS or not isinstance(args, dict):
        return None
    values: list[str] = []
    if name == "Bash":
        if isinstance(args.get("command"), str):
            values.append(args["command"])
    else:
        for key in PATH_KEYS.get(name, ()):
            if isinstance(args.get(key), str):
                values.append(args[key])
    for value in values:
        for match in LEGACY_RE.finditer(value):
            if match.group(1) != "local":
                return "server_resource_used_as_filesystem_path"
        if RESOURCE_RE.search(value):
            return "resource_used_as_filesystem_path"
        if MACHINE_PATH_RE.search(value):
            return "machine_absolute_path"
    return None


def _normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    call = copy.deepcopy(call)
    name = str(call.get("tool_name") or call.get("name") or "")
    args = call.get("arguments")
    if not isinstance(args, dict):
        return call
    args = _normalize_value(args)
    if name == "Bash" and isinstance(args.get("command"), str) and not RESOURCE_RE.search(args["command"]):
        args["command"] = normalize_bash(args["command"])
    for key in PATH_KEYS.get(name, ()):
        if isinstance(args.get(key), str) and not RESOURCE_RE.fullmatch(args[key]):
            args[key] = workspace_path(args[key])
    return {"tool_name": name, "arguments": args}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_refs(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    return value


def normalize_assistant_content(content: str) -> str:
    output: list[str] = []
    for segment in split_assistant_segments(content):
        thoughts = [normalize_refs(str(item)) for item in segment["thoughts"]]
        if segment["tool_calls"]:
            output.append(
                serialize_decision(
                    thoughts=thoughts,
                    tool_calls=[_normalize_call(call) for call in segment["tool_calls"]],
                )
            )
        elif segment["final_answer"] is not None:
            output.append(
                serialize_decision(
                    thoughts=thoughts,
                    final_answer=_normalize_value(segment["final_answer"]),
                )
            )
        elif thoughts:
            output.append("<thought>" + "\n\n".join(thoughts) + "</thought>")
    return "\n".join(output)


def normalize_record(record: dict[str, Any], counters: Counter[str]) -> dict[str, Any]:
    result = copy.deepcopy(record)
    output: list[dict[str, Any]] = []
    messages = result.get("messages", [])
    index = 0
    while index < len(messages):
        message = messages[index]
        violations: list[tuple[dict[str, Any], str]] = []
        if message.get("role") == "assistant":
            violations = [
                (call, reason)
                for call in parse_tool_calls(str(message.get("content", "")))
                if (reason := _local_call_violation(call))
            ]
        normalized = copy.deepcopy(message)
        content = str(normalized.get("content", ""))
        normalized["content"] = normalize_assistant_content(content) if normalized.get("role") == "assistant" else normalize_refs(content)
        if violations:
            normalized["step_loss_mask"] = 0
            normalized["path_contract_supervision_masked"] = True
            counters["masked_assistant_actions"] += 1
            counters["masked_violating_calls"] += len(violations)
            for call, reason in violations:
                counters[f"masked_reason:{reason}"] += 1
                counters[f"masked_tool:{call.get('tool_name') or call.get('name') or 'unknown'}"] += 1
            if index + 1 < len(messages) and messages[index + 1].get("role") == "user":
                counters["preserved_paired_observations"] += 1
        output.append(normalized)
        index += 1
    result["messages"] = output
    metadata = result.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["path_reference_contract"] = "workspace-path-or-resource-uri-v1"
    return result


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def collect_transitions(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages", [])
    found: list[dict[str, Any]] = []
    for index in range(len(messages) - 1):
        observation = messages[index]
        action = messages[index + 1]
        if observation.get("role") != "user" or action.get("role") != "assistant":
            continue
        obs_text = str(observation.get("content", ""))
        refs = {match.group(0) for match in LEGACY_RE.finditer(obs_text)}
        refs.update(RESOURCE_RE.findall(obs_text))
        refs.update(re.findall(r"workspace/[A-Za-z0-9._/-]+", obs_text))
        action_text = str(action.get("content", ""))
        reused = sorted(ref for ref in refs if ref in action_text)
        if reused:
            found.append({"message_index": index, "references": reused})
    return found


def annotate_catalog(catalog: Any, resource_args: set[tuple[str, str]]) -> Any:
    result = copy.deepcopy(catalog)
    tools = result.get("tools", []) if isinstance(result, dict) else result
    if not isinstance(tools, list):
        return result
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(function.get("name", ""))
        schema = function.get("parameters") or function.get("input_schema")
        if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
            continue
        for key, prop in schema["properties"].items():
            if not isinstance(prop, dict):
                continue
            if name in LOCAL_TOOLS and (key in PATH_KEYS.get(name, ()) or (name == "Bash" and key == "command")):
                prop["description"] = "Filesystem input. Use a workspace/... path supplied by the runtime; resource:// handles are not accepted."
            elif (name, str(key)) in resource_args:
                prop["description"] = "Server resource input. Reuse the resource:// handle supplied by a prior observation."
    return result


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    local_violations: list[dict[str, Any]] = []
    masked_local_violations: list[dict[str, Any]] = []
    bad_masks = 0
    for row, record in enumerate(records):
        for message in record.get("messages", []):
            if message.get("role") == "user" and message.get("step_loss_mask", 0) != 0:
                bad_masks += 1
            if message.get("role") == "assistant":
                for call in parse_tool_calls(str(message.get("content", ""))):
                    if reason := _local_call_violation(call):
                        item = {"row": row, "tool": call.get("tool_name") or call.get("name"), "reason": reason}
                        if message.get("step_loss_mask") == 0 and message.get("path_contract_supervision_masked") is True:
                            masked_local_violations.append(item)
                        else:
                            local_violations.append(item)
    return {
        "records": len(records),
        "legacy_artifact_literals": len(LEGACY_RE.findall(serialized)),
        "resource_uri_literals": len(RESOURCE_RE.findall(serialized)),
        "workspace_path_literals": len(re.findall(r"workspace/[A-Za-z0-9._/-]+", serialized)),
        "filesystem_action_violations": local_violations,
        "masked_filesystem_action_violation_count": len(masked_local_violations),
        "observation_nonzero_loss_masks": bad_masks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
        args.output_root.mkdir(parents=True, exist_ok=True)
    source_records = [json.loads(line) for line in (args.input_root / "react_trajectories.jsonl").read_text().splitlines() if line.strip()]

    resource_args: set[tuple[str, str]] = set()
    for record in source_records:
        for message in record.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for call in parse_tool_calls(str(message.get("content", ""))):
                arguments = call.get("arguments") or {}
                if not isinstance(arguments, dict):
                    continue
                for key, value in arguments.items():
                    if any(
                        any(match.group(1) != "local" for match in LEGACY_RE.finditer(text))
                        for text in _iter_strings(value)
                    ):
                        resource_args.add((str(call.get("tool_name") or call.get("name") or ""), str(key)))

    counters: Counter[str] = Counter()
    normalized_records = [normalize_record(record, counters) for record in source_records]
    audit = audit_records(normalized_records)
    if audit["legacy_artifact_literals"] or audit["filesystem_action_violations"] or audit["observation_nonzero_loss_masks"]:
        raise SystemExit(f"path contract audit failed: {audit}")

    samples: list[dict[str, Any]] = []
    for row, (before, after) in enumerate(zip(source_records, normalized_records)):
        before_transitions = collect_transitions(before)
        after_transitions = collect_transitions(after)
        for before_transition in before_transitions:
            expected = sorted(normalize_refs(ref) for ref in before_transition["references"])
            matched = next(
                (
                    after_transition
                    for after_transition in after_transitions
                    if sorted(after_transition["references"]) == expected
                ),
                None,
            )
            if matched is not None:
                samples.append({"row": row, "before": before_transition, "after": matched})
                break
        if len(samples) == 20:
            break
    if len(samples) < 20:
        raise SystemExit(f"only {len(samples)} preserved observation-to-action transitions; expected at least 20")

    catalog = json.loads((args.input_root / "tool_catalog.json").read_text())
    report = {
        "dataset_version": args.dataset_version,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "before": {
            "records": len(source_records),
            "legacy_artifact_literals": sum(len(LEGACY_RE.findall(text)) for record in source_records for text in _iter_strings(record)),
        },
        "after": audit,
        "transform": dict(sorted(counters.items())),
        "resource_argument_contracts": sorted([{"tool": tool, "argument": argument} for tool, argument in resource_args], key=lambda item: (item["tool"], item["argument"])),
        "preserved_transition_samples": samples,
    }
    if not args.dry_run:
        with (args.output_root / "react_trajectories.jsonl").open("w") as handle:
            for record in normalized_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        (args.output_root / "tool_catalog.json").write_text(json.dumps(annotate_catalog(catalog, resource_args), ensure_ascii=False, indent=2) + "\n")
        (args.output_root / "path_contract_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (args.output_root / "DATASET_VERSION").write_text(args.dataset_version + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
