#!/usr/bin/env python3
"""Run MolBench MS-1/MS-2 through a persistent DSH Web host.

Each benchmark item gets an independent DSH project/workspace.  The runner
adapts the existing Claude Code MolClaw skill tree into one DSH-native entry
skill whose complete hierarchy remains available as on-demand resources.
Ground-truth answers are never written into task workspaces.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websocket


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DRUG_PIPE_ROOT = WORKSPACE_ROOT.parent
DEFAULT_MOLBENCH_ROOT = WORKSPACE_ROOT / "molbench"
DEFAULT_DSH_REPO = WORKSPACE_ROOT / "deepseek-harness"
DEFAULT_SKILL_SOURCE = DRUG_PIPE_ROOT / "workdir-skills/molclaw-trajectory-execution"
DEFAULT_RUNS_ROOT = WORKSPACE_ROOT / "outputs/dsh_molbench_evals"
DEFAULT_DSH_URL = "http://127.0.0.1:3080"
DEFAULT_AGENT_PRESET = "standard"
TASK_MARKER = "DSH_CANONICAL_REACT_TASK_V1\n"

MS1_CSV = Path("data/molbench-ms-1/molbench-ms-1.csv")
MS2_CSV = Path("data/molbench-ms-2/molbench-ms-2.csv")
MS2_EXCLUSIONS = Path("slime/drug_agent/evaluation/molbench_exclusions.json")


@dataclass(frozen=True)
class Sample:
    task_id: str
    suite: str
    source_row: int
    prompt: str
    answer: str
    target: str = ""
    molecule_a: str = ""
    molecule_b: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalized_prompt(text).encode("utf-8")).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_ms1_candidates(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "SMILES:") + 1
        end = next(
            index for index, line in enumerate(lines[start:], start)
            if line.strip() in {"Constraints:", "Output format:"}
        )
    except StopIteration:
        raise ValueError("MS-1 prompt does not contain a SMILES/Constraints block")
    candidates = [line.strip() for line in lines[start:end] if line.strip()]
    if not candidates:
        raise ValueError("MS-1 prompt contains an empty SMILES block")
    return candidates


def extract_ms2_pair(prompt: str) -> tuple[str, str]:
    match_a = re.search(r"Molecule A:\s*([^\n]+)", prompt)
    match_b = re.search(r"Molecule B:\s*([^\n]+)", prompt)
    if not match_a or not match_b:
        raise ValueError("MS-2 prompt does not contain both molecule choices")
    return match_a.group(1).strip(), match_b.group(1).strip()


def load_samples(molbench_root: Path, suites: set[str], limit_per_suite: int) -> list[Sample]:
    if limit_per_suite < 0:
        raise ValueError("--limit-per-suite must be non-negative")
    samples: list[Sample] = []
    if "ms1" in suites:
        rows = csv_rows(molbench_root / MS1_CSV)
        if len(rows) != 50:
            raise ValueError(f"expected 50 MS-1 rows, found {len(rows)}")
        for index, row in enumerate(rows, 1):
            prompt = row.get("prompt") or row.get("\ufeffprompt") or ""
            extract_ms1_candidates(prompt)
            samples.append(Sample(
                task_id=f"molbench_ms1_{index:03d}", suite="ms1", source_row=index,
                prompt=prompt, answer=row.get("answer", ""),
            ))

    if "ms2" in suites:
        exclusion_path = WORKSPACE_ROOT / MS2_EXCLUSIONS
        exclusion_data = json.loads(exclusion_path.read_text(encoding="utf-8"))
        excluded = set(exclusion_data["molbench_ms2_prompt_sha256"])
        rows = csv_rows(molbench_root / MS2_CSV)
        if len(rows) != 37:
            raise ValueError(f"expected 37 raw MS-2 rows, found {len(rows)}")
        excluded_count = 0
        for index, row in enumerate(rows, 1):
            prompt = row.get("question") or row.get("\ufeffquestion") or ""
            if prompt_hash(prompt) in excluded:
                excluded_count += 1
                continue
            molecule_a, molecule_b = extract_ms2_pair(prompt)
            samples.append(Sample(
                task_id=f"molbench_ms2_{index:03d}", suite="ms2", source_row=index,
                prompt=prompt, answer=row.get("answer", ""), target=row.get("target", ""),
                molecule_a=molecule_a, molecule_b=molecule_b,
            ))
        if excluded_count != 4:
            raise ValueError(f"expected four MS-2 overlap exclusions, found {excluded_count}")

    if limit_per_suite:
        selected: list[Sample] = []
        for suite in ("ms1", "ms2"):
            selected.extend([sample for sample in samples if sample.suite == suite][:limit_per_suite])
        samples = selected
    return samples


def adapt_skill_text(text: str) -> str:
    """Translate Claude-specific paths/session wording without changing science."""
    text = text.replace("auto-generated-.claude/skills/", "auto-generated-skills/")
    text = text.replace(".claude/skills/", ".dsh/skills/execute-molclaw-trajectory/resources/")
    text = text.replace(".claude/skills", ".dsh/skills/execute-molclaw-trajectory/resources")
    text = text.replace("raw Claude session", "raw DSH session")
    text = text.replace("Claude session", "DSH session")
    return text


def create_skill_snapshot(run_dir: Path, skill_source: Path, skill_visibility: str) -> Path:
    snapshot = run_dir / "skill_snapshot"
    bundle = snapshot / "dsh-bundle"
    resources = snapshot / "resources"
    manifest_path = snapshot / "manifest.json"
    source_digest = tree_digest(skill_source)

    if manifest_path.is_file() and bundle.is_dir() and resources.is_dir():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != source_digest:
            raise RuntimeError("skill source changed after this run was prepared; use a new run directory")
        if manifest.get("skill_visibility", "hierarchy") != skill_visibility:
            raise RuntimeError("skill visibility changed after this run was prepared; use a new run directory")
        return bundle

    if snapshot.exists():
        raise RuntimeError(f"incomplete skill snapshot already exists: {snapshot}")
    snapshot.mkdir(parents=True)
    source_tree = skill_source / ".claude/skills"
    source_entry = source_tree / "execute-molclaw-trajectory"
    if not (source_entry / "SKILL.md").is_file():
        raise FileNotFoundError(source_entry / "SKILL.md")

    if skill_visibility == "l1-only":
        shutil.copytree(source_tree / "L1_tools", resources / "L1_tools")
        bundle.mkdir()
        write_text(
            bundle / "SKILL.md",
            "---\n"
            "name: execute-molclaw-trajectory\n"
            "description: Execute one evidence-grounded MolClaw task using on-demand L1 tool documentation.\n"
            "---\n\n"
            "# Execute a MolClaw trajectory\n\n"
            "Treat `question.json` as the authoritative task input. Select scientific tools from the "
            "available native tool schemas. Before using an unfamiliar tool, inspect its documentation "
            "under `resources/L1_tools/<tool-skill>/SKILL.md`. Base every conclusion on observed tool "
            "results, keep generated files in the task workspace, and write supporting methods and "
            "evidence to `run_log.md` and `result.md`. Return the requested scientific conclusion once.\n",
        )
    else:
        shutil.copytree(source_tree, resources)
        shutil.copytree(source_entry, bundle)

    for path in resources.rglob("*.md"):
        path.write_text(adapt_skill_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    for path in bundle.rglob("*.md"):
        path.write_text(adapt_skill_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    skill_path = bundle / "SKILL.md"
    if skill_visibility == "l1-only":
        runtime_contract = (
            "\n\n## DSH native tool invocation\n\n"
            "Invoke each MolClaw server tool directly through its structured schema named "
            "`mcp__molclaw-scp__<bare_tool_name>`. Never emit XML tool-call tags, call a generic "
            "wrapper, or simulate an MCP call with shell commands.\n"
            "\n## Benchmark finalization\n\n"
            "The final assistant message must obey `question.json.output_contract` and contain "
            "only the selected source SMILES.\n"
        )
    else:
        runtime_contract = (
            "\n\n## Canonical ReAct tool invocation\n\n"
            "Invoke MolClaw tools with their bare authoritative names through canonical "
            "`<tool_call>{\"tool_name\":\"...\",\"arguments\":{...}}</tool_call>` blocks. "
            "The DSH bridge translates these calls to the harness namespace. Never emit a "
            "namespaced `mcp__...` tool name, call a generic wrapper, or simulate an MCP call "
            "with shell commands.\n"
            "\n## Canonical benchmark finalization\n\n"
            "When `question.json` contains `output_contract`, the last assistant message must be the "
            "single canonical `<final_answer>` envelope specified there and nothing else. Copy the "
            "selected SMILES from the verified result. "
            "Do not replace it with a completion acknowledgement such as `Completed` or `Outputting the final answer`.\n"
        )
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8").rstrip() + runtime_contract,
        encoding="utf-8",
    )
    resource_link = bundle / "resources"
    resource_link.symlink_to(resources, target_is_directory=True)
    write_json(manifest_path, {
        "schema_version": "dsh_molclaw_skill_snapshot_v1",
        "created_at": utc_now(),
        "source": str(skill_source),
        "source_sha256": source_digest,
        "skill_visibility": skill_visibility,
        "layout": ".dsh/skills/execute-molclaw-trajectory/{SKILL.md,references,resources}",
        "adaptations": [
            ".claude/skills -> .dsh/skills/execute-molclaw-trajectory/resources",
            "Claude session -> DSH session",
            (
                "native namespaced DSH tool calls"
                if skill_visibility == "l1-only"
                else "bare MolClaw tool names preserved for canonical ReAct bridge translation"
            ),
            "single DSH catalog entry with on-demand resources",
            f"skill visibility: {skill_visibility}",
        ],
    })
    return bundle


def question_payload(sample: Sample, skill_visibility: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "dsh_molbench_question_v1",
        "task": "molbench",
        "task_id": sample.task_id,
        "suite": sample.suite,
        "source_row": sample.source_row,
        "question_text": sample.prompt,
    }
    if sample.suite == "ms1" and skill_visibility == "l1-only":
        payload["output_contract"] = (
            "Final assistant message: selected source SMILES, one per line, and nothing else."
        )
    elif sample.suite == "ms1":
        payload["output_contract"] = (
            "Final assistant message: exactly one canonical "
            "<final_answer>{\"task_type\":\"pf\",\"selected_smiles\":[...],\"evidence\":[...]}</final_answer> "
            "envelope and nothing else."
        )
    elif skill_visibility == "l1-only":
        payload["target"] = sample.target
        payload["output_contract"] = (
            "Final assistant message: exactly one of Molecule A or Molecule B SMILES, and nothing else."
        )
    else:
        payload["target"] = sample.target
        payload["output_contract"] = (
            "Final assistant message: exactly one canonical "
            "<final_answer>{\"task_type\":\"ac\",\"answer_smiles\":\"...\",\"evidence\":[...]}</final_answer> "
            "envelope and nothing else."
        )
    return payload


AGENTS_TEXT = """# DSH MolBench task workspace

This directory contains exactly one benchmark question. Treat `question.json` as the only task input.
Do not inspect parent directories, benchmark source CSV files, other task workspaces, prior predictions, or scoring artifacts.
Use the `execute-molclaw-trajectory` DSH skill and its bundled resources for scientific execution.
Write all generated artifacts inside this workspace. Never write into the skill resource snapshot.
The final assistant message must obey `question.json.output_contract`; put methods and evidence in `result.md` and `run_log.md`.
"""


def prepare_workspace(
    run_dir: Path, sample: Sample, bundle: Path, skill_visibility: str,
) -> Path:
    workdir = run_dir / "workspaces" / sample.task_id
    workdir.mkdir(parents=True, exist_ok=True)
    # DSH discovers project skills relative to the nearest .git marker.  A
    # marker-only directory is intentional: these benchmark workspaces are not
    # source repositories and must not inherit the parent repository's skills.
    (workdir / ".git").mkdir(exist_ok=True)
    skill_dir = workdir / ".dsh/skills/execute-molclaw-trajectory"
    if not skill_dir.exists():
        skill_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, skill_dir, symlinks=True)
    write_text(workdir / "AGENTS.md", AGENTS_TEXT)
    write_json(workdir / "question.json", question_payload(sample, skill_visibility))
    return workdir


class DshApi:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # The development host exports an egress proxy. DSH is loopback-only;
        # never let urllib route its local control plane through that proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rpc(self, method: str, payload: dict[str, Any], timeout: float = 60.0) -> Any:
        rpc_id = f"dsh-molbench-{uuid.uuid4()}"
        body = json.dumps({
            "type": "client-request", "rpcId": rpc_id, "method": method, "payload": payload,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/{method}", data=body, method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DSH {method} HTTP {exc.code}: {detail}") from exc
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict) or not result.get("ok"):
            error = result.get("error") if isinstance(result, dict) else data
            raise RuntimeError(f"DSH {method} failed: {error}")
        return result.get("value")

    def history(self, session_id: str) -> list[dict[str, Any]]:
        value = self.rpc("session.history", {"sessionId": session_id, "maxMessages": 500})
        return [entry["event"] for entry in value.get("events", [])]

    def approve_pending(self, session_id: str, approval_id: str) -> dict[str, Any]:
        """Answer one replayed DSH approval through the official WebSocket/respond flow."""
        ws_url = re.sub(r"^http", "ws", self.base_url) + "/api/events.mux"
        socket = websocket.create_connection(ws_url, timeout=10, http_proxy_host=None)
        try:
            while True:
                message = socket.recv()
                if not message:
                    raise RuntimeError(f"DSH event stream closed before approval {approval_id}")
                envelope = json.loads(message)
                payload = envelope.get("payload") or {}
                if (
                    payload.get("type") != "approval/requested"
                    or payload.get("sessionId") != session_id
                    or payload.get("approvalId") != approval_id
                ):
                    continue
                answer = {
                    "type": "client-response",
                    "rpcId": envelope["rpcId"],
                    "result": {
                        "ok": True,
                        "value": {
                            "sessionId": session_id,
                            "approvalId": approval_id,
                            "outcome": "allowed-once",
                        },
                    },
                }
                answer_request = urllib.request.Request(
                    f"{self.base_url}/api/respond",
                    data=json.dumps(answer).encode("utf-8"),
                    method="POST",
                    headers={"content-type": "application/json"},
                )
                with self.opener.open(answer_request, timeout=30) as answer_response:
                    receipt = json.loads(answer_response.read().decode("utf-8"))
                if not receipt.get("accepted"):
                    raise RuntimeError(f"DSH rejected approval {approval_id}: {receipt}")
                return payload
        finally:
            socket.close()


def task_prompt(sample: Sample, skill_visibility: str) -> str:
    if skill_visibility == "l1-only":
        return sample.prompt
    payload = {
        "task_id": sample.task_id,
        "task_type": "pf" if sample.suite == "ms1" else "ac",
        "question": sample.prompt,
    }
    return TASK_MARKER + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def event_text(event: dict[str, Any]) -> str:
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", "")) for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def completed_turn(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    ends = [event for event in events if event.get("type") == "turn/end"]
    return ends[-1] if ends else None


def pending_approvals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decided = {
        (event.get("data") or {}).get("id")
        for event in events
        if event.get("type") == "approval/decided"
    }
    return [
        event.get("data") or {}
        for event in events
        if event.get("type") == "approval/asked"
        and (event.get("data") or {}).get("id") not in decided
    ]


def pending_user_question(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return an unanswered ask_user_question call in an unattended run."""
    resolved_call_ids = {
        ((event.get("data") or {}).get("message") or {}).get("source", {}).get("callId")
        for event in events
        if event.get("type") == "tool/result"
        and isinstance((event.get("data") or {}).get("message"), dict)
    }
    for event in reversed(events):
        data = event.get("data") or {}
        if (
            event.get("type") == "tool/call"
            and data.get("name") == "ask_user_question"
            and data.get("callId") not in resolved_call_ids
        ):
            return data
    return None


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    end = completed_turn(events)
    turn = (end.get("data") or {}).get("turn") if end else None
    scoped = [
        event for event in events
        if turn is None or not isinstance(event.get("data"), dict) or event["data"].get("turn") in (None, turn)
    ]
    assistants = [event_text(event) for event in scoped if event.get("type") == "assistant/message"]
    calls = [event for event in scoped if event.get("type") == "tool/call"]
    names = [str((event.get("data") or {}).get("name", "")) for event in calls]
    skill_loaded = any(
        event.get("type") == "user/message"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("source"), dict)
        and event["data"]["source"].get("kind") == "skill-invocation"
        and event["data"]["source"].get("name") == "execute-molclaw-trajectory"
        for event in scoped
    )
    return {
        "final_text": assistants[-1] if assistants else "",
        "turn_reason": (end.get("data") or {}).get("reason") if end else None,
        "tool_calls": names,
        "mcp_tool_calls": [name for name in names if name.startswith("mcp__molclaw-scp__")],
        "skill_loaded": skill_loaded,
        "event_count": len(events),
    }


def diagnostic_transcript(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep assembled model/tool events without the per-token chunk flood."""
    transcript: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type in {"request/header", "assistant/message", "tool/call", "tool/result", "turn/end"}:
            entry: dict[str, Any] = {"type": event_type, "seq": event.get("seq")}
            if event_type == "assistant/message":
                message = data.get("message") if isinstance(data.get("message"), dict) else {}
                entry.update({
                    "text": event_text(event),
                    "content": message.get("content"),
                    "usage": message.get("usage"),
                    "provider": message.get("provider"),
                    "model": message.get("model"),
                })
            else:
                entry["data"] = data
            transcript.append(entry)
    return transcript


def clean_output_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("<answer>", "").replace("</answer>", "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+|Answer:\s*)", "", line, flags=re.IGNORECASE)
        lines.append(line.strip("` \t"))
    return lines


def canonical_final_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the last v7 canonical terminal envelope from final assistant text."""
    matches = re.findall(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    if not matches:
        return None, None
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        return None, f"invalid canonical final_answer JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "canonical final_answer payload must be an object"
    return payload, None


def project_prediction(sample: Sample, final_text: str) -> tuple[Any, bool, str | None]:
    payload, payload_error = canonical_final_payload(final_text)
    if payload_error is not None:
        return ([] if sample.suite == "ms1" else ""), False, payload_error
    if payload is not None:
        task_type = str(payload.get("task_type") or "").strip().lower()
        if sample.suite == "ms1":
            if task_type != "pf":
                return [], False, f"canonical final_answer task_type must be 'pf', got {task_type!r}"
            values = payload.get("selected_smiles")
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                return [], False, "canonical PF answer must contain a string selected_smiles list"
            candidates = extract_ms1_candidates(sample.prompt)
            selected = list(dict.fromkeys(value.strip() for value in values))
            valid = all(value in candidates for value in selected)
            return selected, valid, None if valid else "canonical PF answer contains a non-candidate SMILES"

        if task_type != "ac":
            return "", False, f"canonical final_answer task_type must be 'ac', got {task_type!r}"
        value = payload.get("answer_smiles")
        if not isinstance(value, str):
            return "", False, "canonical AC answer_smiles must be a string"
        value = value.strip()
        valid = value in {sample.molecule_a, sample.molecule_b}
        return value, valid, None if valid else "canonical AC answer_smiles is not one of the candidates"

    lines = clean_output_lines(final_text)
    if sample.suite == "ms1":
        candidates = extract_ms1_candidates(sample.prompt)
        selected = [line for line in lines if line in candidates]
        # Deduplicate while preserving the model's order.
        selected = list(dict.fromkeys(selected))
        valid = all(line in candidates for line in lines) and len(selected) == len(lines)
        if not lines and not final_text.strip():
            return [], False, "final assistant text is empty"
        return selected, valid, None if valid else "final text contains non-candidate lines"

    choices = [sample.molecule_a, sample.molecule_b]
    exact = [line for line in lines if line in choices]
    if len(lines) == 1 and len(exact) == 1:
        return exact[0], True, None
    mentioned = [choice for choice in choices if choice and choice in final_text]
    if len(set(mentioned)) == 1:
        return mentioned[0], False, "candidate recovered from non-conforming final text"
    return "", False, "final text does not identify exactly one candidate"


def project_artifact_prediction(sample: Sample, result_text: str) -> tuple[Any, bool]:
    """Project a benchmark answer from the skill's required result artifact.

    Fenced blocks are preferred because MolClaw's report contract places final
    selections there. This is kept distinct from final-message conformance in
    every task record and aggregate summary.
    """
    fenced = re.findall(r"```(?:text|smiles)?\s*\n(.*?)```", result_text, re.DOTALL | re.IGNORECASE)
    for block in reversed(fenced):
        prediction, valid, _ = project_prediction(sample, block.strip())
        if valid:
            return prediction, True
    exact_lines = clean_output_lines(result_text)
    if sample.suite == "ms1":
        candidates = set(extract_ms1_candidates(sample.prompt))
        selected = list(dict.fromkeys(line for line in exact_lines if line in candidates))
        return selected, bool(selected)
    selected = list(dict.fromkeys(
        line for line in exact_lines if line in {sample.molecule_a, sample.molecule_b}
    ))
    return (selected[0], True) if len(selected) == 1 else ("", False)


def run_sample(
    api: DshApi,
    run_dir: Path,
    sample: Sample,
    workdir: Path,
    timeout_sec: int,
    model_provider: str,
    model_id: str,
    agent_preset: str,
    skill_visibility: str,
) -> dict[str, Any]:
    started = time.time()
    session_id = f"session-dsh-molbench-{uuid.uuid4()}"
    record: dict[str, Any] = {
        "schema_version": "dsh_molbench_task_result_v1",
        "task_id": sample.task_id,
        "suite": sample.suite,
        "source_row": sample.source_row,
        "session_id": session_id,
        "workdir": str(workdir),
        "started_at": utc_now(),
        "status": "running",
    }
    write_json(workdir / "record.json", record)
    events: list[dict[str, Any]] = []
    try:
        created = api.rpc("session.create", {
            "cwd": str(workdir), "sessionId": session_id, "agentPreset": agent_preset,
        }, timeout=180)
        session_id = created["sessionId"]
        record["session_id"] = session_id
        selected = api.rpc("session.selectModel", {
            "sessionId": session_id,
            "provider": model_provider,
            "model": model_id,
        }, timeout=60)
        record["model_selection"] = selected["selected"]
        skill_catalog = api.rpc("skill.list", {"sessionId": session_id}, timeout=60).get("skills", [])
        discovered = next(
            (skill for skill in skill_catalog if skill.get("name") == "execute-molclaw-trajectory"),
            None,
        )
        if discovered is None:
            raise RuntimeError("DSH did not discover execute-molclaw-trajectory in the task workspace")
        record["skill_discovered"] = True
        record["skill_catalog_count"] = len(skill_catalog)
        api.rpc("session.prompt", {
            "sessionId": session_id,
            "mode": "queue",
            "content": [{"type": "text", "text": task_prompt(sample, skill_visibility)}],
        }, timeout=60)

        deadline = time.monotonic() + timeout_sec
        delay = 3.0
        approved: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            events = api.history(session_id)
            question = pending_user_question(events)
            if question is not None:
                try:
                    api.rpc("session.cancel", {"sessionId": session_id}, timeout=30)
                except Exception:
                    pass
                raise RuntimeError(
                    "unattended task called ask_user_question "
                    f"(callId={question.get('callId')})"
                )
            for pending in pending_approvals(events):
                approval_id = pending.get("id")
                if not isinstance(approval_id, str) or not approval_id:
                    raise RuntimeError(f"malformed DSH approval event: {pending}")
                requested = api.approve_pending(session_id, approval_id)
                approved.append({
                    "approval_id": approval_id,
                    "tool_name": pending.get("toolName"),
                    "reason": pending.get("reason"),
                    "outcome": "allowed-once",
                    "requested_tool_name": requested.get("toolName"),
                })
                record["approvals"] = approved
                write_json(workdir / "record.json", record)
                delay = 3.0
            if completed_turn(events) is not None:
                break
            time.sleep(delay)
            delay = min(delay * 1.35, 15.0)
        else:
            try:
                api.rpc("session.cancel", {"sessionId": session_id}, timeout=30)
            except Exception:
                pass
            raise TimeoutError(f"task exceeded {timeout_sec} seconds")

        summary = summarize_events(events)
        prediction, valid, projection_error = project_prediction(sample, summary["final_text"])
        reason = summary.get("turn_reason")
        completed = isinstance(reason, dict) and reason.get("kind") == "completed"
        final_output_conforming = valid and completed
        answer_source = "final_assistant_message" if final_output_conforming else None
        record.update(summary)
        record.update({
            "prediction": prediction,
            "valid_output": valid and completed,
            "final_output_conforming": final_output_conforming,
            "answer_source": answer_source,
            "projection_error": projection_error if completed else "DSH turn did not complete",
            "status": "completed" if completed else "failed",
            "approvals": approved,
        })
        write_text(workdir / "final_answer.txt", summary["final_text"] + "\n")
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    if events:
        write_json(workdir / "diagnostic_transcript.json", diagnostic_transcript(events))
    record.update({"finished_at": utc_now(), "elapsed_seconds": round(time.time() - started, 3)})
    write_json(workdir / "record.json", record)
    return record


def load_record(run_dir: Path, sample: Sample) -> dict[str, Any] | None:
    path = run_dir / "workspaces" / sample.task_id / "record.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_scores(run_dir: Path, molbench_root: Path, samples: list[Sample]) -> dict[str, Any]:
    ms1_rows: list[dict[str, Any]] = []
    ms2_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for sample in samples:
        record = load_record(run_dir, sample) or {"status": "missing", "prediction": [] if sample.suite == "ms1" else ""}
        records.append(record)
        prediction = record.get("prediction")
        if sample.suite == "ms1":
            values = prediction if isinstance(prediction, list) else []
            ms1_rows.append({
                "id": sample.task_id,
                "gt": sample.answer,
                "json_results": {"output": "\n".join(str(value) for value in values)},
            })
        else:
            value = prediction if isinstance(prediction, str) else ""
            ms2_rows.append({
                "id": sample.task_id,
                "gt": sample.answer,
                "s1": sample.molecule_a,
                "s2": sample.molecule_b,
                "json_results": {"output": value},
            })

    metrics: dict[str, Any] = {}
    evaluator_path = molbench_root / "eval/eval_runner.py"
    spec = importlib.util.spec_from_file_location("dsh_molbench_official_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MolBench evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if ms1_rows:
        pred_dir = run_dir / "preds/rdkit_bench"
        write_json(pred_dir / "all.json", ms1_rows)
        metrics.update(module.RdkitBenchEval().run(str(pred_dir), str(run_dir), ""))
    if ms2_rows:
        pred_dir = run_dir / "preds/acnet_curated"
        write_json(pred_dir / "all.json", ms2_rows)
        metrics.update(module.ACNetCuratedEval().run(str(pred_dir), str(run_dir), ""))
    write_json(run_dir / "metrics.json", metrics)

    completed = [record for record in records if record.get("status") == "completed"]
    valid = [record for record in completed if record.get("valid_output") is True]
    non_empty = [
        record for record in completed
        if (
            isinstance(record.get("prediction"), list) and len(record["prediction"]) > 0
            or isinstance(record.get("prediction"), str) and bool(record["prediction"].strip())
        )
    ]
    mcp_calls = sum(len(record.get("mcp_tool_calls") or []) for record in records)
    summary = {
        "schema_version": "dsh_molbench_evaluation_summary_v1",
        "updated_at": utc_now(),
        "sample_count": len(samples),
        "completed_count": len(completed),
        "failed_or_missing_count": len(samples) - len(completed),
        "valid_output_count": len(valid),
        "non_empty_prediction_count": len(non_empty),
        "final_output_conforming_count": sum(
            record.get("final_output_conforming") is True for record in records
        ),
        "artifact_answer_count": 0,
        "skill_loaded_count": sum(record.get("skill_loaded") is True for record in records),
        "mcp_tool_call_count": mcp_calls,
        "metrics": metrics,
        "denominator_policy": "all selected samples; failed or missing tasks receive an empty prediction",
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    return summary


def materialize_rollout_summary(run_dir: Path, samples: list[Sample]) -> dict[str, Any]:
    records = [
        load_record(run_dir, sample)
        or {"task_id": sample.task_id, "suite": sample.suite, "status": "missing"}
        for sample in samples
    ]
    completed = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    running = [record for record in records if record.get("status") == "running"]
    non_empty = [
        record for record in completed
        if (
            isinstance(record.get("prediction"), list) and len(record["prediction"]) > 0
            or isinstance(record.get("prediction"), str) and bool(record["prediction"].strip())
        )
    ]
    summary = {
        "schema_version": "dsh_molbench_rollout_summary_v1",
        "updated_at": utc_now(),
        "sample_count": len(samples),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "running_count": len(running),
        "missing_count": len(samples) - len(completed) - len(failed) - len(running),
        "valid_output_count": sum(record.get("valid_output") is True for record in completed),
        "non_empty_prediction_count": len(non_empty),
        "final_output_conforming_count": sum(
            record.get("final_output_conforming") is True for record in completed
        ),
        "artifact_answer_count": 0,
        "skill_loaded_count": sum(record.get("skill_loaded") is True for record in completed),
        "mcp_tool_call_count": sum(len(record.get("mcp_tool_calls") or []) for record in completed),
        "scored": False,
        "scoring_instruction": "Run this runner with --score-only on the login host after rollout completion.",
    }
    write_json(run_dir / "rollout_summary.json", summary)
    return summary


def git_revision(repo: Path) -> str | None:
    head = repo / ".git/HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = repo / ".git" / value.removeprefix("ref: ")
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else value
    return value


def build_manifest(
    run_dir: Path,
    molbench_root: Path,
    dsh_repo: Path,
    skill_source: Path,
    suites: set[str],
    limit_per_suite: int,
    samples: list[Sample],
    dsh_url: str,
    model_provider: str,
    model_id: str,
    agent_preset: str,
    skill_visibility: str,
) -> dict[str, Any]:
    return {
        "schema_version": "dsh_molbench_run_manifest_v1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "dsh_url": dsh_url,
        "dsh_source_repo": str(dsh_repo),
        "dsh_revision": git_revision(dsh_repo),
        "agent_preset": agent_preset,
        "model_provider": model_provider,
        "model": model_id,
        "molbench_root": str(molbench_root),
        "skill_source": str(skill_source),
        "skill_visibility": skill_visibility,
        "suites": sorted(suites),
        "limit_per_suite": limit_per_suite,
        "sample_count": len(samples),
        "sample_ids": [sample.task_id for sample in samples],
        "benchmark_integrity": {
            "ms2_training_overlap_excluded": 4 if "ms2" in suites else 0,
            "answers_written_to_workspaces": False,
            "cross_workspace_reads_forbidden_by_AGENTS_md": True,
        },
        "evaluation_contract": {
            "native_thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 16384,
            "max_prompt_tokens": 49152,
            "context_window": 65536,
            "max_steps": 128,
            "tool_protocol": (
                "qwen_native_structured_calls"
                if skill_visibility == "l1-only"
                else "canonical_react_bridge_to_dsh_structured_calls"
            ),
            "terminal_protocol": (
                "plain_benchmark_answer"
                if skill_visibility == "l1-only"
                else "canonical_final_answer"
            ),
            "empty_final_is_valid": False,
            "artifact_answer_fallback": False,
        },
        "source_files": [
            {"path": str(molbench_root / MS1_CSV), "sha256": sha256_file(molbench_root / MS1_CSV)}
            if "ms1" in suites else None,
            {"path": str(molbench_root / MS2_CSV), "sha256": sha256_file(molbench_root / MS2_CSV)}
            if "ms2" in suites else None,
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSH + local Qwen3.5 on MolBench MS-1/MS-2")
    parser.add_argument("--suite", action="append", choices=("ms1", "ms2"), default=[])
    parser.add_argument("--limit-per-suite", type=int, default=0, help="0 means all selected samples")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--rollout-only", action="store_true",
        help="write trajectories and predictions without importing or running the evaluator",
    )
    parser.add_argument("--task-timeout-sec", type=int, default=3600)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dsh-url", default=DEFAULT_DSH_URL)
    parser.add_argument("--model-provider", default="slime-local")
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument("--agent-preset", default=DEFAULT_AGENT_PRESET)
    parser.add_argument("--molbench-root", type=Path, default=DEFAULT_MOLBENCH_ROOT)
    parser.add_argument("--dsh-repo", type=Path, default=DEFAULT_DSH_REPO)
    parser.add_argument("--skill-source", type=Path, default=DEFAULT_SKILL_SOURCE)
    parser.add_argument(
        "--skill-visibility", choices=("hierarchy", "l1-only"), default="hierarchy",
        help="expose the complete skill hierarchy or only on-demand L1 tool documentation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollout_only and args.score_only:
        raise ValueError("--rollout-only and --score-only are mutually exclusive")
    suites = set(args.suite or ["ms1", "ms2"])
    molbench_root = args.molbench_root.expanduser().resolve()
    dsh_repo = args.dsh_repo.expanduser().resolve()
    skill_source = args.skill_source.expanduser().resolve()
    if args.task_timeout_sec <= 0:
        raise ValueError("--task-timeout-sec must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    samples = load_samples(molbench_root, suites, args.limit_per_suite)

    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = DEFAULT_RUNS_ROOT / f"dsh_qwen35_original_ms1_ms2_{stamp}"
    else:
        run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not (args.resume or args.score_only):
        raise RuntimeError(f"run directory already exists; pass --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sample_ids") != [sample.task_id for sample in samples]:
            raise RuntimeError("resume selection does not match the existing run manifest")
        if manifest.get("skill_visibility", "hierarchy") != args.skill_visibility:
            raise RuntimeError("resume skill visibility does not match the existing run manifest")
    else:
        manifest = build_manifest(
            run_dir, molbench_root, dsh_repo, skill_source, suites,
            args.limit_per_suite, samples, args.dsh_url,
            args.model_provider, args.model_id, args.agent_preset,
            args.skill_visibility,
        )
        manifest["execution_mode"] = "rollout-only" if args.rollout_only else "rollout-and-score"
        manifest["source_files"] = [item for item in manifest["source_files"] if item is not None]
        write_json(manifest_path, manifest)

    bundle = create_skill_snapshot(run_dir, skill_source, args.skill_visibility)
    workdirs = {
        sample.task_id: prepare_workspace(run_dir, sample, bundle, args.skill_visibility)
        for sample in samples
    }
    if args.prepare_only:
        print(json.dumps({"run_dir": str(run_dir), "prepared": len(samples)}, ensure_ascii=False))
        return 0

    if not args.score_only:
        api = DshApi(args.dsh_url)
        # Read-only API check before creating any benchmark session.
        api.rpc("session.list", {}, timeout=30)
        pending: list[tuple[int, Sample]] = []
        for index, sample in enumerate(samples, 1):
            existing = load_record(run_dir, sample)
            if existing and existing.get("status") == "completed":
                print(f"[{index}/{len(samples)}] {sample.task_id}: resume skip completed", flush=True)
                continue
            if existing and not args.retry_failed:
                print(f"[{index}/{len(samples)}] {sample.task_id}: resume skip failed (use --retry-failed)", flush=True)
                continue
            pending.append((index, sample))

        def execute(item: tuple[int, Sample]) -> tuple[int, Sample, dict[str, Any]]:
            index, sample = item
            print(f"[{index}/{len(samples)}] {sample.task_id}: running", flush=True)
            record = run_sample(
                DshApi(args.dsh_url), run_dir, sample, workdirs[sample.task_id], args.task_timeout_sec,
                args.model_provider, args.model_id, args.agent_preset, args.skill_visibility,
            )
            return index, sample, record

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(execute, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                index, sample, record = future.result()
                print(
                    f"[{index}/{len(samples)}] {sample.task_id}: {record.get('status')} "
                    f"valid={record.get('valid_output')} mcp_calls={len(record.get('mcp_tool_calls') or [])} "
                    f"elapsed={record.get('elapsed_seconds')}s",
                    flush=True,
                )
                if args.rollout_only:
                    materialize_rollout_summary(run_dir, samples)
                else:
                    materialize_scores(run_dir, molbench_root, samples)

    if args.rollout_only:
        summary = materialize_rollout_summary(run_dir, samples)
    else:
        summary = materialize_scores(run_dir, molbench_root, samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
