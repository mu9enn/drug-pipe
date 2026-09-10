#!/usr/bin/env python3
"""Run MolBench MS-1/MS-2 through a persistent DSH Web host.

Each benchmark item gets an independent DSH project/workspace. The runner
copies the portable MolClaw L1 workspace template verbatim and concatenates its
prompt prefix with the benchmark question. Ground-truth answers and runner
records are never written into task workspaces.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websocket

from install_eval_preset import SKILL_PRESET_ID, install as install_eval_preset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DRUG_PIPE_ROOT = WORKSPACE_ROOT.parent
DATA_PIPE_ROOT = DRUG_PIPE_ROOT / "data-pipe"
if str(DATA_PIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_PIPE_ROOT))

from pipeline.output_contracts import (  # noqa: E402
    CONTRACT_VERSION,
    normalize_final_answer,
    normalize_task_prompt,
    task_constraints,
)

from pipeline.benchmark_release import UPSTREAM_COMMIT

PINNED_SCORER = Path("upstream") / UPSTREAM_COMMIT / "eval/eval_runner.py"
PINNED_SCORER_SHA256 = "ef9a31c03a89629a0d5c87d906466e32f3832bd2e8f7b697cb6290a43909c47c"

DEFAULT_MOLBENCH_ROOT = WORKSPACE_ROOT / "molbench"
DEFAULT_DSH_REPO = WORKSPACE_ROOT / "deepseek-harness"
DEFAULT_SKILL_SOURCE = DRUG_PIPE_ROOT / "workdir-skills/molclaw-l1-workspace"
DEFAULT_SYSTEM_PROMPT = DATA_PIPE_ROOT / "pipeline/cleaning/prompts/qwen35_system.md"
DEFAULT_RUNS_ROOT = WORKSPACE_ROOT / "outputs/dsh_molbench_evals"
DEFAULT_DSH_URL = "http://127.0.0.1:3080"
DEFAULT_AGENT_PRESET = SKILL_PRESET_ID
LOCAL_EVAL_TOOLS = frozenset({"bash", "read", "write", "edit", "grep", "glob", "skill"})
MOLCLAW_TOOL_PREFIX = "mcp__molclaw-scp__"

MS1_CSV = Path("data/molbench-ms-1/molbench-ms-1.csv")
MS2_CSV = Path("data/molbench-ms-2/molbench-ms-2.csv")
MS3_CSV = Path("data/molbench-ms-3/molbench-ms-3.csv")
SUITE_TASKS = {"ms1": "pf", "ms2": "ac", "ms3": "vs", "mo-opt": "mo-opt", "mo-edit": "mo-edit"}


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
    source_id: str = ""
    subtask: str = ""
    source_molecule: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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
    from pipeline.benchmark_release import SPECS
    if limit_per_suite < 0 or not suites <= set(SUITE_TASKS):
        raise ValueError("invalid suite selection or limit")
    release = molbench_root / "aligned"
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError("benchmark contract changed; rebuild the aligned release")
    samples = []
    for suite, (folder, column, task, count) in SPECS.items():
        if suite not in suites:
            continue
        path = release / "data" / folder / f"{folder}.csv"
        if sha256_file(path) != manifest["suites"][suite]["output_sha256"]:
            raise ValueError(f"{suite}: aligned question hash mismatch")
        rows = csv_rows(path)
        if len(rows) != count:
            raise ValueError(f"{suite}: expected {count} rows, found {len(rows)}")
        for index, row in enumerate(rows, 1):
            prompt = row[column]
            c = task_constraints(prompt, task)
            frozen = manifest['suites'][suite]['constraints'][index - 1]
            if (frozen['id'] != f'molbench_{suite}_{index:03d}' or
                    frozen['exact_count'] != c.exact_count or frozen['complete_ranking'] != c.complete_ranking):
                raise ValueError(f'{suite}/{index}: frozen task constraints differ')
            if normalize_task_prompt(prompt, task) != prompt:
                raise ValueError("benchmark prompt is not canonical")
            if limit_per_suite and index > limit_per_suite:
                continue
            samples.append(Sample(
                task_id=f"molbench_{suite}_{index:03d}", suite=suite, source_row=index,
                prompt=prompt, answer=row.get("answer", ""), target=row.get("target", ""),
                molecule_a=c.candidates[0] if suite == "ms2" else "",
                molecule_b=c.candidates[1] if suite == "ms2" else "",
            ))
    if suites & {"mo-opt", "mo-edit"}:
        from pipeline.mo_benchmark_release import load_rows
        for suite, index, row, molecule in load_rows(molbench_root, suites, limit_per_suite):
            samples.append(Sample(
                task_id=f"molbench_{suite.replace('-', '_')}_{row['subtask']}_{row['id']}",
                suite=suite, source_row=index, prompt=row['query'], answer=row.get('gt', ''),
                source_id=row['id'], subtask=row['subtask'], source_molecule=molecule,
                metadata=json.loads(row['meta']),
            ))
    return samples


def create_workspace_snapshot(run_dir: Path, skill_source: Path) -> Path:
    snapshot = run_dir / "workspace_snapshot"
    payload = snapshot / "payload"
    manifest_path = snapshot / "manifest.json"
    source_digest = tree_digest(skill_source)

    if manifest_path.is_file() and payload.is_dir():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != source_digest:
            raise RuntimeError("skill source changed after this run was prepared; use a new run directory")
        if manifest.get("schema_version") != "dsh_molclaw_workspace_snapshot_v1":
            raise RuntimeError("snapshot layout changed; use a new run directory")
        return payload

    if snapshot.exists():
        raise RuntimeError(f"incomplete skill snapshot already exists: {snapshot}")
    if not (skill_source / "prompt_prefix.md").is_file():
        raise FileNotFoundError(skill_source / "prompt_prefix.md")
    if not (skill_source / ".agents/skills").is_dir():
        raise FileNotFoundError(skill_source / ".agents/skills")
    snapshot.mkdir(parents=True)
    shutil.copytree(skill_source, payload)
    write_json(manifest_path, {
        "schema_version": "dsh_molclaw_workspace_snapshot_v1",
        "created_at": utc_now(),
        "source": str(skill_source),
        "source_sha256": source_digest,
        "layout": ".agents/skills/<skill-name>",
        "adaptations": [],
    })
    return payload


def prepare_workspace(
    run_dir: Path, sample: Sample, workspace_payload: Path,
) -> Path:
    workdir = run_dir / "workspaces" / sample.task_id
    if not workdir.exists():
        workdir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(workspace_payload, workdir)
    # DSH resolves project-local instructions relative to the nearest .git
    # marker. These benchmark workspaces must not inherit parent-repository
    # configuration or skills.
    (workdir / ".git").mkdir(exist_ok=True)
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


def protocol_snapshot(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, event in enumerate(events):
        if event.get("type") != "request/header":
            continue
        header = event.get("data", {}).get("header")
        if not isinstance(header, dict) or not isinstance(header.get("tools"), list):
            return None
        names = [
            tool.get("name") for tool in header["tools"]
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        ]
        catalogs = [
            prior for prior in events[:index]
            if prior.get("type") == "user/message"
            and prior.get("data", {}).get("source", {}).get("kind") == "skill-catalog"
        ]
        entries = (
            catalogs[-1].get("data", {}).get("source", {}).get("entries", [])
            if catalogs else []
        )
        return {
            "system": header.get("system"),
            "config": header.get("config"),
            "tool_names": names,
            "tools": header["tools"],
            "molclaw_tool_count": sum(name.startswith(MOLCLAW_TOOL_PREFIX) for name in names),
            "local_tools": sorted(name for name in names if not name.startswith(MOLCLAW_TOOL_PREFIX)),
            "skill_catalog_count": len(entries) if isinstance(entries, list) else 0,
            "skill_entries": entries,
            "skill_names": [
                entry.get("name") for entry in entries
                if isinstance(entry, dict) and isinstance(entry.get("name"), str)
            ] if isinstance(entries, list) else [],
        }
    return None


def workspace_catalog(workdir: Path) -> list[dict[str, str]]:
    import yaml
    entries = []
    for path in sorted((workdir / ".agents/skills").iterdir()):
        skill = path / "SKILL.md" if path.is_dir() else path
        if not skill.is_file() or skill.suffix != ".md":
            continue
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", skill.read_text(), re.S)
        if not match:
            continue
        header = yaml.safe_load(match[1])
        description = re.sub(r"\s+", " ", str(header['description'])).strip()
        if len(description) > 500:
            description = description[:497] + "..."
        entries.append({"name": header['name'], "description": description})
    return sorted(entries, key=lambda entry: entry['name'])


def protocol_snapshot_matches(
    snapshot: dict[str, Any],
    expected_system: str,
    expected_mcp_tools: int,
    expected_skill_count: int,
    expected_catalog: list[dict[str, str]] | None = None,
) -> bool:
    expected = json.loads((DATA_PIPE_ROOT / "configs/dsh_molclaw_tool_set.json").read_text())["tools"]
    expected_tools = [{"name": t["name"], "description": t["description"], "parameters": t["input_schema"]} for t in expected]
    return (
        snapshot.get("tools") == expected_tools
        and expected_catalog is not None
        and sorted(snapshot.get("skill_entries", []), key=lambda entry: entry["name"]) == expected_catalog
        and snapshot["system"] == expected_system
        and snapshot["molclaw_tool_count"] == expected_mcp_tools
        and len(set(snapshot["tool_names"])) == expected_mcp_tools + len(LOCAL_EVAL_TOOLS)
        and set(snapshot["local_tools"]) == LOCAL_EVAL_TOOLS
        and len(snapshot["tool_names"]) == expected_mcp_tools + len(LOCAL_EVAL_TOOLS)
        and snapshot["skill_catalog_count"] == expected_skill_count
        and len(set(snapshot["skill_names"])) == expected_skill_count
        and isinstance(snapshot["config"], dict)
        and snapshot["config"].get("maxTokens") == 16384
    )


def wait_for_protocol_ready(
    api: DshApi,
    workdir: Path,
    model_provider: str,
    model_id: str,
    agent_preset: str,
    expected_system: str,
    expected_mcp_tools: int,
    expected_skill_count: int,
    timeout_sec: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    attempts: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        session_id = f"session-dsh-molbench-protocol-probe-{uuid.uuid4()}"
        snapshot: dict[str, Any] | None = None
        try:
            created = api.rpc("session.create", {
                "cwd": str(workdir), "sessionId": session_id, "agentPreset": agent_preset,
            }, timeout=180)
            session_id = created["sessionId"]
            api.rpc("session.selectModel", {
                "sessionId": session_id, "provider": model_provider, "model": model_id,
            }, timeout=60)
            api.rpc("session.prompt", {
                "sessionId": session_id,
                "mode": "queue",
                "content": [{"type": "text", "text": "Protocol readiness probe. Reply OK."}],
            }, timeout=60)
            header_deadline = min(deadline, time.monotonic() + 30)
            while time.monotonic() < header_deadline:
                snapshot = protocol_snapshot(api.history(session_id))
                if snapshot is not None:
                    break
                time.sleep(0.5)
        except Exception as exc:
            attempts.append({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            try:
                api.rpc("session.cancel", {"sessionId": session_id}, timeout=30)
            except Exception:
                pass

        if snapshot is not None:
            passed = protocol_snapshot_matches(
                snapshot, expected_system, expected_mcp_tools, expected_skill_count, workspace_catalog(workdir),
            )
            attempts.append({
                "molclaw_tool_count": snapshot["molclaw_tool_count"],
                "local_tools": snapshot["local_tools"],
                "skill_catalog_count": snapshot["skill_catalog_count"],
                "passed": passed,
            })
            if passed:
                return {
                    "schema_version": "dsh_molbench_startup_protocol_probe_v1",
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "passed": True,
                }
        if time.monotonic() < deadline:
            time.sleep(5)
    raise RuntimeError(f"DSH protocol did not become ready within {timeout_sec}s: {attempts}")


def aligned_task_text(sample: Sample) -> str:
    return sample.prompt


def task_prompt(sample: Sample, prompt_prefix: str) -> str:
    return f"{prompt_prefix.strip()}\n\n# Task\n\n{aligned_task_text(sample)}"


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
    assistants = [event for event in scoped if event.get("type") == "assistant/message"]
    final_text = event_text(assistants[-1]) if assistants else ''
    if assistants and any(block.get('type') == 'tool-call' for block in
                          assistants[-1].get('data', {}).get('message', {}).get('content', [])):
        final_text = ''
    calls = [event for event in scoped if event.get("type") == "tool/call"]
    names = [str((event.get("data") or {}).get("name", "")) for event in calls]
    return {
        "final_text": final_text,
        "turn_reason": (end.get("data") or {}).get("reason") if end else None,
        "tool_calls": names,
        "mcp_tool_calls": [name for name in names if name.startswith("mcp__molclaw-scp__")],
        "event_count": len(events),
    }


def diagnostic_transcript(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep assembled model/tool events without the per-token chunk flood."""
    transcript: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type in {"user/message", "request/header", "assistant/message", "tool/call", "tool/result", "turn/end"}:
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


def project_prediction(sample: Sample, final_text: str) -> tuple[Any, bool, str | None]:
    from pipeline.output_contracts import ANSWER_KEYS, strict_json_loads
    if sample.suite == "ms3":
        try:
            payload = strict_json_loads(final_text)
            ranking = payload.get("ranked_smiles") if isinstance(payload, dict) else None
            if not isinstance(ranking, list) or not all(isinstance(value, str) for value in ranking):
                raise ValueError("ranked_smiles must be a list of strings")
            return ranking, True, None
        except ValueError as exc:
            return [], False, str(exc)
    task = SUITE_TASKS[sample.suite]
    empty = "" if task in {"ac", "mo-opt", "mo-edit"} else []
    try:
        payload = json.loads(normalize_final_answer(
            final_text, task, constraints=task_constraints(sample.prompt, task),
        ))
        return payload[ANSWER_KEYS[task]], True, None
    except ValueError as exc:
        return empty, False, str(exc)


def json_format_valid(sample: Sample, text: str) -> bool:
    try:
        normalize_final_answer(text, SUITE_TASKS[sample.suite])
        return True
    except ValueError:
        return False


def publishable_records(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(
        record.get("status") == "completed"
        or (record.get("status") == "failed" and record.get("failure_class") in
            {"model_max_tokens", "model_or_protocol_failure"})
        for record in records
    ) and all(record.get("protocol_verified") is True for record in records)


def run_sample(
    api: DshApi,
    run_dir: Path,
    sample: Sample,
    workdir: Path,
    timeout_sec: int,
    model_provider: str,
    model_id: str,
    agent_preset: str,
    prompt_prefix: str,
    attempt: int = 1,
    expected_system: str | None = None,
    expected_mcp_tools: int = 0,
    expected_skill_count: int = 0,
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
        "attempt": attempt,
        "status": "running",
    }
    result_dir = run_dir / "results" / sample.task_id
    write_json(result_dir / "record.json", record)
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
        api.rpc("session.prompt", {
            "sessionId": session_id,
            "mode": "queue",
            "content": [{"type": "text", "text": task_prompt(sample, prompt_prefix)}],
        }, timeout=60)

        deadline = time.monotonic() + timeout_sec
        if expected_system is not None:
            header_deadline = min(deadline, time.monotonic() + 30)
            snapshot = None
            while time.monotonic() < header_deadline:
                events = api.history(session_id)
                snapshot = protocol_snapshot(events)
                if snapshot is not None:
                    break
                time.sleep(0.5)
            if snapshot is None or not protocol_snapshot_matches(
                snapshot, expected_system, expected_mcp_tools, expected_skill_count, workspace_catalog(workdir),
            ):
                try:
                    api.rpc("session.cancel", {"sessionId": session_id}, timeout=30)
                except Exception:
                    pass
                raise RuntimeError(f"protocol mismatch for task request: {snapshot}")
            record["protocol_verified"] = True
            record["protocol_snapshot"] = snapshot
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
                write_json(result_dir / "record.json", record)
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
        record["json_format_valid"] = json_format_valid(sample, summary["final_text"])
        record.update({
            "prediction": prediction,
            "valid_output": valid and completed,
            "final_output_conforming": final_output_conforming,
            "answer_source": answer_source,
            "projection_error": projection_error if completed else "DSH turn did not complete",
            "status": "completed" if completed else "failed",
            "approvals": approved,
        })
        write_text(result_dir / "final_answer.txt", summary["final_text"] + "\n")
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    if events:
        write_json(result_dir / "diagnostic_transcript.json", diagnostic_transcript(events))
    record.update({"finished_at": utc_now(), "elapsed_seconds": round(time.time() - started, 3)})
    record["failure_class"] = failure_class(record)
    write_json(result_dir / "record.json", record)
    return record


RETRYABLE_INFRA_CODES = frozenset({"TIMEOUT", "TRANSPORT", "SERVER", "MCP_CONNECTION"})
RETRYABLE_INFRA_TEXT = re.compile(
    r"(?:stream idle timeout|connection (?:reset|refused|closed)|temporarily unavailable|"
    r"timed out|mcp.*(?:connect|transport)|websocket.*closed)",
    re.IGNORECASE,
)


def failure_class(record: dict[str, Any]) -> str | None:
    if record.get("status") != "failed":
        return None
    reason = record.get("turn_reason")
    if isinstance(reason, dict) and reason.get("kind") == "max-tokens":
        return "model_max_tokens"
    if isinstance(reason, dict) and reason.get("kind") == "error":
        error = reason.get("error")
        if isinstance(error, dict) and str(error.get("code") or "") in RETRYABLE_INFRA_CODES:
            return "retryable_infra"
        if RETRYABLE_INFRA_TEXT.search(json.dumps(error, ensure_ascii=False)):
            return "retryable_infra"
        return "unclassified_failure"
    if RETRYABLE_INFRA_TEXT.search(str(record.get("error") or "")):
        return "retryable_infra"
    if re.search(r'unattended task called ask_user_question|task exceeded \d+ seconds', str(record.get('error') or '')):
        return 'model_or_protocol_failure'
    return "unclassified_failure"


def archive_failed_attempt(run_dir: Path, sample: Sample, record: dict[str, Any]) -> None:
    result_dir = run_dir / "results" / sample.task_id
    attempt = int(record.get("attempt") or 1)
    archive = result_dir / "attempts" / f"attempt-{attempt}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("record.json", "diagnostic_transcript.json", "final_answer.txt"):
        source = result_dir / name
        if source.is_file():
            shutil.copy2(source, archive / name)


def load_record(run_dir: Path, sample: Sample) -> dict[str, Any] | None:
    path = run_dir / "results" / sample.task_id / "record.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_scores(run_dir: Path, molbench_root: Path, samples: list[Sample]) -> dict[str, Any]:
    ms1_rows: list[dict[str, Any]] = []
    ms2_rows: list[dict[str, Any]] = []
    ms3_rows: list[dict[str, Any]] = []
    mo_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for sample in samples:
        record = load_record(run_dir, sample) or {"status": "missing", "prediction": "" if sample.suite == "ms2" else []}
        if record.get("status") == "completed":
            prediction, valid, projection_error = project_prediction(sample, str(record.get("final_text") or ""))
            record.update({
                "prediction": prediction,
                "valid_output": valid,
                "final_output_conforming": valid,
                "answer_source": "final_assistant_message" if valid else None,
                "projection_error": projection_error,
            })
            write_json(run_dir / "results" / sample.task_id / "record.json", record)
        records.append(record)
        record["json_format_valid"] = json_format_valid(sample, str(record.get("final_text") or ""))
        prediction = record.get("prediction") if record.get("status") == "completed" and record.get("valid_output") else None
        if sample.suite == "ms1":
            values = prediction if isinstance(prediction, list) else []
            ms1_rows.append({
                "id": sample.task_id,
                "gt": sample.answer,
                "json_results": {"output": "\n".join(str(value) for value in values)},
            })
        elif sample.suite == "ms3":
            ms3_rows.append({"index": sample.source_row, "answer": json.loads(sample.answer),
                             "candidates": list(task_constraints(sample.prompt, "vs").candidates),
                             "json_results": {"ranking": prediction if isinstance(prediction, list) else []}})
        elif sample.suite in {"mo-opt", "mo-edit"}:
            from pipeline.output_contracts import ANSWER_KEYS
            row = {"id": sample.source_id, "src_smiles": sample.source_molecule,
                   "molecule": sample.source_molecule, **sample.metadata,
                   "json_results": {ANSWER_KEYS[sample.suite]: prediction} if prediction else {}}
            mo_rows.setdefault((sample.suite, sample.subtask), []).append(row)
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
    evaluator_path = molbench_root / PINNED_SCORER
    if sha256_file(evaluator_path) != PINNED_SCORER_SHA256:
        raise ValueError("pinned upstream scorer changed")
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
    if ms3_rows:
        pred_dir = run_dir / "preds/molbench_vs"
        write_json(pred_dir / "all.json", ms3_rows)
        metrics.update(module.MolbenchVsEval().run(str(pred_dir), str(run_dir), ""))
    if mo_rows:
        manifest_path = run_dir / 'run_manifest.json'
        if manifest_path.exists():
            for item in json.loads(manifest_path.read_text()).get('mo_scoring_source_files', []):
                if sha256_file(molbench_root / item['path']) != item['sha256']:
                    raise ValueError('MO scorer source changed after rollout')
        eval_root = molbench_root / "ChemCoTBench/baseline_and_eval"
        for (suite, subtask), rows in mo_rows.items():
            write_json(run_dir / "preds" / suite / f"{subtask}.json", rows)
        for suite in sorted({key[0] for key in mo_rows}):
            evaluator = (module.ChemCoTBenchMolEditEval() if suite == "mo-edit"
                         else module.ChemCoTBenchMolOptPhyschemEval())
            metrics.update(evaluator.run(str(run_dir / "preds" / suite), str(run_dir), str(eval_root)))
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
        "strict_format_rate": len(valid) / len(samples) if samples else 0.0,
        "json_format_rate": sum(r.get("json_format_valid") is True for r in records) / len(samples) if samples else 0.0,
        "task_constraint_rate": len(valid) / len(samples) if samples else 0.0,
        "non_empty_prediction_count": len(non_empty),
        "final_output_conforming_count": sum(
            record.get("final_output_conforming") is True for record in records
        ),
        "artifact_answer_count": 0,
        "mcp_tool_call_count": mcp_calls,
        "model_failure_count": sum(
            record.get("status") == "completed" and record.get("valid_output") is not True
            or record.get("failure_class") in {"model_max_tokens", "model_or_protocol_failure"}
            for record in records
        ),
        "infra_retry_count": sum(max(int(record.get("attempt") or 1) - 1, 0) for record in records),
        "infra_failure_count": sum(record.get("failure_class") == "retryable_infra" for record in records),
        "publishable": publishable_records(records),
        "metrics": metrics,
        "projection_protocol": "structured_v8_strict_json",
        "ms3_scoring_policy": "top3_list_v1: accept string lists without candidate/count/uniqueness gates; score original first three positions" if any(s.suite == "ms3" for s in samples) else None,
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
        "strict_format_rate": (
            sum(record.get("valid_output") is True for record in completed) / len(samples)
            if samples else 0.0
        ),
        "non_empty_prediction_count": len(non_empty),
        "final_output_conforming_count": sum(
            record.get("final_output_conforming") is True for record in completed
        ),
        "artifact_answer_count": 0,
        "mcp_tool_call_count": sum(len(record.get("mcp_tool_calls") or []) for record in completed),
        "model_failure_count": sum(
            record.get("status") == "completed" and record.get("valid_output") is not True
            or record.get("failure_class") in {"model_max_tokens", "model_or_protocol_failure"}
            for record in records
        ),
        "infra_retry_count": sum(max(int(record.get("attempt") or 1) - 1, 0) for record in records),
        "infra_failure_count": sum(record.get("failure_class") == "retryable_infra" for record in records),
        "publishable": publishable_records(records),
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
    system_prompt_file: Path,
    tokenizer_config: Path | None,
    runtime_audit_file: Path | None,
) -> dict[str, Any]:
    prompt_prefix_file = skill_source / "prompt_prefix.md"
    prompt_prefix = prompt_prefix_file.read_text(encoding="utf-8")
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
        "workspace_layout": ".agents/skills/<skill-name>",
        "suites": sorted(suites),
        "limit_per_suite": limit_per_suite,
        "sample_count": len(samples),
        "sample_ids": [sample.task_id for sample in samples],
        "task_prompt_sha256": {
            sample.task_id: hashlib.sha256(
                task_prompt(sample, prompt_prefix).encode("utf-8")
            ).hexdigest()
            for sample in samples
        },
        "benchmark_integrity": {
            "ms2_training_overlap_excluded": 0,
            "training_isolation": "all_112_upstream_ms_tasks; MO overlap audit recorded in extension experiment manifest",
            "answers_written_to_workspaces": False,
            "runner_records_written_to_workspaces": False,
        },
        "evaluation_contract": {
            "contract_version": CONTRACT_VERSION,
            "native_thinking": True,
            "temperature": 0.0,
            "max_output_tokens": 16384,
            "max_prompt_tokens": 245760,
            "context_window": 262144,
            "tool_protocol": "dsh_native_structured_calls",
            "terminal_protocol": "structured_v8_strict_bare_json",
            "empty_final_is_valid": False,
            "artifact_answer_fallback": False,
        },
        "benchmark_release_sha256": sha256_file(molbench_root / "aligned/manifest.json"),
        "mo_scoring_source_files": [
            {"path": str(path.relative_to(molbench_root)), "sha256": sha256_file(path)}
            for path in sorted((molbench_root / "ChemCoTBench/baseline_and_eval/eval").glob("*.py"))
        ] if suites & {"mo-opt", "mo-edit"} else [],
        "mo_benchmark_release_sha256": (
            sha256_file(molbench_root / "aligned-mo/manifest.json")
            if suites & {"mo-opt", "mo-edit"} else None
        ),
        "tool_set_sha256": sha256_file(DATA_PIPE_ROOT / "configs/dsh_molclaw_tool_set.json"),
        "skill_tree_sha256": tree_digest(skill_source),
        "protocol_inputs": {
            "system_prompt": {"path": str(system_prompt_file), "sha256": sha256_file(system_prompt_file)},
            "prompt_prefix": {"path": str(prompt_prefix_file), "sha256": sha256_file(prompt_prefix_file)},
            "tokenizer_config": (
                {"path": str(tokenizer_config), "sha256": sha256_file(tokenizer_config)}
                if tokenizer_config is not None else None
            ),
            "runtime_audit": (
                json.loads(runtime_audit_file.read_text(encoding="utf-8"))
                if runtime_audit_file is not None else None
            ),
        },
        "source_files": [
            {"path": str(molbench_root / PINNED_SCORER), "sha256": sha256_file(molbench_root / PINNED_SCORER)},
            {"path": str(molbench_root / "upstream" / UPSTREAM_COMMIT / MS1_CSV), "sha256": sha256_file(molbench_root / "upstream" / UPSTREAM_COMMIT / MS1_CSV)}
            if "ms1" in suites else None,
            {"path": str(molbench_root / "upstream" / UPSTREAM_COMMIT / MS2_CSV), "sha256": sha256_file(molbench_root / "upstream" / UPSTREAM_COMMIT / MS2_CSV)}
            if "ms2" in suites else None,
            {"path": str(molbench_root / "upstream" / UPSTREAM_COMMIT / MS3_CSV), "sha256": sha256_file(molbench_root / "upstream" / UPSTREAM_COMMIT / MS3_CSV)}
            if "ms3" in suites else None,
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSH + local Qwen on MolBench MS-1/MS-2/MS-3")
    parser.add_argument("--suite", action="append", choices=tuple(SUITE_TASKS), default=[])
    parser.add_argument("--limit-per-suite", type=int, default=0, help="0 means all selected samples")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-infra-failed", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--rollout-only", action="store_true",
        help="write trajectories and predictions without importing or running the evaluator",
    )
    parser.add_argument("--task-timeout-sec", type=int, default=14400)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dsh-url", default=DEFAULT_DSH_URL)
    parser.add_argument("--model-provider", default="slime-local")
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument("--agent-preset", default=DEFAULT_AGENT_PRESET)
    parser.add_argument("--molbench-root", type=Path, default=DEFAULT_MOLBENCH_ROOT)
    parser.add_argument("--dsh-repo", type=Path, default=DEFAULT_DSH_REPO)
    parser.add_argument("--skill-source", type=Path, default=DEFAULT_SKILL_SOURCE)
    parser.add_argument("--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--runtime-audit-file", type=Path)
    parser.add_argument("--required-mcp-tools", type=int, default=81)
    parser.add_argument("--required-skill-count", type=int, default=52)
    parser.add_argument("--protocol-ready-timeout-sec", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollout_only and args.score_only:
        raise ValueError("--rollout-only and --score-only are mutually exclusive")
    suites = set(args.suite or ["ms1", "ms2"])
    molbench_root = args.molbench_root.expanduser().resolve()
    dsh_repo = args.dsh_repo.expanduser().resolve()
    skill_source = args.skill_source.expanduser().resolve()
    system_prompt_file = args.system_prompt_file.expanduser().resolve()
    tokenizer_config = args.tokenizer_config.expanduser().resolve() if args.tokenizer_config else None
    runtime_audit_file = args.runtime_audit_file.expanduser().resolve() if args.runtime_audit_file else None
    dsh_home = Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh").expanduser().resolve()
    if args.task_timeout_sec <= 0:
        raise ValueError("--task-timeout-sec must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.protocol_ready_timeout_sec <= 0:
        raise ValueError("--protocol-ready-timeout-sec must be positive")
    samples = load_samples(molbench_root, suites, args.limit_per_suite)
    installed_preset = None
    if not args.score_only and args.agent_preset == SKILL_PRESET_ID:
        installed_preset = install_eval_preset(
            dsh_repo, dsh_home, args.agent_preset, system_prompt_file,
        )

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
        if not args.score_only and manifest.get('execution_settings') != {
                'task_timeout_sec': args.task_timeout_sec, 'max_workers': args.max_workers,
                'max_infrastructure_retries': 2}:
            raise RuntimeError('execution budgets differ from the existing run')
        if manifest.get("benchmark_release_sha256") != sha256_file(molbench_root / "aligned/manifest.json"):
            raise RuntimeError("benchmark release changed; use a new run directory")
        if manifest.get("tool_set_sha256") != sha256_file(DATA_PIPE_ROOT / "configs/dsh_molclaw_tool_set.json") or manifest.get("skill_tree_sha256") != tree_digest(skill_source):
            raise RuntimeError("tools or skills changed; use a new run directory")
        if not args.score_only and (manifest.get("model") != args.model_id or manifest.get("model_provider") != args.model_provider):
            raise RuntimeError("model differs from the existing run")
        expected_hashes = {sample.task_id: hashlib.sha256(task_prompt(sample, (skill_source / "prompt_prefix.md").read_text()).encode()).hexdigest() for sample in samples}
        if manifest.get("task_prompt_sha256") != expected_hashes:
            raise RuntimeError("task prompts changed; use a new run directory")
        if manifest.get("protocol_inputs", {}).get("system_prompt", {}).get("sha256") != sha256_file(system_prompt_file):
            raise RuntimeError("system prompt changed; use a new run directory")
        if manifest.get("sample_ids") != [sample.task_id for sample in samples]:
            raise RuntimeError("resume selection does not match the existing run manifest")
    else:
        manifest = build_manifest(
            run_dir, molbench_root, dsh_repo, skill_source, suites,
            args.limit_per_suite, samples, args.dsh_url,
            args.model_provider, args.model_id, args.agent_preset,
            system_prompt_file, tokenizer_config, runtime_audit_file,
        )
        manifest["execution_mode"] = "rollout-only" if args.rollout_only else "rollout-and-score"
        manifest['execution_settings'] = {'task_timeout_sec': args.task_timeout_sec,
                                          'max_workers': args.max_workers, 'max_infrastructure_retries': 2}
        if installed_preset is not None:
            preset_composition = installed_preset / "agent.cordis.yml"
            manifest["derived_agent_preset"] = {
                "path": str(installed_preset),
                "composition_sha256": sha256_file(preset_composition),
                "complete_system_prompt": True,
                "include_runtime_context": False,
            }
        manifest["source_files"] = [item for item in manifest["source_files"] if item is not None]
        write_json(manifest_path, manifest)

    workspace_payload = create_workspace_snapshot(run_dir, skill_source)
    prompt_prefix = (workspace_payload / "prompt_prefix.md").read_text(encoding="utf-8")
    workdirs = {
        sample.task_id: prepare_workspace(run_dir, sample, workspace_payload)
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
            if existing:
                existing_class = existing.get("failure_class") or failure_class(existing)
                attempt = int(existing.get("attempt") or 1)
                if not args.retry_infra_failed or existing_class != "retryable_infra" or attempt >= 3:
                    print(
                        f"[{index}/{len(samples)}] {sample.task_id}: resume skip failed "
                        f"class={existing_class} attempt={attempt}",
                        flush=True,
                    )
                    continue
                archive_failed_attempt(run_dir, sample, existing)
                shutil.rmtree(workdirs[sample.task_id])
                workdirs[sample.task_id] = prepare_workspace(run_dir, sample, workspace_payload)
            pending.append((index, sample))

        if pending and (args.required_mcp_tools or args.required_skill_count):
            probe = wait_for_protocol_ready(
                api,
                workdirs[pending[0][1].task_id],
                args.model_provider,
                args.model_id,
                args.agent_preset,
                system_prompt_file.read_text(encoding="utf-8").rstrip(),
                args.required_mcp_tools,
                args.required_skill_count,
                args.protocol_ready_timeout_sec,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            probes = manifest.setdefault("startup_protocol_probes", [])
            if not isinstance(probes, list):
                raise RuntimeError("run manifest has malformed startup_protocol_probes")
            probes.append(probe)
            manifest["startup_protocol_probe"] = probe
            write_json(manifest_path, manifest)

        def execute(item: tuple[int, Sample]) -> tuple[int, Sample, dict[str, Any]]:
            index, sample = item
            print(f"[{index}/{len(samples)}] {sample.task_id}: running", flush=True)
            existing = load_record(run_dir, sample)
            attempt = int(existing.get("attempt") or 0) + 1 if existing else 1
            record = run_sample(
                DshApi(args.dsh_url), run_dir, sample, workdirs[sample.task_id], args.task_timeout_sec,
                args.model_provider, args.model_id, args.agent_preset, prompt_prefix, attempt,
                system_prompt_file.read_text(encoding="utf-8").rstrip()
                if args.required_mcp_tools or args.required_skill_count else None,
                args.required_mcp_tools,
                args.required_skill_count,
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
