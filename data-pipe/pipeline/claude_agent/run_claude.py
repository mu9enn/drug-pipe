#!/usr/bin/env python3
"""Run MolBench tasks (VS/AC/PF/E2E/KG) with Claude CLI + cc-switch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

try:
    from pipeline.claude_agent.session_capture import (
        next_attempt_index,
        run_stream_json,
        select_attempt,
    )
except ModuleNotFoundError:  # Direct script execution from launch_claude.sh.
    from session_capture import next_attempt_index, run_stream_json, select_attempt

try:
    from pipeline.kg.tool_admission import (
        POLICY_PATH as TOOL_CONCURRENCY_POLICY_PATH,
        expected_tools_from_task_spec,
        first_admissible_index,
        load_tool_limits,
        serial_tool_claims,
        task_spec_from_raw_question_json,
    )
except ModuleNotFoundError:  # Direct script execution from launch_claude.sh.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.kg.tool_admission import (
        POLICY_PATH as TOOL_CONCURRENCY_POLICY_PATH,
        expected_tools_from_task_spec,
        first_admissible_index,
        load_tool_limits,
        serial_tool_claims,
        task_spec_from_raw_question_json,
    )


ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
SOLUTION_RE = re.compile(r"<solution>([\s\S]*?)</solution>", re.IGNORECASE)
SMILES_LINE_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+$")
SMILES_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{6,}")


@dataclass
class Sample:
    row_number: int
    dataset_index: str
    question_text: str
    raw_question_json: str
    candidates: list[str]
    answer: list[str]
    n_active: int


@dataclass
class RolloutResult:
    rollout_index: int
    sample_dir: Path
    return_code: int
    timed_out: bool
    timeout_sec: int | None
    duration_sec: float
    answer_block: str
    answer: list[str]
    parse_error: str | None
    parse_source: str
    parse_attempts: list[dict[str, Any]]
    raw_answer_len: int


def _environment_capacity(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _validate_kg_worker_capacity(max_workers: int) -> None:
    if max_workers > 4:
        raise ValueError("KG --max-workers must be <= 4")
    if max_workers <= 2:
        return
    global_capacity = _environment_capacity("CLAUDE_GATE_MAX_CONCURRENCY", 4)
    data_pipe_capacity = _environment_capacity(
        "CLAUDE_GATE_DATA_PIPE_MAX_CONCURRENCY",
        2,
    )
    if max_workers > global_capacity or max_workers > data_pipe_capacity:
        raise ValueError(
            "KG --max-workers exceeds Claude gate capacity: "
            f"requested={max_workers} global={global_capacity} data_pipe={data_pipe_capacity}"
        )


def _load_expected_mcp_servers(mcp_config_file: Path | None) -> list[str]:
    if mcp_config_file is None:
        return []
    if not mcp_config_file.is_file():
        return []
    try:
        cfg = json.loads(mcp_config_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(cfg, dict):
        return []
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict):
        return []
    out = [str(k).strip() for k in servers.keys() if str(k).strip()]
    return out


def _load_mcp_tool_timeout_ms(
    mcp_config_file: Path | None,
    server_name: str = "molclaw-scp",
) -> int | None:
    """Read the effective per-server tool timeout without retaining credentials."""
    if mcp_config_file is None or not mcp_config_file.is_file():
        return None
    try:
        cfg = json.loads(mcp_config_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    servers = cfg.get("mcpServers") if isinstance(cfg, dict) else None
    server = servers.get(server_name) if isinstance(servers, dict) else None
    timeout = server.get("timeout") if isinstance(server, dict) else None
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1000:
        return None
    return timeout


def _check_session_mcp_ready(
    session_path: Path,
    expected_mcp_servers: list[str],
) -> tuple[bool, str, dict[str, Any]]:
    if not session_path.is_file():
        return False, "missing_session_file", {}

    init_obj: dict[str, Any] | None = None
    tool_uses: dict[str, str] = {}
    completed_tool_result_ids: set[str] = set()
    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "system" and obj.get("subtype") == "init":
                init_obj = obj
            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_use":
                    tool_id = str(item.get("id") or "").strip()
                    tool_name = str(item.get("name") or "").strip()
                    if tool_id and tool_name.startswith("mcp__"):
                        tool_uses[tool_id] = tool_name
                elif item.get("type") == "tool_result":
                    tool_id = str(item.get("tool_use_id") or "").strip()
                    if tool_id and item.get("is_error") is not True:
                        completed_tool_result_ids.add(tool_id)

    tools = init_obj.get("tools") if init_obj is not None else []
    tools = tools if isinstance(tools, list) else []
    mcp_tools = [t for t in tools if isinstance(t, str) and t.startswith("mcp__")]

    mcp_servers = init_obj.get("mcp_servers") if init_obj is not None else []
    mcp_servers = mcp_servers if isinstance(mcp_servers, list) else []
    status_by_name: dict[str, str] = {}
    for item in mcp_servers:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        status_by_name[name] = str(item.get("status") or "").strip().lower()

    observed_by_server: dict[str, list[str]] = {}
    for server_name in expected_mcp_servers:
        prefix = f"mcp__{server_name}__"
        observed_by_server[server_name] = sorted(
            {
                tool_name
                for tool_id, tool_name in tool_uses.items()
                if tool_id in completed_tool_result_ids and tool_name.startswith(prefix)
            }
        )

    snapshot = {
        "mcp_tools_count": len(mcp_tools),
        "mcp_servers": status_by_name,
        "observed_mcp_tool_results": observed_by_server,
    }

    if expected_mcp_servers:
        for name in expected_mcp_servers:
            if status_by_name.get(name) != "connected" and not observed_by_server.get(name):
                got = status_by_name.get(name, "missing")
                return False, f"mcp_server_not_connected:{name}:{got}", snapshot
    elif status_by_name and not any(v == "connected" for v in status_by_name.values()):
        return False, "mcp_server_not_connected:any", snapshot

    observed_any = any(observed_by_server.values())
    if not mcp_tools and not observed_any:
        if init_obj is None:
            return False, "missing_system_init_event", snapshot
        return False, "mcp_tools_missing_in_init", snapshot

    if observed_any and any(status_by_name.get(name) != "connected" for name in expected_mcp_servers):
        return True, "observed_mcp_tool_result", snapshot
    return True, "ok", snapshot


def _safe_name(text: str) -> str:
    s = (text or "sample").strip().lower()
    s = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "sample"


def _rollout_dir(sample_root: Path, num_rollouts: int, rollout_index: int) -> Path:
    if num_rollouts <= 1:
        return sample_root
    return sample_root / f"rollout{rollout_index:04d}"


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _switch_provider(provider: str) -> None:
    # cmd = ["cc-switch", "provider", "switch", provider]
    # proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    # if proc.returncode != 0:
    #     raise RuntimeError(
    #         "cc-switch failed: "
    #         f"cmd={' '.join(cmd)} stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
    #     )
    # Provider switching is now expected to be done externally before execution.
    return


def _extract_answer_block(text: str) -> str:
    raw = text or ""
    m = ANSWER_RE.search(raw)
    if m:
        return m.group(1).strip()
    m = SOLUTION_RE.search(raw)
    if m:
        return m.group(1).strip()
    return ""


def _extract_text_from_stream_jsonl(session_path: Path) -> str:
    if not session_path.is_file():
        return ""
    chunks: list[str] = []
    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") != "assistant":
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt)
            elif isinstance(content, str) and content.strip():
                chunks.append(content)
    return "\n".join(chunks)


def _extract_result_text_from_stream_jsonl(session_path: Path) -> str:
    if not session_path.is_file():
        return ""

    last_result = ""
    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if str(obj.get("type") or "") != "result":
                continue
            result = obj.get("result")
            if isinstance(result, str) and result.strip():
                last_result = result.strip()
    return last_result


def _extract_code_block_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    m = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\n([\s\S]*?)```", text)
    if not m:
        return ""
    return "\n".join(s.strip() for s in m if s and s.strip()).strip()


def _extract_json_blocks(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    blocks: list[str] = []

    # Capture JSON arrays/objects inside markdown code blocks first.
    for code in re.findall(r"```(?:json|JSON)?\n([\s\S]*?)```", text):
        s = code.strip()
        if s.startswith("[") or s.startswith("{"):
            blocks.append(s)

    # Generic greedy extraction for list/object fragments.
    for m in re.findall(r"(\[[\s\S]*?\]|\{[\s\S]*?\})", text):
        s = m.strip()
        if s:
            blocks.append(s)
    return blocks


def _parse_json_list(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if x is not None and str(x).strip()]


def _parse_lines_or_json(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []

    for cand in [text, text.replace("'", '"')]:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            out = [str(x).strip() for x in parsed if str(x).strip()]
            if out:
                return out
        elif isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("```"):
            continue
        if s.startswith("- ") or s.startswith("* "):
            s = s[2:].strip()
        s = re.sub(r"^\d+\.\s+", "", s).strip()
        s = s.strip("`\"'")
        if s:
            lines.append(s)
    return lines


def _filter_smiles_like(lines: list[str], keep_original_if_empty: bool = False) -> list[str]:
    out = [s for s in lines if SMILES_LINE_RE.match(s) and any(ch.isalpha() for ch in s)]
    if out:
        return out
    return lines if keep_original_if_empty else []


@lru_cache(maxsize=1)
def _load_rdkit_chem() -> Any | None:
    try:
        from rdkit import Chem  # type: ignore

        return Chem
    except Exception:
        return None


def _canonical_if_valid(smiles_list: list[str]) -> list[str]:
    if not smiles_list:
        return []
    chem = _load_rdkit_chem()
    if chem is None:
        return smiles_list

    out: list[str] = []
    seen: set[str] = set()
    for s in smiles_list:
        mol = chem.MolFromSmiles(s)
        if mol is None:
            continue
        c = chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _extract_smiles_tokens(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    candidates = [tok.strip("`\"'.,;:") for tok in SMILES_TOKEN_RE.findall(text)]
    candidates = [x for x in candidates if _looks_like_smiles_token(x)]
    return candidates


def _looks_like_smiles_token(token: str) -> bool:
    s = (token or "").strip()
    if not s:
        return False
    if not any(ch.isalpha() for ch in s):
        return False
    # Avoid common English tokens that appear in logs.
    if re.fullmatch(r"[A-Za-z_]+", s):
        return False
    # Typical SMILES tends to include at least one structural marker.
    markers = set("[]()=#@\\/+-.0123456789")
    return any(ch in markers for ch in s)


def _try_parse_answer_array(answer_block: str) -> tuple[list[str] | None, str | None]:
    if not answer_block:
        return None, "no <answer>/<solution> block found"
    block = answer_block.strip()
    parsed = None
    first_error: str | None = None

    def _load_candidate(text: str) -> Any:
        s = (text or "").strip()
        if not s:
            raise json.JSONDecodeError("empty", s, 0)
        return json.loads(s)

    candidates: list[str] = [block]
    try:
        candidates.append(bytes(block, "utf-8").decode("unicode_escape").strip())
    except Exception:
        pass

    for cand in candidates:
        try:
            parsed = _load_candidate(cand)
        except json.JSONDecodeError as e:
            if first_error is None:
                first_error = str(e)
            l = cand.find("[")
            r = cand.rfind("]")
            if l != -1 and r != -1 and r > l:
                sub = cand[l : r + 1].strip()
                try:
                    parsed = _load_candidate(sub)
                except json.JSONDecodeError:
                    continue
                else:
                    break
            continue
        else:
            break

    if parsed is None:
        return None, f"answer is not valid JSON: {first_error or 'unknown parse error'}"

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as e:
            return None, f"answer JSON string is not a valid JSON array: {e}"
    if not isinstance(parsed, list):
        return None, "answer JSON is not an array"
    if not all(isinstance(x, str) for x in parsed):
        return None, "answer array contains non-string entries"
    return [str(x).strip() for x in parsed if str(x).strip()], None


def _parse_answer_by_task(task: str, answer_block: str) -> tuple[list[str], str | None]:
    if task == "vs":
        parsed, err = _try_parse_answer_array(answer_block)
        return parsed or [], err

    if task in {"e2e", "kg"}:
        text = (answer_block or "").strip()
        return ([text] if text else []), None

    parsed = _filter_smiles_like(_parse_lines_or_json(answer_block), keep_original_if_empty=False)
    if task == "ac":
        if not parsed:
            return [], "empty answer for AC"
        return [parsed[0]], None

    if not parsed:
        return [], "empty answer for PF"
    return parsed, None


def _collect_parse_candidates(
    *,
    task: str,
    answer_block: str,
    result_text: str,
    session_text: str,
    raw_transcript: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    def _add(source: str, text: str) -> None:
        s = (text or "").strip()
        if s:
            candidates.append((source, s))

    _add("answer_tag", answer_block)
    _add("result_code_block", _extract_code_block_text(result_text))
    _add("result_text", result_text)

    for blk in _extract_json_blocks(result_text):
        _add("result_json_block", blk)
    for blk in _extract_json_blocks(session_text):
        _add("session_json_block", blk)

    _add("session_text", session_text)
    _add("transcript_text", raw_transcript)

    if task in {"ac", "pf"}:
        # Last resort: pull SMILES-like tokens from assistant outputs.
        _add("result_smiles_tokens", "\n".join(_extract_smiles_tokens(result_text)))
        _add("session_smiles_tokens", "\n".join(_extract_smiles_tokens(session_text)))

    # De-duplicate while preserving order.
    uniq: list[tuple[str, str]] = []
    seen: set[str] = set()
    for src, txt in candidates:
        key = f"{src}\n{txt}"
        if key in seen:
            continue
        seen.add(key)
        uniq.append((src, txt))
    return uniq


def _parse_answer_with_fallback(
    *,
    task: str,
    answer_block: str,
    result_text: str,
    session_text: str,
    raw_transcript: str,
) -> tuple[list[str], str | None, str, list[dict[str, Any]], int]:
    attempts: list[dict[str, Any]] = []
    raw_answer_len = len((answer_block or "").strip())

    if task in {"e2e", "kg"}:
        for source, text in (
            ("answer_tag", answer_block),
            ("result_text", result_text),
            ("session_text", session_text),
            ("transcript_text", raw_transcript),
        ):
            s = (text or "").strip()
            attempts.append({"source": source, "error": None, "count": 1 if s else 0})
            if s:
                return [s], None, source, attempts, raw_answer_len
        return [], None, "none", attempts, raw_answer_len

    first_err: str | None = None
    api_err_text = (result_text or "").strip() or (session_text or "").strip() or (answer_block or "").strip()
    if "API Error:" in api_err_text:
        return [], "api_error_response", "api_error", [{"source": "api_error", "error": "api_error_response", "count": 0}], raw_answer_len

    for source, text in _collect_parse_candidates(
        task=task,
        answer_block=answer_block,
        result_text=result_text,
        session_text=session_text,
        raw_transcript=raw_transcript,
    ):
        parsed, err = _parse_answer_by_task(task, text)
        if task in {"ac", "pf"}:
            parsed = _canonical_if_valid(parsed)
        if task == "ac" and parsed:
            parsed = [parsed[0]]
        attempts.append({"source": source, "error": err, "count": len(parsed)})
        if parsed:
            return parsed, None, source, attempts, raw_answer_len
        if first_err is None and err:
            first_err = err

    return [], first_err or f"no parseable {task} answer found", "none", attempts, raw_answer_len


def _parse_pf_gt(raw: str) -> list[str]:
    vals = _parse_json_list(raw)
    if vals:
        return vals
    return _parse_lines_or_json(raw)


def _load_samples(dataset_csv: Path, task: str) -> list[Sample]:
    samples: list[Sample] = []

    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_no, row in enumerate(reader, start=1):
            if task == "vs":
                raw_q = (row.get("questions") or "").strip()
                q_obj: dict[str, Any] = {}
                question_text = raw_q
                if raw_q:
                    try:
                        q_obj = json.loads(raw_q)
                        question_text = json.dumps(q_obj, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        q_obj = {}
                candidates = []
                if isinstance(q_obj.get("candidates"), list):
                    candidates = [str(x).strip() for x in q_obj["candidates"] if x is not None and str(x).strip()]
                samples.append(
                    Sample(
                        row_number=row_no,
                        dataset_index=str(row.get("index") or row_no),
                        question_text=question_text,
                        raw_question_json=raw_q,
                        candidates=candidates,
                        answer=_parse_json_list((row.get("answer") or "").strip()),
                        n_active=int((row.get("n_active") or 0) or 0),
                    )
                )
                continue

            if task == "ac":
                question_text = (row.get("question") or "").strip()
                gt = _parse_pf_gt((row.get("answer") or "").strip())
                if gt:
                    gt = [gt[0]]
                samples.append(
                    Sample(
                        row_number=row_no,
                        dataset_index=str(row.get("index") or row_no),
                        question_text=question_text,
                        raw_question_json="",
                        candidates=[],
                        answer=gt,
                        n_active=0,
                    )
                )
                continue

            if task == "e2e":
                question_text = (row.get("question") or row.get("prompt") or "").strip()
                raw_q = (row.get("raw_question_json") or "").strip()
                samples.append(
                    Sample(
                        row_number=row_no,
                        dataset_index=str(row.get("question_id") or row.get("index") or row_no),
                        question_text=question_text,
                        raw_question_json=raw_q,
                        candidates=[],
                        answer=_parse_pf_gt((row.get("answer") or "").strip()),
                        n_active=0,
                    )
                )
                continue

            if task == "kg":
                question_text = (row.get("question") or row.get("prompt") or "").strip()
                raw_q = (row.get("raw_question_json") or "").strip()
                samples.append(
                    Sample(
                        row_number=row_no,
                        dataset_index=str(row.get("question_id") or row.get("index") or row_no),
                        question_text=question_text,
                        raw_question_json=raw_q,
                        candidates=[],
                        answer=_parse_pf_gt((row.get("answer") or "").strip()),
                        n_active=0,
                    )
                )
                continue

            # PF
            question_text = (row.get("prompt") or row.get("question") or "").strip()
            gt = _parse_pf_gt((row.get("answer") or "").strip())
            samples.append(
                Sample(
                    row_number=row_no,
                    dataset_index=str(row.get("index") or row_no),
                    question_text=question_text,
                    raw_question_json="",
                    candidates=[],
                    answer=gt,
                    n_active=0,
                )
            )

    return samples


def _build_user_prompt(question_text: str, task: str) -> str:
    label_map = {
        "vs": "Question payload (MolBench-VS):",
        "ac": "Question payload (MolBench-AC):",
        "pf": "Question payload (MolBench-PF):",
        "e2e": "Question payload (MolBench-E2E):",
        "kg": "Question payload (KG-Sampled Task):",
    }
    return (
        label_map[task]
        + "\n"
        + question_text.strip()
        + "\n"
    )


def _run_one(
    claude_bin: str,
    prompt: str,
    system_prompt: str,
    workdir: Path,
    archive_root: Path | None = None,
    attempt_index: int | None = None,
    mcp_config_file: Path | None = None,
    strict_mcp_config: bool = False,
) -> dict[str, Any]:
    cmd = [
        claude_bin,
        "--dangerously-skip-permissions",
        "--verbose",
        "--output-format",
        "stream-json",
    ]
    if mcp_config_file is not None:
        cmd.extend(["--mcp-config", str(mcp_config_file)])
        if strict_mcp_config:
            cmd.append("--strict-mcp-config")
    cmd.extend(
        [
        "--system-prompt",
        system_prompt,
        "-p",
        prompt,
        ]
    )

    result = run_stream_json(
        cmd,
        cwd=workdir,
        archive_root=archive_root or workdir,
        attempt_index=attempt_index,
    )
    result["command"] = cmd
    return result


def _prepare_claude_workdir(
    target: Path,
    *,
    source_claude_dir: Path,
    question_payload: dict[str, Any],
    prompt: str,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    _copy_tree(source_claude_dir, target / ".claude")
    (target / "question.json").write_text(
        json.dumps(question_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "prompt.txt").write_text(prompt, encoding="utf-8")


def _promote_attempt_workdir(
    source: Path,
    canonical: Path,
    *,
    attempt_index: int,
) -> None:
    """Project one selected attempt into the canonical sample directory."""
    if not source.is_dir():
        raise FileNotFoundError(f"Selected Claude attempt workdir not found: {source}")
    canonical.mkdir(parents=True, exist_ok=True)
    manifest_path = canonical / "selected_attempt_artifacts.json"
    previous_manifest = _safe_read_json(manifest_path)
    for name in previous_manifest.get("promoted_entries", []):
        if not isinstance(name, str) or not name or Path(name).name != name:
            continue
        target = canonical / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()

    immutable_inputs = {".claude", "question.json", "prompt.txt"}
    promoted_entries: list[str] = []
    for item in source.iterdir():
        if item.name in immutable_inputs:
            continue
        target = canonical / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        promoted_entries.append(item.name)
    manifest_path.write_text(
        json.dumps(
            {
                "selected_attempt": attempt_index,
                "selected_attempt_workdir": str(source),
                "promoted_entries": sorted(promoted_entries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_single_rollout(
    *,
    task: str,
    sample: Sample,
    sample_root: Path,
    rollout_index: int,
    num_rollouts: int,
    prompt: str,
    system_prompt: str,
    source_claude_dir: Path,
    provider: str,
    claude_bin: str,
    mcp_config_file: Path | None,
    strict_mcp_config: bool,
) -> RolloutResult:
    workdir = _rollout_dir(sample_root, num_rollouts, rollout_index)
    workdir.mkdir(parents=True, exist_ok=True)

    question_payload = {
        "task": task,
        "row_number": sample.row_number,
        "dataset_index": sample.dataset_index,
        "rollout_index": rollout_index,
        "num_rollouts": num_rollouts,
        "raw_question_json": sample.raw_question_json,
        "question_text": sample.question_text,
        "candidates": sample.candidates,
        "answer": sample.answer,
        "n_active": sample.n_active,
    }
    if task == "kg":
        kg_task_spec: dict[str, Any] = {}
        raw_spec = (sample.raw_question_json or "").strip()
        if raw_spec:
            try:
                parsed_spec = json.loads(raw_spec)
            except json.JSONDecodeError:
                parsed_spec = {}
            if isinstance(parsed_spec, dict):
                kg_task_spec = parsed_spec
        question_payload["kg_task_spec"] = kg_task_spec
    _prepare_claude_workdir(
        workdir,
        source_claude_dir=source_claude_dir,
        question_payload=question_payload,
        prompt=prompt,
    )

    session_path = workdir / "complete_session.jsonl"
    expected_mcp_servers = _load_expected_mcp_servers(mcp_config_file)
    mcp_tool_timeout_ms = _load_mcp_tool_timeout_ms(mcp_config_file)
    enforce_mcp_ready = bool(expected_mcp_servers)
    max_ready_retries = max(0, int(os.environ.get("CLAUDE_MCP_READY_RETRIES", "2")))
    ready_retry_wait_sec = max(0.0, float(os.environ.get("CLAUDE_MCP_READY_RETRY_WAIT_SEC", "2")))
    mcp_ready = not enforce_mcp_ready
    mcp_ready_reason = "mcp_check_skipped_no_config"
    mcp_snapshot: dict[str, Any] = {}
    mcp_attempts = 0
    claude_attempts: list[dict[str, Any]] = []

    while True:
        mcp_attempts += 1
        archive_attempt_index = next_attempt_index(workdir)
        attempt_workdir = (
            workdir
            / "attempts"
            / f"attempt_{archive_attempt_index:04d}"
            / "workdir"
        )
        _prepare_claude_workdir(
            attempt_workdir,
            source_claude_dir=source_claude_dir,
            question_payload=question_payload,
            prompt=prompt,
        )
        cli_meta = _run_one(
            claude_bin=claude_bin,
            prompt=prompt,
            system_prompt=system_prompt,
            workdir=attempt_workdir,
            archive_root=workdir,
            attempt_index=archive_attempt_index,
            mcp_config_file=mcp_config_file,
            strict_mcp_config=strict_mcp_config,
        )
        claude_attempts.append(cli_meta)
        attempt_session_path = Path(str(cli_meta["session_file"]))

        if not enforce_mcp_ready:
            mcp_ready = True
            mcp_ready_reason = "mcp_check_skipped_no_expected_server"
            cli_meta["mcp_ready"] = mcp_ready
            cli_meta["mcp_ready_reason"] = mcp_ready_reason
            cli_meta["mcp_snapshot"] = mcp_snapshot
            break

        mcp_ready, mcp_ready_reason, mcp_snapshot = _check_session_mcp_ready(
            session_path=attempt_session_path,
            expected_mcp_servers=expected_mcp_servers,
        )
        cli_meta["mcp_ready"] = mcp_ready
        cli_meta["mcp_ready_reason"] = mcp_ready_reason
        cli_meta["mcp_snapshot"] = mcp_snapshot
        if mcp_ready:
            break
        if int(cli_meta.get("return_code", 0)) in {124, 127}:
            break
        if mcp_attempts > max_ready_retries:
            break
        time.sleep(ready_retry_wait_sec)

    selected_attempt_workdir = Path(str(cli_meta["workdir"]))
    _promote_attempt_workdir(
        selected_attempt_workdir,
        workdir,
        attempt_index=int(cli_meta["attempt_index"]),
    )
    selected_session = select_attempt(cli_meta, session_path)
    if not bool(cli_meta.get("raw_session_valid")) and int(cli_meta.get("return_code", 0)) == 0:
        cli_meta["return_code"] = 97
        cli_meta["failure"] = "raw_session_invalid"
    if enforce_mcp_ready and not mcp_ready and int(cli_meta.get("return_code", 0)) == 0:
        cli_meta["return_code"] = 98

    session_text = _extract_text_from_stream_jsonl(session_path)
    answer_block = _extract_answer_block(session_text)
    result_text = _extract_result_text_from_stream_jsonl(session_path)
    raw_transcript = session_path.read_text(encoding="utf-8", errors="ignore") if session_path.exists() else ""
    if not answer_block:
        answer_block = _extract_answer_block(raw_transcript)
    if not answer_block and task in {"ac", "pf", "e2e", "kg"}:
        # The canonical MolClaw prompt may omit XML tags; fall back to final result text.
        answer_block = _extract_code_block_text(result_text) or result_text or session_text

    parsed_answer, parse_error, parse_source, parse_attempts, raw_answer_len = _parse_answer_with_fallback(
        task=task,
        answer_block=answer_block,
        result_text=result_text,
        session_text=session_text,
        raw_transcript=raw_transcript,
    )
    if task in {"e2e", "kg"}:
        # E2E keeps raw final output and does not enforce parse-error gating.
        parse_error = None
    if enforce_mcp_ready and not mcp_ready:
        parsed_answer = []
        parse_error = f"mcp_not_ready:{mcp_ready_reason}"
        parse_source = "mcp_not_ready"
        parse_attempts = [{"source": "mcp_not_ready", "error": parse_error, "count": 0}]
        answer_block = ""
        raw_answer_len = 0
    parsed_payload = {
        "task": task,
        "row_number": sample.row_number,
        "dataset_index": sample.dataset_index,
        "rollout_index": rollout_index,
        "answer_block": answer_block,
        "answer_size": len(parsed_answer),
        "answer": parsed_answer,
        "parse_error": parse_error,
        "parse_source": parse_source,
        "parse_attempts": parse_attempts,
        "raw_answer_len": raw_answer_len,
        "timed_out": bool(cli_meta.get("timed_out")),
        "timeout_sec": cli_meta.get("timeout_sec"),
    }
    (workdir / "parsed_answer.json").write_text(
        json.dumps(parsed_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_meta = {
        "task": task,
        "provider": provider,
        "sample_dir": str(workdir),
        "session_file": str(session_path),
        "rollout_index": rollout_index,
        "return_code": cli_meta["return_code"],
        "timed_out": bool(cli_meta.get("timed_out")),
        "timeout_sec": cli_meta.get("timeout_sec"),
        "duration_sec": cli_meta["duration_sec"],
        "command": cli_meta["command"],
        "mcp_ready": bool(mcp_ready),
        "mcp_ready_reason": mcp_ready_reason,
        "mcp_attempts": mcp_attempts,
        "mcp_snapshot": mcp_snapshot,
        "mcp_tool_timeout_ms": mcp_tool_timeout_ms,
        "claude_attempts": claude_attempts,
        "selected_claude_attempt": int(cli_meta["attempt_index"]),
        "selected_attempt_workdir": str(selected_attempt_workdir),
        "selected_session_byte_count": selected_session["byte_count"],
        "selected_session_sha256": selected_session["sha256"],
        "raw_session_valid": bool(selected_session["raw_session_valid"]),
    }
    (workdir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return RolloutResult(
        rollout_index=rollout_index,
        sample_dir=workdir,
        return_code=cli_meta["return_code"],
        timed_out=bool(cli_meta.get("timed_out")),
        timeout_sec=cli_meta.get("timeout_sec"),
        duration_sec=float(cli_meta["duration_sec"]),
        answer_block=answer_block,
        answer=parsed_answer,
        parse_error=parse_error,
        parse_source=parse_source,
        parse_attempts=parse_attempts,
        raw_answer_len=raw_answer_len,
    )


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _check_run_completeness(run_dir: Path, num_rollouts: int, task: str) -> dict[str, Any]:
    row_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("row") and "_idx" in p.name])
    summary_path = run_dir / "run_summary.jsonl"

    expected_pairs: set[tuple[int, int]] = set()
    summary_lines = 0
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                summary_lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_no = int(obj.get("row_number") or -1)
                r_idx = int(obj.get("rollout_index") or 1)
                if row_no >= 1 and r_idx >= 1:
                    expected_pairs.add((row_no, r_idx))

    missing_files: list[dict[str, Any]] = []
    checked_samples = 0

    for row_dir in row_dirs:
        q_path = row_dir / "question.json"
        q = _safe_read_json(q_path)
        row_no = int(q.get("row_number") or -1)
        if row_no < 0:
            try:
                row_no = int(row_dir.name.split("_idx", 1)[0].replace("row", ""))
            except Exception:
                row_no = -1

        rollout_dirs = sorted([p for p in row_dir.iterdir() if p.is_dir() and p.name.startswith("rollout")])
        if rollout_dirs:
            sample_dirs = rollout_dirs
        else:
            sample_dirs = [row_dir]

        for sdir in sample_dirs:
            checked_samples += 1
            parsed = _safe_read_json(sdir / "parsed_answer.json")
            rollout_index = int(parsed.get("rollout_index") or 1)
            pair = (row_no, rollout_index)
            if pair not in expected_pairs:
                missing_files.append(
                    {
                        "sample_dir": str(sdir),
                        "missing": [f"run_summary entry missing for pair={pair}"],
                    }
                )
            required = [
                sdir / "complete_session.jsonl",
                sdir / "parsed_answer.json",
                sdir / "run_meta.json",
                sdir / "prompt.txt",
                sdir / "question.json",
            ]
            miss = [str(p.name) for p in required if not p.exists()]
            if miss:
                missing_files.append({"sample_dir": str(sdir), "missing": miss})

    preds_root = run_dir / "preds" / f"molbench_{task}"
    preds_main = preds_root / f"molbench_{task}.json"
    rollouts_root = preds_root / "rollouts"

    pred_issues: list[str] = []
    if not preds_main.is_file():
        pred_issues.append(f"missing {preds_main}")
    if not rollouts_root.is_dir():
        pred_issues.append(f"missing {rollouts_root}")
    else:
        for r_idx in range(1, num_rollouts + 1):
            rp = rollouts_root / f"rollout_{r_idx:04d}.json"
            if not rp.is_file():
                pred_issues.append(f"missing {rp}")

    expected_total = len(row_dirs) * max(1, num_rollouts)
    completeness_ok = (len(missing_files) == 0) and (len(pred_issues) == 0) and (len(expected_pairs) == expected_total)

    report = {
        "task": task,
        "run_dir": str(run_dir),
        "row_count": len(row_dirs),
        "num_rollouts": num_rollouts,
        "expected_sample_pairs": expected_total,
        "run_summary_lines": summary_lines,
        "run_summary_pairs": len(expected_pairs),
        "checked_samples": checked_samples,
        "missing_files": missing_files,
        "prediction_issues": pred_issues,
        "completeness_ok": completeness_ok,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MolBench tasks with Claude CLI and stream-json logs.")
    parser.add_argument("--task", choices=["vs", "ac", "pf", "e2e", "kg"], default="vs")
    parser.add_argument("--dataset-csv", default="molbench/molbench-vs-900.csv")
    parser.add_argument("--skills-root", default="../workdir-skills/molclaw-trajectory-execution")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--system-prompt-file", default="", help="Optional prompt filename under skills root")
    parser.add_argument("--provider", default=os.environ.get("CC_SWITCH_PROVIDER", "manual"))
    parser.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    parser.add_argument("--start-row", type=int, default=1, help="1-based row index in CSV")
    parser.add_argument("--end-row", type=int, default=0, help="1-based inclusive row index; 0 means all")
    parser.add_argument("--limit", type=int, default=0, help="max number of rows after slicing; 0 means no limit")
    parser.add_argument("--num-rollouts", type=int, default=1, help="how many rollouts to sample per task row")
    parser.add_argument("--rollout-seed-base", type=int, default=0, help="metadata-only seed base for rollouts")
    parser.add_argument(
        "--parallel-rollouts",
        type=int,
        default=1,
        help="Compatibility worker setting; used only when --max-workers is not set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("MAX_WORKERS", "0") or 0),
        help="Maximum concurrent Claude invocations across task rows and rollouts; 0 uses --parallel-rollouts.",
    )
    parser.add_argument("--mcp-config-file", default="", help="Optional MCP config JSON path passed to Claude CLI")
    parser.add_argument("--strict-mcp-config", action="store_true", help="Use Claude --strict-mcp-config")
    parser.add_argument("--skip-provider-switch", action="store_true")
    args = parser.parse_args()

    if args.start_row < 1:
        raise ValueError("--start-row must be >= 1")
    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be >= 1")
    if args.parallel_rollouts < 1:
        raise ValueError("--parallel-rollouts must be >= 1")
    if args.max_workers < 0:
        raise ValueError("--max-workers must be >= 0")
    max_workers = args.max_workers or args.parallel_rollouts
    if args.task == "kg":
        _validate_kg_worker_capacity(max_workers)

    repo_root = Path(__file__).resolve().parents[2]
    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.is_absolute():
        dataset_csv = (repo_root / dataset_csv).resolve()
    skills_root = Path(args.skills_root)
    if not skills_root.is_absolute():
        skills_root = (repo_root / skills_root).resolve()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (repo_root / results_root).resolve()
    mcp_config_file: Path | None = None
    if args.mcp_config_file.strip():
        mcp_config_file = Path(args.mcp_config_file)
        if not mcp_config_file.is_absolute():
            mcp_config_file = (repo_root / mcp_config_file).resolve()

    default_prompt_name = "system_prompt.md"
    prompt_name = args.system_prompt_file.strip() or default_prompt_name
    prompt_path = Path(prompt_name)
    if prompt_path.is_absolute():
        system_prompt_file = prompt_path
    else:
        system_prompt_file = skills_root / prompt_name

    source_claude_dir = skills_root / ".claude"

    if not dataset_csv.is_file():
        raise FileNotFoundError(f"dataset csv not found: {dataset_csv}")
    if not system_prompt_file.is_file():
        raise FileNotFoundError(f"system prompt file not found: {system_prompt_file}")
    if not source_claude_dir.is_dir():
        raise FileNotFoundError(f"scene skill payload not found: {source_claude_dir}")
    if mcp_config_file is not None and not mcp_config_file.is_file():
        raise FileNotFoundError(f"mcp config file not found: {mcp_config_file}")
    mcp_tool_timeout_ms = _load_mcp_tool_timeout_ms(mcp_config_file)

    system_prompt = system_prompt_file.read_text(encoding="utf-8")
    all_samples = _load_samples(dataset_csv, args.task)

    selected = [s for s in all_samples if s.row_number >= args.start_row]
    if args.end_row and args.end_row >= args.start_row:
        selected = [s for s in selected if s.row_number <= args.end_row]
    if args.limit > 0:
        selected = selected[: args.limit]

    if not selected:
        print("No samples selected.")
        return

    tool_limits = load_tool_limits() if args.task == "kg" else {}

    if not args.skip_provider_switch:
        # _switch_provider(args.provider)
        # print(f"[run] provider switched via cc-switch: {args.provider}", flush=True)
        print("[run] provider switch step disabled in script (expect external cc-switch before run)", flush=True)
    else:
        print("[run] skip provider switch", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"molbench_{args.task}_{_safe_name(args.provider)}_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "task": args.task,
        "provider": args.provider,
        "dataset_csv": str(dataset_csv),
        "skills_root": str(skills_root),
        "system_prompt_file": str(system_prompt_file),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "num_rollouts": args.num_rollouts,
        "parallel_rollouts": args.parallel_rollouts,
        "max_workers": max_workers,
        "rollout_seed_base": args.rollout_seed_base,
        "mcp_config_file": str(mcp_config_file) if mcp_config_file is not None else "",
        "mcp_tool_timeout_ms": mcp_tool_timeout_ms,
        "strict_mcp_config": bool(args.strict_mcp_config),
        "selected_rows": len(selected),
        "tool_admission_policy": (
            {
                "mode": "strict_same_limit4_tool_serial",
                "policy_file": str(TOOL_CONCURRENCY_POLICY_PATH),
                "registered_tool_count": len(tool_limits),
            }
            if args.task == "kg"
            else None
        ),
        "timestamp": ts,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = run_dir / "run_summary.jsonl"
    rollouts_preds: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, args.num_rollouts + 1)}

    with summary_path.open("w", encoding="utf-8") as summary_f:
        print(
            f"[route] task={args.task} skills_root={skills_root} system_prompt={system_prompt_file} "
            f"mcp_config={mcp_config_file if mcp_config_file is not None else '(default)'} "
            f"strict_mcp={int(bool(args.strict_mcp_config))} mcp_tool_timeout_ms={mcp_tool_timeout_ms}",
            flush=True,
        )
        prepared: list[tuple[Sample, Path, str]] = []
        for s in selected:
            sample_name = f"row{s.row_number:04d}_idx{_safe_name(s.dataset_index)}"
            sample_root = run_dir / sample_name
            sample_root.mkdir(parents=True, exist_ok=True)

            question_base = {
                "task": args.task,
                "row_number": s.row_number,
                "dataset_index": s.dataset_index,
                "raw_question_json": s.raw_question_json,
                "question_text": s.question_text,
                "candidates": s.candidates,
                "answer": s.answer,
                "n_active": s.n_active,
                "num_rollouts": args.num_rollouts,
            }
            (sample_root / "question.json").write_text(
                json.dumps(question_base, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            prompt = _build_user_prompt(s.question_text, args.task)
            prepared.append((s, sample_root, prompt))

        results_by_row: dict[int, list[RolloutResult]] = {s.row_number: [] for s in selected}
        admission_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        jobs: list[dict[str, Any]] = []
        for s, sample_root, prompt in prepared:
            task_spec = task_spec_from_raw_question_json(s.raw_question_json)
            expected_tools = (
                expected_tools_from_task_spec(task_spec) if args.task == "kg" else ()
            )
            if args.task == "kg" and not expected_tools:
                raise ValueError(
                    f"KG task {s.dataset_index!r} is missing an expected toolchain"
                )
            claims = (
                serial_tool_claims(expected_tools, tool_limits)
                if args.task == "kg"
                else frozenset()
            )
            for rollout_index in range(1, args.num_rollouts + 1):
                jobs.append(
                    {
                        "sample": s,
                        "sample_root": sample_root,
                        "prompt": prompt,
                        "rollout_index": rollout_index,
                        "expected_tools": expected_tools,
                        "serial_tools": claims,
                        "queued_monotonic": time.monotonic(),
                        "blocked_tools": set(),
                    }
                )
        print(
            f"[run] selected_tasks={len(selected)} total_claude_invocations={len(jobs)} "
            f"max_workers={max_workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = list(jobs)
            active: dict[Any, dict[str, Any]] = {}
            progress = tqdm(total=len(jobs), desc=f"MolBench-{args.task.upper()}", unit="rollout")
            while pending or active:
                while pending and len(active) < max_workers:
                    occupied = (
                        set().union(*(job["serial_tools"] for job in active.values()))
                        if active
                        else set()
                    )
                    for job in pending:
                        job["blocked_tools"].update(job["serial_tools"].intersection(occupied))
                    pending_index = first_admissible_index(
                        [job["serial_tools"] for job in pending],
                        [job["serial_tools"] for job in active.values()],
                    )
                    if pending_index is None:
                        break
                    job = pending.pop(pending_index)
                    sample = job["sample"]
                    rollout_index = int(job["rollout_index"])
                    admission = {
                        "expected_tools": list(job["expected_tools"]),
                        "serial_tools": sorted(job["serial_tools"]),
                        "blocked_by_tools": sorted(job["blocked_tools"]),
                        "wait_sec": round(
                            time.monotonic() - float(job["queued_monotonic"]),
                            4,
                        ),
                        "admitted_at": datetime.now().astimezone().isoformat(),
                    }
                    admission_by_pair[(sample.row_number, rollout_index)] = admission
                    print(
                        f"[admit] row={sample.row_number} idx={sample.dataset_index} "
                        f"rollout={rollout_index} serial_tools={admission['serial_tools']} "
                        f"blocked_by={admission['blocked_by_tools']} "
                        f"wait_sec={admission['wait_sec']}",
                        flush=True,
                    )
                    future = executor.submit(
                        _run_single_rollout,
                        task=args.task,
                        sample=sample,
                        sample_root=job["sample_root"],
                        rollout_index=rollout_index,
                        num_rollouts=args.num_rollouts,
                        prompt=job["prompt"],
                        system_prompt=system_prompt,
                        source_claude_dir=source_claude_dir,
                        provider=args.provider,
                        claude_bin=args.claude_bin,
                        mcp_config_file=mcp_config_file,
                        strict_mcp_config=bool(args.strict_mcp_config),
                    )
                    active[future] = job
                if not active:
                    raise RuntimeError("tool admission deadlock with no active rollout")
                completed, _unfinished = wait(
                    tuple(active),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    job = active.pop(future)
                    sample = job["sample"]
                    results_by_row[sample.row_number].append(future.result())
                    progress.update(1)
            progress.close()

        for s in selected:
            results = sorted(results_by_row[s.row_number], key=lambda item: item.rollout_index)

            for rr in results:
                summary_entry = {
                    "task": args.task,
                    "row_number": s.row_number,
                    "dataset_index": s.dataset_index,
                    "rollout_index": rr.rollout_index,
                    "sample_dir": str(rr.sample_dir),
                    "return_code": rr.return_code,
                    "timed_out": rr.timed_out,
                    "timeout_sec": rr.timeout_sec,
                    "duration_sec": rr.duration_sec,
                    "answer_size": len(rr.answer),
                    "parse_error": rr.parse_error,
                    "parse_source": rr.parse_source,
                    "parse_attempts": rr.parse_attempts,
                    "raw_answer_len": rr.raw_answer_len,
                    "tool_admission": admission_by_pair.get(
                        (s.row_number, rr.rollout_index),
                        {},
                    ),
                }
                summary_f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
                summary_f.flush()

                if args.task == "vs":
                    json_results = {
                        "ranking": rr.answer,
                        "raw_answer": rr.answer_block,
                    }
                else:
                    json_results = {
                        "prediction": rr.answer,
                        "raw_answer": rr.answer_block,
                        "parse_source": rr.parse_source,
                    }

                pred_entry = {
                    "task": args.task,
                    "index": s.dataset_index,
                    "row_number": s.row_number,
                    "rollout_index": rr.rollout_index,
                    "candidates": s.candidates,
                    "answer": s.answer,
                    "n_active": s.n_active,
                    "json_results": json_results,
                    "metrics": {},
                }
                rollouts_preds[rr.rollout_index].append(pred_entry)

                tqdm.write(
                    f"[run] task={args.task} row={s.row_number} idx={s.dataset_index} rollout={rr.rollout_index} "
                    f"rc={rr.return_code} timeout={int(rr.timed_out)} answer_n={len(rr.answer)} parse={rr.parse_source}"
                )

    preds_dir = run_dir / "preds" / f"molbench_{args.task}"
    preds_dir.mkdir(parents=True, exist_ok=True)

    rollouts_dir = preds_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    for r_idx in range(1, args.num_rollouts + 1):
        out_path = rollouts_dir / f"rollout_{r_idx:04d}.json"
        out_path.write_text(
            json.dumps(rollouts_preds.get(r_idx, []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    primary_preds = rollouts_preds.get(1, [])
    (preds_dir / f"molbench_{args.task}.json").write_text(
        json.dumps(primary_preds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    completion_report = _check_run_completeness(run_dir, args.num_rollouts, args.task)
    (run_dir / "completion_report.json").write_text(
        json.dumps(completion_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not completion_report.get("completeness_ok"):
        print(f"[error] run completeness check failed: {run_dir / 'completion_report.json'}")
        raise RuntimeError("run completeness check failed")

    print(f"RESULTS_DIR={run_dir}")


if __name__ == "__main__":
    main()
