#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_TASKS = ("vs", "ac", "pf", "kg", "e2e")
SFT_SCHEMA_VERSION = "drug_agent_sft_react_json_v1"
RL_SCHEMA_VERSION = "mcp-rl-prompt-v2-all"
DEFAULT_SYSTEM_PROMPT = (
    "You are a scientific agent for MolBench-style tasks. "
    "Follow a ReAct-style protocol: write reasoning in <thought>...</thought>, "
    "write MCP calls in <tool_call>...</tool_call>, "
    "write tool observations in <observation tool_name=\"...\">...</observation>, "
    "and write the final response in <final_answer>...</final_answer>. "
    "Use only real MolClaw MCP tools and do not fabricate tool outputs."
)
DEFAULT_OBSERVATION_MAX_CHARS = 6000
TARGET_MCP_PREFIXES = (
    "mcp__molclaw-scp__",
    "mcp__molclaw-vs__",
)

TRIPLE_FENCE_RE = re.compile(r"^\s*```(?:[^\n`]*)\n([\s\S]*?)\n```\s*$")
OBS_TAG_RE = re.compile(r"<observation\s+tool_name=\"([^\"]+)\">([\s\S]*?)</observation>")
THOUGHT_TAG_RE = re.compile(r"<thought>([\s\S]*?)</thought>")
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
FINAL_TAG_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
ABS_PATH_RE = re.compile(r"/(?:root|home|tmp|mnt|workspace)/(?:[^\s\"'<>`\[\]{}(),;]+)?")
ARTIFACT_LINK_RE = re.compile(r"\[artifact:([^\]]+)\]\(artifact:[^)]+\)")
ARTIFACT_TEXT_RE = re.compile(r"<artifact:[^>]+>")

ENGINEERING_HINTS = (
    "repo",
    "workspace",
    "working directory",
    "file",
    "directory",
    "script",
    "shell",
    "bash",
    "log",
    "todo list",
    "write the file",
    "read the file",
    "inspect the repository",
    "open the file",
    "save the file",
    "git",
    "patch",
    "pytest",
    "unit test",
    "codebase",
)
SCIENCE_HINTS = (
    "smiles",
    "molecule",
    "protein",
    "docking",
    "affinity",
    "binding",
    "pocket",
    "ligand",
    "admet",
    "virtual screening",
    "target",
    "compound",
    "rank",
    "screening",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json_dir(dir_path: Path, rows: list[dict[str, Any]], *, prefix: str) -> list[Path]:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, row in enumerate(rows, 1):
        sample_id = str(row.get("id") or f"sample_{idx:06d}")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_")
        file_name = f"{idx:06d}__{safe_id or prefix}.json"
        file_path = dir_path / file_name
        _write_json(file_path, row)
        written.append(file_path)
    return written


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _preview_text(text: str, limit: int = 160) -> str:
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    if limit <= 3:
        return s[:limit]
    return s[: limit - 3] + "..."


def _artifact_placeholder_for_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return "<artifact:local/result>"
    try:
        base = Path(raw).name.strip()
    except Exception:
        base = ""
    if not base or base in {".", ".."}:
        base = "result"

    raw_low = raw.lower()
    base_low = base.lower()
    ext = Path(base).suffix.lower()

    if "fpocket" in raw_low:
        if not base or base_low in {"", "result", "results", "output", "outputs", "run", "rundir", "run_dir", "output_dir"} or "." not in base:
            base = "result"
        return f"<artifact:fpocket/{base}>"
    if any(tok in raw_low for tok in ("pdbfixer", "fixed_pdb", "fix_pdb")):
        return f"<artifact:pdbfixer/{base}>"
    if any(tok in raw_low for tok in ("docking", "exp_data")):
        return f"<artifact:docking/{base}>"
    if any(tok in raw_low for tok in ("boltz2", "boltz")):
        return f"<artifact:boltz/{base}>"
    if (
        ext in {".pdb", ".cif", ".mmcif", ".pdbqt", ".ent", ".gro", ".mol", ".mol2", ".sdf"}
        or "protein_structure" in raw_low
        or "protein_structures" in raw_low
    ):
        return f"<artifact:protein_structures/{base}>"
    return f"<artifact:local/{base}>"


def _sanitize_text_with_artifacts(text: Any, path_cache: dict[str, str]) -> str:
    s = str(text or "")
    if not s:
        return ""

    # Normalize historical markdown artifacts before protecting pure artifacts.
    s = ARTIFACT_LINK_RE.sub(lambda match: f"<artifact:{match.group(1)}>", s)
    protected: list[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"__ARTIFACT_PLACEHOLDER_{len(protected) - 1}__"

    s = ARTIFACT_TEXT_RE.sub(_protect, s)

    def _replace(match: re.Match[str]) -> str:
        raw_path = match.group(0)
        if raw_path not in path_cache:
            path_cache[raw_path] = _artifact_placeholder_for_path(raw_path)
        return path_cache[raw_path]

    s = ABS_PATH_RE.sub(_replace, s)

    for idx, original in enumerate(protected):
        s = s.replace(f"__ARTIFACT_PLACEHOLDER_{idx}__", original)
    return s


def _sanitize_structure_with_artifacts(value: Any, path_cache: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _sanitize_text_with_artifacts(value, path_cache)
    if isinstance(value, list):
        return [_sanitize_structure_with_artifacts(v, path_cache) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_structure_with_artifacts(v, path_cache) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_structure_with_artifacts(v, path_cache) for k, v in value.items()}
    return value


def _record_action(
    actions: list[dict[str, Any]],
    *,
    sample_id: str,
    field_path: str,
    cleaning_type: str,
    before_preview: Any,
    after_preview: Any,
    original_backed_up: bool = True,
    extra: dict[str, Any] | None = None,
) -> None:
    action: dict[str, Any] = {
        "sample_id": sample_id,
        "field_path": field_path,
        "cleaning_type": cleaning_type,
        "before_preview": _preview_text(str(before_preview)),
        "after_preview": _preview_text(str(after_preview)),
        "original_backed_up": bool(original_backed_up),
    }
    if extra:
        action.update(extra)
    actions.append(action)


def _strip_triple_backtick_fence(text: str, *, actions: list[dict[str, Any]], sample_id: str, field_path: str) -> tuple[str, bool]:
    s = str(text)
    m = TRIPLE_FENCE_RE.match(s)
    if not m:
        return s, False
    inner = m.group(1)
    _record_action(
        actions,
        sample_id=sample_id,
        field_path=field_path,
        cleaning_type="strip_fence_wrapper",
        before_preview=s,
        after_preview=inner,
        original_backed_up=True,
    )
    return inner, True


def _looks_engineering_chatter(text: str) -> bool:
    low = text.lower()
    if not any(h in low for h in ENGINEERING_HINTS):
        return False
    return not any(h in low for h in SCIENCE_HINTS)


def _clean_text_piece(
    text: Any,
    *,
    actions: list[dict[str, Any]],
    sample_id: str,
    field_path: str,
    allow_engineering_drop: bool,
    path_cache: dict[str, str],
) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    s = str(text)
    if not s.strip():
        return None, False

    s2, fence_stripped = _strip_triple_backtick_fence(s, actions=actions, sample_id=sample_id, field_path=field_path)
    cleaned = _sanitize_text_with_artifacts(s2.strip(), path_cache)
    changed = fence_stripped or cleaned != s.strip()
    if not cleaned:
        return None, changed

    return cleaned, changed


def _normalize_tool_name(raw_name: str) -> tuple[str | None, str | None]:
    raw_name = str(raw_name or "").strip()
    for prefix in TARGET_MCP_PREFIXES:
        if raw_name.startswith(prefix):
            norm = raw_name[len(prefix) :].strip()
            return (norm or raw_name), prefix.removeprefix("mcp__").removesuffix("__")
    return None, None


def _extract_pointer_map(value: Any) -> dict[str, str]:
    pointers: dict[str, str] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(v, str):
                continue
            key = str(k)
            if key.endswith("_path") or key.endswith("_dir") or key in {
                "path",
                "output_file",
                "output_dir",
                "output_path",
                "source_path",
                "result_path",
                "prot_structure_path",
                "workdir",
                "file_path",
                "artifact_path",
            }:
                pointers[key] = v
            elif "/" in v or v.startswith("."):
                if key not in pointers:
                    pointers[key] = v
    elif isinstance(value, str):
        s = value.strip()
        if s and ("/" in s or s.startswith(".") or s.endswith(".pdb") or s.endswith(".pdbqt") or s.endswith(".json")):
            pointers["raw_pointer"] = s
    return pointers


def _truncate_json_compatible(value: Any, limit: int) -> tuple[Any, bool, str, str]:
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except Exception:
        serialized = str(value)
        value = str(value)
    before_preview = _preview_text(serialized)
    if len(serialized) <= limit:
        return value, False, before_preview, before_preview
    truncated_preview = _preview_text(serialized, limit)
    after = {
        "preview": truncated_preview,
        "truncated": True,
    }
    after_preview = _preview_text(json.dumps(after, ensure_ascii=False))
    return after, True, before_preview, after_preview


def _load_summary_map(csv_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            copied = str(row.get("copied_path") or "").strip()
            if copied:
                resolved = str(Path(copied).resolve())
                out[f"path::{resolved}"] = row
                out[f"name::{Path(copied).name}"] = row
    return out


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _infer_run_and_row_dir(copied_path: Path, summary_row: dict[str, Any]) -> tuple[str, str]:
    run_dir = str(summary_row.get("run_dir") or "").strip()
    stem = copied_path.stem
    parts = stem.split("__")
    if not run_dir and parts:
        run_dir = parts[0]
    row_dir = ""
    if len(parts) >= 2:
        row_dir = parts[1]
    elif "row" in stem:
        m = re.search(r"(row[^_]+_idx[^_]+)", stem)
        if m:
            row_dir = m.group(1)
    return run_dir, row_dir


def _infer_run_row_rollout_dir(copied_path: Path, summary_row: dict[str, Any]) -> tuple[str, str, str]:
    run_dir, row_dir = _infer_run_and_row_dir(copied_path, summary_row)
    rollout_dir = ""
    parts = copied_path.stem.split("__")
    if len(parts) >= 3:
        rollout_dir = parts[2]
    elif "rollout" in copied_path.stem:
        m = re.search(r"(rollout\d+)", copied_path.stem)
        if m:
            rollout_dir = m.group(1)
    return run_dir, row_dir, rollout_dir


def _infer_task_from_sample_id(sample_id: str) -> str:
    m = re.match(r"^mcp_sft_(vs|ac|pf|kg|e2e)_", str(sample_id or ""))
    return m.group(1) if m else "unknown"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if v is not None and str(v).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return [s]
    return []


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            num = float(s)
        except Exception:
            return None
        if num.is_integer():
            return int(num)
        return num
    return None


def _coerce_triplet(value: Any) -> list[float | int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        out: list[float | int] = []
        for item in value[:3]:
            num = _coerce_number(item)
            if num is None:
                return None
            out.append(num)
        return out
    return None


def _first_nonempty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple)):
            s = str(value).strip()
            if s:
                return s
    return ""


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def _fpocket_candidate_list(value: Any) -> list[dict[str, Any]]:
    keys = (
        "top_pocket",
        "best_pocket",
        "selected_pocket",
        "pocket",
        "pockets",
        "pocket_list",
        "pocket_results",
        "pocket_candidates",
        "results",
        "cavities",
        "sites",
    )

    def _looks_like_pocket_dict(obj: dict[str, Any]) -> bool:
        return any(
            k in obj
            for k in (
                "center",
                "pocket_center",
                "center_x",
                "pocket_center_x",
                "size",
                "pocket_size",
                "size_x",
                "pocket_size_x",
                "score",
                "druggability_score",
                "probability",
                "chains",
                "chain_ids",
                "involved_chains",
                "residues",
            )
        )

    if isinstance(value, dict):
        for key in keys:
            cand = value.get(key)
            if isinstance(cand, dict) and _looks_like_pocket_dict(cand):
                return [cand]
            if isinstance(cand, list):
                pockets = [item for item in cand if isinstance(item, dict) and _looks_like_pocket_dict(item)]
                if pockets:
                    return pockets
        for nested in value.values():
            pockets = _fpocket_candidate_list(nested)
            if pockets:
                return pockets
    elif isinstance(value, list):
        pockets = [item for item in value if isinstance(item, dict) and _looks_like_pocket_dict(item)]
        if pockets:
            return pockets
        for nested in value:
            pockets = _fpocket_candidate_list(nested)
            if pockets:
                return pockets
    return []


def _fpocket_extract_count(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("pocket_count", "nb_pockets", "num_pockets", "pockets_number", "n_pockets"):
            num = _coerce_number(value.get(key))
            if isinstance(num, int):
                return int(num)
        pockets = _fpocket_candidate_list(value)
        if pockets:
            return len(pockets)
        for nested in value.values():
            cnt = _fpocket_extract_count(nested)
            if cnt is not None:
                return cnt
    elif isinstance(value, list):
        pockets = [item for item in value if isinstance(item, dict)]
        if pockets:
            return len(pockets)
        for nested in value:
            cnt = _fpocket_extract_count(nested)
            if cnt is not None:
                return cnt
    return None


def _fpocket_extract_chain_list(value: Any) -> list[str]:
    chains = _normalize_string_list(value)
    if chains:
        return list(dict.fromkeys(chains))
    if isinstance(value, dict):
        for key in ("chains", "chain_ids", "involved_chains", "chain_list"):
            chains = _normalize_string_list(value.get(key))
            if chains:
                return list(dict.fromkeys(chains))
        residues = value.get("residues") or value.get("residue_ids") or value.get("pocket_residues")
        if isinstance(residues, list):
            out: list[str] = []
            for item in residues:
                if isinstance(item, dict):
                    chain = _first_nonempty_text(item.get("chain"), item.get("chain_id"), item.get("chain_name"))
                    if chain:
                        out.append(chain)
                elif isinstance(item, str):
                    m = re.match(r"^\s*([A-Za-z0-9])(?:[:_.-]\d+.*|[\s,:].*)?$", item)
                    if m:
                        out.append(m.group(1))
            if out:
                return list(dict.fromkeys(out))
    return []


def _fpocket_extract_triplet(value: Any, names: tuple[str, ...]) -> list[float | int] | None:
    if not isinstance(value, dict):
        return None
    for key in names:
        trip = _coerce_triplet(value.get(key))
        if trip:
            return trip
    prefixes: list[str] = []
    if any("center" in name or name == "centroid" for name in names):
        prefixes.extend(("center", "pocket_center", "centroid"))
    if any(name in {"size", "pocket_size", "box_size", "dimension", "dimensions"} for name in names):
        prefixes.extend(("size", "pocket_size", "box_size"))
    for prefix in prefixes:
        vals = [value.get(f"{prefix}_x"), value.get(f"{prefix}_y"), value.get(f"{prefix}_z")]
        trip = _coerce_triplet(vals)
        if trip:
            return trip
    return None


def _fpocket_extract_score(value: Any) -> float | int | None:
    if not isinstance(value, dict):
        return None
    for key in ("score", "druggability_score", "probability", "rank_score", "pocket_score"):
        num = _coerce_number(value.get(key))
        if num is not None:
            return num
    return None


def _fpocket_pick_top_pocket(value: Any) -> dict[str, Any] | None:
    pockets = _fpocket_candidate_list(value)
    if not pockets:
        return None
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, pocket in enumerate(pockets):
        score = _fpocket_extract_score(pocket)
        if isinstance(score, (int, float)):
            scored.append((float(score), idx, pocket))
    pocket = None
    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        pocket = scored[0][2]
    else:
        pocket = pockets[0]
    if not isinstance(pocket, dict):
        return None
    top: dict[str, Any] = {}
    center = _fpocket_extract_triplet(pocket, ("center", "pocket_center", "centroid", "center_xyz", "pocket_center_xyz"))
    if center:
        top["center"] = center
    size = _fpocket_extract_triplet(pocket, ("size", "pocket_size", "box_size", "dimension", "dimensions"))
    if size and size != center and all(float(v) >= 0 for v in size):
        top["size"] = size
    score = _coerce_number(pocket.get("score"))
    if score is None:
        score = _coerce_number(pocket.get("pocket_score"))
    if score is not None:
        top["score"] = score
    druggability_score = _coerce_number(pocket.get("druggability_score"))
    if druggability_score is not None:
        top["druggability_score"] = druggability_score
    chains = _fpocket_extract_chain_list(pocket)
    if chains:
        top["chains"] = chains
    return top or None


def _compress_fpocket_observation_payload(
    raw_content: Any,
    *,
    sample_id: str,
    actions: list[dict[str, Any]],
    raw_event_index: int,
    tool_use_id: str,
    path_cache: dict[str, str],
) -> dict[str, Any]:
    parsed = _maybe_parse_json(raw_content)
    if isinstance(parsed, str):
        parsed_text = _sanitize_text_with_artifacts(parsed, path_cache)
        msg = _first_nonempty_text(parsed_text, "fpocket output parse failed")
        return {
            "ok": False,
            "tool_name": "fpocket_toolkit",
            "status": "error",
            "content": {
                "status": "error",
                "msg": msg[:240],
            },
        }

    if not isinstance(parsed, dict):
        parsed = {} if parsed is None else {"value": parsed}

    status = _first_nonempty_text(parsed.get("status"), parsed.get("state"), parsed.get("result_status"), "success").lower()
    if status not in {"success", "error", "timeout", "partial_success"}:
        status = "success"

    pocket_count = _fpocket_extract_count(parsed)
    top_pocket = _fpocket_pick_top_pocket(parsed)
    msg = _first_nonempty_text(
        parsed.get("msg"),
        parsed.get("message"),
        parsed.get("summary"),
        parsed.get("status_message"),
    )
    if not msg:
        if status == "success":
            if pocket_count is not None:
                msg = f"fpocket 运行成功，检测到 {pocket_count} 个口袋"
            else:
                msg = "fpocket 运行成功"
        else:
            msg = "fpocket 执行失败"

    artifact = "<artifact:fpocket/result>"
    content: dict[str, Any] = {
        "status": status,
        "msg": msg,
    }
    if status == "success":
        if pocket_count is not None:
            content["pocket_count"] = pocket_count
        if top_pocket:
            content["top_pocket"] = top_pocket
        content["artifact"] = artifact

    content = _sanitize_structure_with_artifacts(content, {})
    ok = status in {"success", "partial_success"}
    if not ok:
        content = {
            "status": "error",
            "msg": _sanitize_text_with_artifacts(msg, path_cache)[:240],
        }
    return {
        "ok": ok,
        "tool_name": "fpocket_toolkit",
        "status": status if ok else "error",
        "content": content,
    }


def _load_sidecar_json_from_original(original_path: str, copied_path: Path, summary_row: dict[str, Any], filename: str) -> dict[str, Any]:
    if not original_path:
        original_path = ""
    p = Path(original_path)
    candidate_dirs: list[Path] = []
    if p.is_dir():
        candidate_dirs.extend([p, p.parent])
    else:
        candidate_dirs.extend([p.parent, p.parent.parent])
    seen: set[str] = set()
    for d in candidate_dirs:
        if not d or not d.exists():
            continue
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        cand = d / filename
        if cand.is_file():
            return _load_json(cand)

    run_dir, row_dir, rollout_dir = _infer_run_row_rollout_dir(copied_path, summary_row)
    search_roots = [REPO_ROOT / "results", REPO_ROOT / "results" / "results"]
    for base in search_roots:
        if not base.exists():
            continue
        direct_candidates = [
            base / run_dir / row_dir / rollout_dir / filename,
            base / "cached" / run_dir / row_dir / rollout_dir / filename,
            base / run_dir / row_dir / filename,
            base / "cached" / run_dir / row_dir / filename,
        ]
        for cand in direct_candidates:
            if cand.is_file():
                return _load_json(cand)

    if run_dir and row_dir:
        for base in search_roots:
            if not base.exists():
                continue
            for qpath in base.rglob(filename):
                parts = set(qpath.parts)
                if run_dir in parts and row_dir in parts and (not rollout_dir or rollout_dir in parts):
                    return _load_json(qpath)
    return {}


def _extract_task_answer_values_from_text(task: str, text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return _normalize_string_list(parsed)
    if isinstance(parsed, dict):
        if task == "ac":
            for key in ("answer_smiles", "selected_molecule", "selected_smiles", "answer", "output"):
                vals = _normalize_string_list(parsed.get(key))
                if vals:
                    return vals
        elif task == "vs":
            for key in ("ranked_smiles", "selected_smiles", "ranking", "ranked", "ordered", "predicted_ranking", "top3", "prediction", "output", "answer"):
                vals = _normalize_string_list(parsed.get(key))
                if vals:
                    return vals
        elif task == "pf":
            for key in ("selected_smiles", "prediction", "output", "answer"):
                vals = _normalize_string_list(parsed.get(key))
                if vals:
                    return vals
        for key in ("ranking", "ranked", "ordered", "predicted_ranking", "top3", "prediction", "output", "answer"):
            vals = _normalize_string_list(parsed.get(key))
            if vals:
                return vals
        if task in {"e2e", "kg"}:
            for key in ("answer", "output", "result", "text"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    return [val.strip()]
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return []
    if task == "ac":
        return [lines[0]]
    return lines


def _load_task_answer_values(
    *,
    task: str,
    original_path: str,
    copied_path: Path,
    summary_row: dict[str, Any],
    cleaned_text: str,
) -> tuple[list[str], str]:
    sidecar = _load_sidecar_json_from_original(original_path, copied_path, summary_row, "parsed_answer.json")
    source = "missing"
    values: list[str] = []

    for key in ("answer", "ranking", "predicted_ranking", "prediction", "top3", "output"):
        values = _normalize_string_list(sidecar.get(key))
        if values:
            source = f"parsed_answer.{key}"
            break

    if not values:
        answer_block = str(sidecar.get("answer_block") or "").strip()
        if answer_block:
            values = _extract_task_answer_values_from_text(task, answer_block)
            if values:
                source = "parsed_answer.answer_block"

    if not values:
        values = _extract_task_answer_values_from_text(task, cleaned_text)
        if values:
            source = "assistant_text"

    return values, source


_QUESTION_TEXT_CACHE: dict[tuple[str, str], str] = {}


def _load_question_text_from_original(original_session_path: str, *, copied_path: Path, summary_row: dict[str, Any]) -> str:
    if not original_session_path:
        original_session_path = ""
    p = Path(original_session_path)
    if p.is_file():
        sample_dir = p.parent
        candidates = [
            sample_dir / "question.json",
            sample_dir.parent / "question.json",
        ]
        for qpath in candidates:
            q = _load_json(qpath)
            if not q:
                continue
            for key in ("question_text", "question", "public_question_text"):
                qtxt = str(q.get(key) or "").strip()
                if qtxt:
                    return qtxt
        for pth in [sample_dir / "prompt.txt", sample_dir.parent / "prompt.txt"]:
            if pth.is_file():
                txt = pth.read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    return txt

    run_dir, row_dir = _infer_run_and_row_dir(copied_path, summary_row)
    cache_key = (run_dir, row_dir)
    if cache_key in _QUESTION_TEXT_CACHE:
        return _QUESTION_TEXT_CACHE[cache_key]

    search_roots = [REPO_ROOT / "results", REPO_ROOT / "results" / "results"]
    for base in search_roots:
        if not base.exists():
            continue
        direct_candidates = [
            base / run_dir / row_dir / "question.json",
            base / run_dir / row_dir / "prompt.txt",
            base / "cached" / run_dir / row_dir / "question.json",
            base / "cached" / run_dir / row_dir / "prompt.txt",
        ]
        for cand in direct_candidates:
            if cand.is_file():
                if cand.suffix == ".json":
                    q = _load_json(cand)
                    for key in ("question_text", "question", "public_question_text"):
                        qtxt = str(q.get(key) or "").strip()
                        if qtxt:
                            _QUESTION_TEXT_CACHE[cache_key] = qtxt
                            return qtxt
                else:
                    txt = cand.read_text(encoding="utf-8", errors="ignore").strip()
                    if txt:
                        _QUESTION_TEXT_CACHE[cache_key] = txt
                        return txt
        if run_dir and row_dir:
            for qpath in base.rglob("question.json"):
                parts = set(qpath.parts)
                if run_dir in parts and row_dir in parts:
                    q = _load_json(qpath)
                    for key in ("question_text", "question", "public_question_text"):
                        qtxt = str(q.get(key) or "").strip()
                        if qtxt:
                            _QUESTION_TEXT_CACHE[cache_key] = qtxt
                            return qtxt
            for pth in base.rglob("prompt.txt"):
                parts = set(pth.parts)
                if run_dir in parts and row_dir in parts:
                    txt = pth.read_text(encoding="utf-8", errors="ignore").strip()
                    if txt:
                        _QUESTION_TEXT_CACHE[cache_key] = txt
                        return txt
    return ""


def _build_observation_object(
    *,
    sample_id: str,
    tool_name: str,
    raw_tool_name: str,
    tool_use_id: str,
    raw_content: Any,
    raw_is_error: bool,
    raw_event_index: int,
    actions: list[dict[str, Any]],
    max_obs_chars: int,
    path_cache: dict[str, str],
) -> tuple[dict[str, Any], bool, bool]:
    raw_status = "error" if raw_is_error else "success"
    if tool_name == "fpocket_toolkit":
        if isinstance(raw_content, str):
            cleaned_text, fence_stripped = _clean_text_piece(
                raw_content,
                actions=actions,
                sample_id=sample_id,
                field_path=f"user[{raw_event_index}].tool_result[{tool_use_id}].content",
                allow_engineering_drop=False,
                path_cache=path_cache,
            )
            parsed_content: Any = _maybe_parse_json(cleaned_text or "")
        else:
            fence_stripped = False
            parsed_content = raw_content
        compressed = _compress_fpocket_observation_payload(
            parsed_content,
            sample_id=sample_id,
            actions=actions,
            raw_event_index=raw_event_index,
            tool_use_id=tool_use_id,
            path_cache=path_cache,
        )
        compressed = _sanitize_structure_with_artifacts(compressed, path_cache)
        return compressed, bool(fence_stripped), False

    normalized_content = raw_content
    fence_stripped = False
    if isinstance(normalized_content, str):
        cleaned_text, fence_stripped = _clean_text_piece(
            normalized_content,
            actions=actions,
            sample_id=sample_id,
            field_path=f"user[{raw_event_index}].tool_result[{tool_use_id}].content",
            allow_engineering_drop=False,
            path_cache=path_cache,
        )
        if cleaned_text is None:
            cleaned_text = ""
        normalized_content = cleaned_text
        try:
            parsed = json.loads(cleaned_text)
            normalized_content = parsed
        except Exception:
            normalized_content = cleaned_text
    elif isinstance(normalized_content, (dict, list, int, float, bool)) or normalized_content is None:
        pass
    else:
        normalized_content = str(normalized_content)

    pointers = _extract_pointer_map(raw_content)
    if not pointers:
        pointers = _extract_pointer_map(normalized_content)
    pointers = _sanitize_structure_with_artifacts(pointers, path_cache) if pointers else {}

    status = raw_status
    content_status = None
    if isinstance(normalized_content, dict):
        maybe_status = normalized_content.get("status")
        if isinstance(maybe_status, str) and maybe_status.strip():
            content_status = maybe_status.strip().lower()
            if content_status in {"success", "error", "timeout", "partial_success"}:
                status = content_status
    elif isinstance(normalized_content, str):
        low = normalized_content.lower()
        if "timeout" in low:
            status = "timeout"
        elif "error" in low or raw_is_error:
            status = "error"

    ok = status in {"success", "partial_success"} and not raw_is_error
    normalized_content = _sanitize_structure_with_artifacts(normalized_content, path_cache)
    content_to_store, truncated, before_preview, after_preview = _truncate_json_compatible(normalized_content, max_obs_chars)
    if truncated:
        _record_action(
            actions,
            sample_id=sample_id,
            field_path=f"user[{raw_event_index}].tool_result[{tool_use_id}].content",
            cleaning_type="truncate_observation",
            before_preview=before_preview,
            after_preview=after_preview,
            original_backed_up=True,
            extra={"tool_name": tool_name, "raw_tool_name": raw_tool_name},
        )

    metadata: dict[str, Any] = {
        "tool_use_id": tool_use_id,
        "raw_tool_name": raw_tool_name,
        "raw_status": content_status or raw_status,
        "raw_is_error": bool(raw_is_error),
        "raw_event_index": raw_event_index,
    }
    if pointers:
        metadata["pointers"] = pointers
    if fence_stripped:
        metadata["fence_wrapper_stripped"] = True

    obs = {
        "ok": bool(ok),
        "tool_name": tool_name,
        "status": status,
        "content": content_to_store,
        "metadata": metadata,
    }
    obs = _sanitize_structure_with_artifacts(obs, path_cache)
    return obs, fence_stripped, truncated


def _render_thought_tag(texts: list[str]) -> str:
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    if not joined:
        return ""
    return f"<thought>{joined}</thought>"


def _render_tool_call_tag(tool_name: str, arguments: Any) -> str:
    payload = {
        "tool_name": tool_name,
        "arguments": arguments if isinstance(arguments, dict) else {},
    }
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def _render_observation_tag(tool_name: str, obs: dict[str, Any]) -> str:
    return f"<observation tool_name=\"{tool_name}\">{json.dumps(obs, ensure_ascii=False, separators=(',', ':'))}</observation>"


def _render_final_answer_tag(payload: dict[str, Any]) -> str:
    return f"<final_answer>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</final_answer>"


def _extract_assistant_text_items(content: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(content, str):
        s = content.strip()
        if s:
            out.append(("text", s))
        return out
    if not isinstance(content, list):
        return out
    for item in content:
        if not isinstance(item, dict):
            continue
        it = str(item.get("type") or "")
        if it == "thinking":
            txt = item.get("thinking")
            if isinstance(txt, str) and txt.strip():
                out.append(("thinking", txt.strip()))
        elif it == "text":
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                out.append(("text", txt.strip()))
    return out


def _extract_tool_use_items(content: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return out
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") == "tool_use":
            out.append(item)
    return out


def _extract_tool_result_items(content: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return out
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") == "tool_result":
            out.append(item)
    return out


def _build_task_specific_final_answer(
    *,
    task: str,
    canonical_values: list[str],
    cleaned_text: str,
) -> dict[str, Any]:
    values = [v for v in canonical_values if isinstance(v, str) and v.strip()]
    if task == "ac":
        answer_smiles = values[0] if values else ""
        short_reason = f"Selected {answer_smiles} as the predicted molecule." if answer_smiles else "Selected the predicted molecule."
        return {
            "task_type": "ac",
            "answer_smiles": answer_smiles,
            "short_reason": short_reason,
            "evidence": [],
        }
    if task == "vs":
        ranked_smiles = values
        selected_smiles = ranked_smiles[0] if ranked_smiles else ""
        short_reason = (
            f"Ranked the candidate SMILES and selected {selected_smiles} as the top candidate."
            if selected_smiles
            else "Ranked the candidate SMILES and selected the top candidate."
        )
        return {
            "task_type": "vs",
            "ranked_smiles": ranked_smiles,
            "selected_smiles": selected_smiles,
            "short_reason": short_reason,
            "evidence": [],
        }
    if task == "pf":
        selected_smiles = values
        short_reason = f"Extracted {len(selected_smiles)} predicted SMILES from the final response."
        return {
            "task_type": "pf",
            "selected_smiles": selected_smiles,
            "short_reason": short_reason,
            "evidence": [],
        }
    if task in {"kg", "e2e"}:
        answer_text = cleaned_text.strip() or (values[0] if values else "")
        short_reason = _preview_text(answer_text, 180) if answer_text else "Final response extracted from the completed session."
        return {
            "task_type": task,
            "answer": answer_text,
            "steps_summary": short_reason,
            "evidence": [],
        }
    answer_text = cleaned_text.strip() or (values[0] if values else "")
    short_reason = _preview_text(answer_text, 180) if answer_text else "Final response extracted from the completed session."
    return {
        "task_type": task,
        "answer": answer_text,
        "short_reason": short_reason,
        "evidence": [],
    }


def _build_final_answer_payload(
    *,
    sample_id: str,
    task: str,
    text: str,
    source_kind: str,
    actions: list[dict[str, Any]],
    canonical_values: list[str],
    path_cache: dict[str, str],
) -> tuple[str, dict[str, Any], bool]:
    cleaned_text, fence_stripped = _clean_text_piece(
        text,
        actions=actions,
        sample_id=sample_id,
        field_path=f"final_answer[{source_kind}]",
        allow_engineering_drop=False,
        path_cache=path_cache,
    )
    cleaned_text = cleaned_text or ""
    task_result = _build_task_specific_final_answer(task=task, canonical_values=canonical_values, cleaned_text=cleaned_text)
    payload = _sanitize_structure_with_artifacts(task_result, path_cache)
    return _render_final_answer_tag(payload), payload, fence_stripped


def _build_sample_record(
    *,
    task: str,
    copied_path: Path,
    summary_row: dict[str, Any],
    tool_role_mode: str,
    max_observation_chars: int,
    split_multi_tool_calls: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    events = _load_jsonl(copied_path)
    if not events:
        return None, None, {"copied_path": str(copied_path), "reason": "empty_or_invalid_jsonl"}

    original_path = str(summary_row.get("original_path") or copied_path)
    question_text = _load_question_text_from_original(original_path, copied_path=copied_path, summary_row=summary_row)
    if not question_text:
        return None, None, {"copied_path": str(copied_path), "reason": "missing_question_text"}

    sample_id = f"mcp_sft_{task}_{_sha(str(copied_path.resolve()))}"
    run_dir_name = str(summary_row.get("run_dir") or copied_path.name.split("__", 1)[0])
    path_cache: dict[str, str] = {}
    question_text = _sanitize_text_with_artifacts(question_text, path_cache)

    raw_block_hist: Counter[str] = Counter()
    retained_tool_counts: Counter[str] = Counter()
    dropped_tool_counts: Counter[str] = Counter()
    tool_name_by_id: dict[str, str] = {}
    raw_tool_name_by_id: dict[str, str] = {}
    tool_namespace_by_id: dict[str, str] = {}
    cleaning_actions: list[dict[str, Any]] = []
    rejected_reason: str | None = None
    retained_tool_use_count = 0
    dropped_non_mcp_tool_count = 0
    orphan_tool_results = 0
    fence_wrappers_stripped = 0
    truncated_observations = 0
    final_answer_source = "assistant_text"

    parsed_events: list[dict[str, Any]] = []
    last_retained_tool_event_idx = -1

    for idx, ev in enumerate(events):
        et = str(ev.get("type") or "")
        if et == "assistant":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            text_items = _extract_assistant_text_items(content)
            tool_items = _extract_tool_use_items(content)

            pieces: list[dict[str, Any]] = []
            kept_tool_count_in_event = 0
            for item_idx, (kind, txt) in enumerate(text_items):
                raw_block_hist[f"assistant.{kind}"] += 1
                cleaned, changed = _clean_text_piece(
                    txt,
                    actions=cleaning_actions,
                    sample_id=sample_id,
                    field_path=f"assistant[{idx}].{kind}[{item_idx}]",
                    allow_engineering_drop=True,
                    path_cache=path_cache,
                )
                if cleaned is None:
                    continue
                if changed:
                    fence_wrappers_stripped += 1
                pieces.append({"kind": "thought_text", "text": cleaned})

            for item_idx, tu in enumerate(tool_items):
                raw_block_hist["assistant.tool_use"] += 1
                raw_name = str(tu.get("name") or "").strip()
                tool_use_id = str(tu.get("id") or "").strip()
                norm_name, tool_namespace = _normalize_tool_name(raw_name)
                if not tool_namespace or not norm_name:
                    dropped_tool_counts[raw_name or "<empty>"] += 1
                    dropped_non_mcp_tool_count += 1
                    _record_action(
                        cleaning_actions,
                        sample_id=sample_id,
                        field_path=f"assistant[{idx}].tool_use[{item_idx}]",
                        cleaning_type="remove_non_mcp_tool",
                        before_preview=raw_name or "<empty>",
                        after_preview="",
                        original_backed_up=True,
                        extra={"tool_use_id": tool_use_id},
                    )
                    continue
                retained_tool_counts[raw_name] += 1
                retained_tool_use_count += 1
                if tool_use_id:
                    tool_name_by_id[tool_use_id] = norm_name
                    raw_tool_name_by_id[tool_use_id] = raw_name
                    tool_namespace_by_id[tool_use_id] = tool_namespace
                args = tu.get("input") if isinstance(tu.get("input"), dict) else {}
                if not isinstance(tu.get("input"), dict):
                    _record_action(
                        cleaning_actions,
                        sample_id=sample_id,
                        field_path=f"assistant[{idx}].tool_use[{item_idx}].input",
                        cleaning_type="coerce_tool_arguments",
                        before_preview=str(tu.get("input")),
                        after_preview="{}",
                        original_backed_up=True,
                        extra={"tool_name": norm_name, "raw_tool_name": raw_name},
                    )
                args = _sanitize_structure_with_artifacts(args, path_cache)
                pieces.append({
                    "kind": "tool_call",
                    "tool_name": norm_name,
                    "raw_tool_name": raw_name,
                    "tool_namespace": tool_namespace,
                    "tool_use_id": tool_use_id,
                    "arguments": args,
                })
                kept_tool_count_in_event += 1
                last_retained_tool_event_idx = max(last_retained_tool_event_idx, idx)

            if pieces:
                parsed_events.append({
                    "idx": idx,
                    "type": "assistant",
                    "pieces": pieces,
                    "tool_count": kept_tool_count_in_event,
                    "raw_has_tool_use": bool(tool_items),
                })
        elif et == "user":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            result_items = _extract_tool_result_items(content)
            raw_block_hist["user.tool_result"] += len(result_items)
            obs_items: list[dict[str, Any]] = []
            for item_idx, tr in enumerate(result_items):
                tool_use_id = str(tr.get("tool_use_id") or "").strip()
                raw_tool_name = raw_tool_name_by_id.get(tool_use_id, "")
                tool_name = tool_name_by_id.get(tool_use_id, "")
                tool_namespace = tool_namespace_by_id.get(tool_use_id, "")
                if not tool_name:
                    orphan_tool_results += 1
                    _record_action(
                        cleaning_actions,
                        sample_id=sample_id,
                        field_path=f"user[{idx}].tool_result[{item_idx}]",
                        cleaning_type="drop_orphan_tool_result",
                        before_preview=str(tr.get("content") or ""),
                        after_preview="",
                        original_backed_up=True,
                        extra={"tool_use_id": tool_use_id},
                    )
                    continue
                obs_obj, fence_stripped, truncated = _build_observation_object(
                    sample_id=sample_id,
                    tool_name=tool_name,
                    raw_tool_name=raw_tool_name,
                    tool_use_id=tool_use_id,
                    raw_content=tr.get("content"),
                    raw_is_error=bool(tr.get("is_error")),
                    raw_event_index=idx,
                    actions=cleaning_actions,
                    max_obs_chars=max_observation_chars,
                    path_cache=path_cache,
                )
                if fence_stripped:
                    fence_wrappers_stripped += 1
                if truncated:
                    truncated_observations += 1
                obs_items.append({
                    "kind": "observation",
                    "tool_name": tool_name,
                    "raw_tool_name": raw_tool_name,
                    "tool_namespace": tool_namespace,
                    "tool_use_id": tool_use_id,
                    "obs": obs_obj,
                })
                last_retained_tool_event_idx = max(last_retained_tool_event_idx, idx)
            if obs_items:
                parsed_events.append({"idx": idx, "type": "user", "pieces": obs_items})
        elif et == "result":
            raw_block_hist["result"] += 1
            result_text = str(ev.get("result") or "").strip()
            if result_text:
                parsed_events.append({"idx": idx, "type": "result", "result_text": result_text})
        else:
            raw_block_hist[f"other.{et or 'unknown'}"] += 1

    if retained_tool_use_count <= 0:
        rejected_reason = "no_retained_mcp_tool_calls"
    if last_retained_tool_event_idx < 0:
        rejected_reason = rejected_reason or "missing_molclaw_usage"

    if rejected_reason:
        return None, None, {
            "copied_path": str(copied_path),
            "reason": rejected_reason,
            "task": task,
            "source_raw_path": original_path,
        }

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT, "step_loss_mask": 0},
        {"role": "user", "content": question_text, "step_loss_mask": 0},
    ]

    current_final_texts: list[str] = []
    result_fallback_text = ""
    retained_tool_call_count = 0
    retained_observation_count = 0
    assistant_turn_count = 0
    user_observation_turn_count = 0
    final_answer_turn_count = 0

    def flush_assistant_block(text_parts: list[str], tool_calls: list[dict[str, Any]]) -> None:
        nonlocal assistant_turn_count, retained_tool_call_count
        if not text_parts and not tool_calls:
            return
        parts: list[str] = []
        thought_text = _render_thought_tag(text_parts)
        if thought_text:
            parts.append(thought_text)
        if not split_multi_tool_calls or len(tool_calls) <= 1:
            for call in tool_calls:
                parts.append(_render_tool_call_tag(call["tool_name"], call["arguments"]))
                retained_tool_call_count += 1
        else:
            first = True
            for call in tool_calls:
                if first and thought_text:
                    parts.append(_render_tool_call_tag(call["tool_name"], call["arguments"]))
                else:
                    parts.append(_render_tool_call_tag(call["tool_name"], call["arguments"]))
                retained_tool_call_count += 1
                first = False
        content = "\n".join(parts).strip()
        if content:
            messages.append({"role": "assistant", "content": content, "step_loss_mask": 1})
            assistant_turn_count += 1

    def flush_observation_block(obs_parts: list[dict[str, Any]]) -> None:
        nonlocal user_observation_turn_count, retained_observation_count
        if not obs_parts:
            return
        def _obs_message(tool_name: str, content: str) -> dict[str, Any]:
            if tool_role_mode == "tool":
                return {"role": "tool", "name": tool_name, "content": content, "step_loss_mask": 0}
            return {"role": "user", "content": content, "step_loss_mask": 0}

        if split_multi_tool_calls and len(obs_parts) > 1:
            for obs in obs_parts:
                content = _render_observation_tag(obs["tool_name"], obs["obs"])
                messages.append(_obs_message(obs["tool_name"], content))
                retained_observation_count += 1
                user_observation_turn_count += 1
        else:
            content = "\n".join(_render_observation_tag(obs["tool_name"], obs["obs"]) for obs in obs_parts)
            first_name = obs_parts[0]["tool_name"] if obs_parts else "unknown"
            messages.append(_obs_message(first_name, content))
            retained_observation_count += len(obs_parts)
            user_observation_turn_count += 1

    assistant_buffer: list[dict[str, Any]] = []
    current_obs_buffer: list[dict[str, Any]] = []

    def flush_assistant_buffer() -> None:
        nonlocal assistant_buffer, assistant_turn_count, retained_tool_call_count
        if not assistant_buffer:
            return

        rendered_parts: list[str] = []
        thought_parts: list[str] = []

        def flush_thought_parts() -> None:
            if not thought_parts:
                return
            thought_text = _render_thought_tag(thought_parts)
            thought_parts.clear()
            if thought_text:
                rendered_parts.append(thought_text)

        for piece in assistant_buffer:
            kind = str(piece.get("kind") or "")
            if kind == "thought_text":
                text = str(piece.get("text") or "").strip()
                if text:
                    thought_parts.append(text)
                continue
            if kind == "tool_call":
                flush_thought_parts()
                rendered_parts.append(_render_tool_call_tag(str(piece.get("tool_name") or ""), piece.get("arguments") if isinstance(piece.get("arguments"), dict) else {}))
                retained_tool_call_count += 1
                continue

        flush_thought_parts()

        content = "\n".join(part for part in rendered_parts if part).strip()
        assistant_buffer = []
        if content:
            messages.append({"role": "assistant", "content": content, "step_loss_mask": 1})
            assistant_turn_count += 1

    for ev in parsed_events:
        idx = int(ev["idx"])
        if idx <= last_retained_tool_event_idx:
            if ev["type"] == "assistant":
                if current_obs_buffer:
                    flush_observation_block(current_obs_buffer)
                    current_obs_buffer = []
                pieces = list(ev["pieces"])
                if not pieces:
                    continue
                if split_multi_tool_calls:
                    flush_assistant_buffer()
                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []
                    for piece in pieces:
                        if piece["kind"] == "thought_text":
                            text_parts.append(str(piece.get("text") or ""))
                        elif piece["kind"] == "tool_call":
                            tool_calls.append(piece)
                    if len(tool_calls) > 1:
                        first_text = list(text_parts)
                        for call_index, call in enumerate(tool_calls):
                            parts: list[str] = []
                            if call_index == 0:
                                thought = _render_thought_tag(first_text)
                                if thought:
                                    parts.append(thought)
                            parts.append(_render_tool_call_tag(str(call.get("tool_name") or ""), call.get("arguments") if isinstance(call.get("arguments"), dict) else {}))
                            content = "\n".join(parts).strip()
                            if content:
                                messages.append({"role": "assistant", "content": content, "step_loss_mask": 1})
                                assistant_turn_count += 1
                                retained_tool_call_count += 1
                        continue
                    content_parts: list[str] = []
                    thought_parts: list[str] = []
                    for piece in pieces:
                        if piece["kind"] == "thought_text":
                            text = str(piece.get("text") or "").strip()
                            if text:
                                thought_parts.append(text)
                        elif piece["kind"] == "tool_call":
                            if thought_parts:
                                thought = _render_thought_tag(thought_parts)
                                thought_parts = []
                                if thought:
                                    content_parts.append(thought)
                            content_parts.append(_render_tool_call_tag(str(piece.get("tool_name") or ""), piece.get("arguments") if isinstance(piece.get("arguments"), dict) else {}))
                    if thought_parts:
                        thought = _render_thought_tag(thought_parts)
                        if thought:
                            content_parts.append(thought)
                    content = "\n".join(part for part in content_parts if part).strip()
                    if content:
                        messages.append({"role": "assistant", "content": content, "step_loss_mask": 1})
                        assistant_turn_count += 1
                    continue
                assistant_buffer.extend(pieces)
            elif ev["type"] == "user":
                obs_parts = list(ev["pieces"])
                if obs_parts:
                    flush_assistant_buffer()
                    flush_observation_block(obs_parts)
                    current_obs_buffer = []
            elif ev["type"] == "result":
                flush_assistant_buffer()
                result_fallback_text = ev["result_text"]
        else:
            if ev["type"] == "assistant":
                for piece in ev["pieces"]:
                    if piece["kind"] == "thought_text":
                        current_final_texts.append(piece["text"])
            elif ev["type"] == "result":
                result_fallback_text = ev["result_text"]

    flush_assistant_buffer()

    if current_obs_buffer:
        flush_observation_block(current_obs_buffer)

    final_source_text = "\n\n".join(t for t in current_final_texts if t.strip()).strip()
    if not final_source_text:
        final_source_text = result_fallback_text.strip()
        final_answer_source = "result_event"
    else:
        final_answer_source = "assistant_text"

    if not final_source_text:
        return None, None, {
            "copied_path": str(copied_path),
            "reason": "missing_final_answer",
            "task": task,
            "source_raw_path": original_path,
        }

    canonical_values, canonical_source = _load_task_answer_values(
        task=task,
        original_path=original_path,
        copied_path=copied_path,
        summary_row=summary_row,
        cleaned_text=final_source_text,
    )
    if task in {"ac", "vs", "pf"} and not canonical_values:
        return None, None, {
            "copied_path": str(copied_path),
            "reason": "missing_task_specific_final_answer",
            "task": task,
            "source_raw_path": original_path,
        }

    final_tag, final_payload, final_fence_stripped = _build_final_answer_payload(
        sample_id=sample_id,
        task=task,
        text=final_source_text,
        source_kind=final_answer_source,
        actions=cleaning_actions,
        canonical_values=canonical_values,
        path_cache=path_cache,
    )
    if final_fence_stripped:
        fence_wrappers_stripped += 1
    final_answer_turn_count += 1
    messages.append({"role": "assistant", "content": final_tag, "step_loss_mask": 1})
    assistant_turn_count += 1

    if len(messages) < 4:
        return None, None, {
            "copied_path": str(copied_path),
            "reason": "insufficient_messages",
            "task": task,
            "source_raw_path": original_path,
        }

    raw_tool_name_map = []
    for raw_name, count in sorted(retained_tool_counts.items()):
        tool_name, tool_namespace = _normalize_tool_name(raw_name)
        raw_tool_name_map.append(
            {
                "raw_tool_name": raw_name,
                "tool_namespace": tool_namespace or "",
                "tool_name": tool_name or "",
                "count": count,
                "kept": True,
            }
        )
    for raw_name, count in sorted(dropped_tool_counts.items()):
        raw_tool_name_map.append(
            {
                "raw_tool_name": raw_name,
                "tool_namespace": "",
                "tool_name": "",
                "count": count,
                "kept": False,
            }
        )

    sample_report = {
        "sample_id": sample_id,
        "task": task,
        "task_type": task,
        "source_raw_path": original_path,
        "copied_path": str(copied_path),
        "source_run": run_dir_name,
        "question_text_preview": _preview_text(question_text),
        "final_answer_source": final_answer_source,
        "final_answer_canonical_source": canonical_source,
        "counts": {
            "retained_mcp_tool_calls": retained_tool_use_count,
            "dropped_non_mcp_tool_calls": dropped_non_mcp_tool_count,
            "orphan_tool_results": orphan_tool_results,
            "fence_wrappers_stripped": fence_wrappers_stripped,
            "truncated_observations": truncated_observations,
            "assistant_turns": assistant_turn_count,
            "user_observation_turns": user_observation_turn_count,
            "final_answer_turns": final_answer_turn_count,
        },
        "raw_block_type_hist": dict(raw_block_hist),
        "raw_tool_name_map": raw_tool_name_map,
        "cleaning_actions": cleaning_actions,
        "validation": {
            "final_answer_preview": _preview_text(json.dumps(final_payload, ensure_ascii=False), 240),
            "tool_role_mode": tool_role_mode,
            "split_multi_tool_calls": bool(split_multi_tool_calls),
            "final_answer_schema": task,
        },
    }

    cleaning_report = {
        **sample_report,
        "cleaning_actions_count": len(cleaning_actions),
    }

    sft_rec = {
        "schema_version": SFT_SCHEMA_VERSION,
        "id": sample_id,
        "messages": messages,
    }

    return sft_rec, cleaning_report, None


def _build_rl_prompt_from_sft(
    rec: dict[str, Any],
    idx: int,
    *,
    task: str,
    summary_row: dict[str, Any],
    cleaning_report: dict[str, Any],
) -> dict[str, Any]:
    msgs = rec.get("messages") if isinstance(rec.get("messages"), list) else []
    prompt: list[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role in {"system", "user"}:
            prompt.append({"role": role, "content": str(m.get("content") or "")})
        if len(prompt) >= 2:
            break

    data_source = f"mol_pipeline_{task}"
    allowed_tools = [
        str(item.get("tool_name") or "").strip()
        for item in (cleaning_report.get("raw_tool_name_map") or [])
        if isinstance(item, dict) and item.get("kept") and isinstance(item.get("tool_name"), str) and str(item.get("tool_name") or "").strip()
    ]

    return {
        "id": rec.get("id"),
        "data_source": data_source,
        "prompt": prompt,
        "ability": "mol_pipeline_tool_use",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": idx,
            "task_type": task,
            "source_run": summary_row.get("run_dir"),
            "trajectory_id": rec.get("id"),
            "used_molclaw": True,
            "answer_hit_pass": summary_row.get("answer_hit_pass"),
            "tool_call_count": cleaning_report.get("counts", {}).get("retained_mcp_tool_calls"),
            "final_answer_source": cleaning_report.get("final_answer_source"),
        },
        "env_kwargs": {
            "task": {
                "task_id": rec.get("id"),
                "task_type": task,
                "instruction": prompt[1]["content"] if len(prompt) > 1 else "",
                "inputs": {},
                "allowed_tools": allowed_tools,
                "max_steps": 8,
                "data_source": data_source,
            }
        },
        "metadata": {
            "source_project": "data-pipe",
            "schema_version": RL_SCHEMA_VERSION,
        },
    }


def _validate_react_sft_record(rec: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    summary = {
        "assistant_messages": 0,
        "user_messages": 0,
        "system_messages": 0,
        "tool_calls": 0,
        "final_answers": 0,
        "observations": 0,
        "assistant_json_parse_failed": 0,
        "observation_json_parse_failed": 0,
        "fence_wrappers_stripped": 0,
    }
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False, summary, ["missing_messages"]

    rid = str(rec.get("id") or "")
    task = _infer_task_from_sample_id(rid)
    local_path_leak_re = ABS_PATH_RE

    first_role = str(msgs[0].get("role") or "") if isinstance(msgs[0], dict) else ""
    second_role = str(msgs[1].get("role") or "") if len(msgs) > 1 and isinstance(msgs[1], dict) else ""
    if first_role != "system":
        errors.append("first_message_not_system")
    if second_role != "user":
        errors.append("second_message_not_user")

    seen_assistant = False
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            errors.append(f"message_{i}_not_object")
            continue
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if role not in {"system", "user", "assistant"}:
            errors.append(f"message_{i}_invalid_role:{role}")
            continue
        if local_path_leak_re.search(content):
            errors.append(f"message_{i}_contains_local_absolute_path")
        if role == "system":
            summary["system_messages"] += 1
            if i != 0:
                errors.append("system_not_first")
            continue
        if role == "user":
            summary["user_messages"] += 1
            if "<observation" not in content and i != 1:
                errors.append(f"message_{i}_user_not_observation")
            obs_matches = list(OBS_TAG_RE.finditer(content))
            if not obs_matches and i != 1:
                errors.append(f"message_{i}_missing_observation_tag")
            for obs_match in obs_matches:
                summary["observations"] += 1
                inner = obs_match.group(2).strip()
                try:
                    obs_obj = json.loads(inner)
                except Exception:
                    summary["observation_json_parse_failed"] += 1
                    errors.append(f"message_{i}_observation_json_parse_failed")
                    continue
                if not isinstance(obs_obj, dict):
                    errors.append(f"message_{i}_observation_not_object")
                    continue
                if not isinstance(obs_obj.get("tool_name"), str) or not obs_obj.get("tool_name"):
                    errors.append(f"message_{i}_observation_tool_name_invalid")
            continue

        seen_assistant = True
        summary["assistant_messages"] += 1
        if "<tool_call>" not in content and "<final_answer>" not in content and "<thought>" not in content:
            errors.append(f"message_{i}_assistant_missing_react_tags")
        if re.sub(r"<thought>[\s\S]*?</thought>|<tool_call>[\s\S]*?</tool_call>|<final_answer>[\s\S]*?</final_answer>", "", content).strip():
            errors.append(f"message_{i}_assistant_has_unwrapped_text")

        for thought_match in THOUGHT_TAG_RE.finditer(content):
            if not thought_match.group(1).strip():
                errors.append(f"message_{i}_empty_thought")

        for call_match in TOOL_CALL_TAG_RE.finditer(content):
            summary["tool_calls"] += 1
            inner = call_match.group(1).strip()
            try:
                call_obj = json.loads(inner)
            except Exception:
                summary["assistant_json_parse_failed"] += 1
                errors.append(f"message_{i}_tool_call_json_parse_failed")
                continue
            if not isinstance(call_obj, dict):
                errors.append(f"message_{i}_tool_call_not_object")
                continue
            tool_name = call_obj.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                errors.append(f"message_{i}_tool_call_tool_name_invalid")
            args = call_obj.get("arguments")
            if not isinstance(args, dict):
                errors.append(f"message_{i}_tool_call_arguments_invalid")
            if isinstance(tool_name, str) and any(tool_name.startswith(prefix) for prefix in TARGET_MCP_PREFIXES):
                errors.append(f"message_{i}_tool_call_not_normalized:{tool_name}")

        for final_match in FINAL_TAG_RE.finditer(content):
            summary["final_answers"] += 1
            inner = final_match.group(1).strip()
            try:
                final_obj = json.loads(inner)
            except Exception:
                summary["assistant_json_parse_failed"] += 1
                errors.append(f"message_{i}_final_answer_json_parse_failed")
                continue
            if not isinstance(final_obj, dict):
                errors.append(f"message_{i}_final_answer_not_object")
                continue
            if "type" in final_obj or isinstance(final_obj.get("answer"), dict):
                errors.append(f"message_{i}_final_answer_wrapped_schema_present")
            payload_task = str(final_obj.get("task_type") or "").strip() or task
            if payload_task != task and task != "unknown":
                errors.append(f"message_{i}_final_answer_task_mismatch:{payload_task}")
            if task == "ac":
                answer_smiles = str(final_obj.get("answer_smiles") or "").strip()
                selected_molecule = str(final_obj.get("selected_molecule") or "").strip()
                short_reason = str(final_obj.get("short_reason") or "").strip()
                evidence = final_obj.get("evidence")
                if not answer_smiles:
                    errors.append(f"message_{i}_final_answer_ac_answer_smiles_invalid")
                if not short_reason:
                    errors.append(f"message_{i}_final_answer_ac_short_reason_invalid")
                if not isinstance(evidence, list):
                    errors.append(f"message_{i}_final_answer_ac_evidence_invalid")
                if selected_molecule and selected_molecule != answer_smiles:
                    errors.append(f"message_{i}_final_answer_ac_selected_molecule_mismatch")
            elif task == "vs":
                ranked = final_obj.get("ranked_smiles")
                selected = final_obj.get("selected_smiles")
                ranked_ok = isinstance(ranked, list) and any(isinstance(v, str) and v.strip() for v in ranked)
                selected_ok = isinstance(selected, str) and bool(selected.strip())
                if not (ranked_ok or selected_ok):
                    errors.append(f"message_{i}_final_answer_vs_ranking_invalid")
                if not isinstance(final_obj.get("short_reason"), str) or not str(final_obj.get("short_reason") or "").strip():
                    errors.append(f"message_{i}_final_answer_vs_short_reason_invalid")
                if not isinstance(final_obj.get("evidence"), list):
                    errors.append(f"message_{i}_final_answer_vs_evidence_invalid")
            elif task == "pf":
                selected_smiles = final_obj.get("selected_smiles")
                if not isinstance(selected_smiles, list) or not any(isinstance(v, str) and v.strip() for v in selected_smiles):
                    errors.append(f"message_{i}_final_answer_pf_selected_smiles_invalid")
                labels = final_obj.get("labels")
                if labels is not None and not isinstance(labels, list):
                    errors.append(f"message_{i}_final_answer_pf_labels_invalid")
                if not isinstance(final_obj.get("short_reason"), str) or not str(final_obj.get("short_reason") or "").strip():
                    errors.append(f"message_{i}_final_answer_pf_short_reason_invalid")
                if not isinstance(final_obj.get("evidence"), list):
                    errors.append(f"message_{i}_final_answer_pf_evidence_invalid")
            elif task in {"kg", "e2e"}:
                if "evidence" in final_obj and not isinstance(final_obj.get("evidence"), list):
                    errors.append(f"message_{i}_final_answer_task_evidence_invalid")
                if task == "e2e" and "steps_summary" in final_obj and not isinstance(final_obj.get("steps_summary"), str):
                    errors.append(f"message_{i}_final_answer_steps_summary_invalid")

    if not seen_assistant:
        errors.append("missing_assistant_turn")
    ok = not errors
    return ok, summary, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert accepted molclaw usage sessions to unified ReAct-style SFT/RL JSONL (all-in-one).")
    ap.add_argument("--input-root", required=True, help="Directory from scan_molclaw_usage.py")
    ap.add_argument("--output-dir", default="", help="Default: <input-root>/sft_outputs")
    ap.add_argument("--summary-csv", default="", help="Default: <input-root>/molclaw_usage_summary.csv")
    ap.add_argument("--answer-hit-only", action="store_true", help="Only keep answer-hit samples for vs/ac/pf. kg/e2e are not filtered.")
    ap.add_argument(
        "--tool-role-mode",
        choices=["user_observation", "tool"],
        default="user_observation",
        help="Compatibility mode for observation role. Default keeps role=user.",
    )
    ap.add_argument(
        "--split-multi-tool-calls",
        action="store_true",
        help="Split multiple tool calls from one assistant event into multiple assistant/user turns.",
    )
    ap.add_argument(
        "--max-observation-chars",
        type=int,
        default=DEFAULT_OBSERVATION_MAX_CHARS,
        help="Maximum serialized observation size before truncation.",
    )
    args = ap.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir.strip() else (input_root / "sft_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaning_reports_dir = output_dir / "cleaning_reports"
    cleaning_reports_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = Path(args.summary_csv).expanduser().resolve() if args.summary_csv.strip() else (input_root / "molclaw_usage_summary.csv")
    summary_map = _load_summary_map(summary_csv)

    candidates: list[tuple[str, Path]] = []
    for task in SUPPORTED_TASKS:
        task_dir = input_root / task
        if not task_dir.is_dir():
            continue
        for p in sorted(task_dir.glob("*.jsonl")):
            candidates.append((task, p.resolve()))

    sft_records: list[dict[str, Any]] = []
    rl_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cleaning_report_index: list[dict[str, Any]] = []
    filtered_answer_hit = 0
    raw_block_hist: Counter[str] = Counter()
    aggregate_tool_name_hist: Counter[str] = Counter()
    aggregate_dropped_tool_hist: Counter[str] = Counter()
    aggregate_validation_hist: Counter[str] = Counter()
    aggregate_cleaning_action_hist: Counter[str] = Counter()
    aggregate_task_hist: Counter[str] = Counter()
    retained_mcp_tool_calls = 0
    dropped_non_mcp_tool_calls = 0
    orphan_tool_results = 0
    fence_wrappers_stripped = 0
    truncated_observations = 0

    for task, copied_path in candidates:
        summary_row = summary_map.get(f"path::{str(copied_path.resolve())}") or summary_map.get(f"name::{copied_path.name}") or {}
        if args.answer_hit_only and task in {"vs", "ac", "pf"}:
            if not _to_bool(summary_row.get("answer_hit_pass")):
                filtered_answer_hit += 1
                continue

        sample_rec, cleaning_report, rej = _build_sample_record(
            task=task,
            copied_path=copied_path,
            summary_row=summary_row,
            tool_role_mode=args.tool_role_mode,
            max_observation_chars=args.max_observation_chars,
            split_multi_tool_calls=bool(args.split_multi_tool_calls),
        )
        if sample_rec is None:
            rejected.append(rej or {"copied_path": str(copied_path), "reason": "unknown"})
            continue

        sample_id = str(sample_rec.get("id") or "")
        cleaning_report_path = cleaning_reports_dir / f"{sample_id}.json"
        cleaning_report["cleaning_report_path"] = str(cleaning_report_path)
        _write_json(cleaning_report_path, cleaning_report)
        cleaning_report_index.append({
            "sample_id": sample_id,
            "task_type": task,
            "cleaning_report_path": str(cleaning_report_path),
            "retained_mcp_tool_calls": cleaning_report["counts"]["retained_mcp_tool_calls"],
            "dropped_non_mcp_tool_calls": cleaning_report["counts"]["dropped_non_mcp_tool_calls"],
            "orphan_tool_results": cleaning_report["counts"]["orphan_tool_results"],
            "fence_wrappers_stripped": cleaning_report["counts"]["fence_wrappers_stripped"],
            "truncated_observations": cleaning_report["counts"]["truncated_observations"],
        })

        ok, val_summary, val_errors = _validate_react_sft_record(sample_rec)
        aggregate_validation_hist.update(val_errors)
        aggregate_task_hist[task] += 1
        if not ok:
            rejected.append({
                "copied_path": str(copied_path),
                "reason": "validation_failed",
                "task": task,
                "source_raw_path": str((summary_row.get("original_path") or "").strip()),
                "errors": val_errors,
            })
            continue

        sft_records.append(sample_rec)
        rl_records.append(
            _build_rl_prompt_from_sft(
                sample_rec,
                len(rl_records),
                task=task,
                summary_row=summary_row,
                cleaning_report=cleaning_report,
            )
        )

        retained_mcp_tool_calls += int(cleaning_report["counts"]["retained_mcp_tool_calls"])
        dropped_non_mcp_tool_calls += int(cleaning_report["counts"]["dropped_non_mcp_tool_calls"])
        orphan_tool_results += int(cleaning_report["counts"]["orphan_tool_results"])
        fence_wrappers_stripped += int(cleaning_report["counts"]["fence_wrappers_stripped"])
        truncated_observations += int(cleaning_report["counts"]["truncated_observations"])
        aggregate_tool_name_hist.update({k: int(v["count"]) for k, v in [(x["raw_tool_name"], x) for x in cleaning_report["raw_tool_name_map"] if x.get("kept")]})
        aggregate_dropped_tool_hist.update({k: int(v["count"]) for k, v in [(x["raw_tool_name"], x) for x in cleaning_report["raw_tool_name_map"] if not x.get("kept")]})
        raw_block_hist.update(cleaning_report.get("raw_block_type_hist") or {})
        for action in cleaning_report.get("cleaning_actions") or []:
            aggregate_cleaning_action_hist[action.get("cleaning_type") or "unknown"] += 1

    for report_row in cleaning_report_index:
        report_row["task_type"] = report_row.get("task_type") or "unknown"
    _write_jsonl(cleaning_reports_dir / "cleaning_report_index.jsonl", cleaning_report_index)

    for task in SUPPORTED_TASKS:
        aggregate_task_hist.setdefault(task, 0)

    sft_dir = output_dir / "mcp_sft_all"
    sft_path = output_dir / "mcp_sft_all.jsonl"
    rl_path = output_dir / "mcp_rl_prompts_all.jsonl"
    rej_path = output_dir / "rejected_samples.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"
    report_md_path = output_dir / "schema_validation_report.md"
    report_json_path = output_dir / "schema_validation_report.json"

    sft_sample_paths = _write_json_dir(sft_dir, sft_records, prefix="mcp_sft_all")
    _write_jsonl(sft_path, sft_records)
    _write_jsonl(rl_path, rl_records)
    _write_jsonl(rej_path, rejected)

    validation_summary = {
        "ok": len(aggregate_validation_hist) == 0,
        "total_sessions": len(candidates),
        "total_sft_samples": len(sft_records),
        "retained_mcp_tool_calls": retained_mcp_tool_calls,
        "dropped_non_mcp_tool_calls": dropped_non_mcp_tool_calls,
        "orphan_tool_results": orphan_tool_results,
        "fence_wrappers_stripped": fence_wrappers_stripped,
        "fence_inner_content_preserved": True,
        "react_json_parse_failed": int(sum(v for k, v in aggregate_validation_hist.items() if "json_parse_failed" in k)),
        "chat_template_failed": int(sum(v for k, v in aggregate_validation_hist.items() if k.startswith("message_") or k.startswith("missing_") or k.endswith("invalid"))),
        "raw_block_type_hist": dict(raw_block_hist),
        "cleaning_action_hist": dict(aggregate_cleaning_action_hist),
        "task_counts": dict(aggregate_task_hist),
        "validation_errors_hist": dict(aggregate_validation_hist),
        "filtered_by_answer_hit": filtered_answer_hit,
        "rejected_samples": len(rejected),
        "truncated_observations": truncated_observations,
    }

    manifest = {
        "schema_version": SFT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "input_root": str(input_root),
        "summary_csv": str(summary_csv),
        "output_dir": str(output_dir),
        "output_files": {
            "sft_all_dir": str(sft_dir),
            "sft_all": str(sft_path),
            "rl_all": str(rl_path),
            "rejected": str(rej_path),
            "cleaning_report_index": str(cleaning_reports_dir / "cleaning_report_index.jsonl"),
            "schema_validation_report": str(report_json_path),
        },
        "counts": {
            "raw_candidates": len(candidates),
            "accepted": len(sft_records),
            "rejected": len(rejected),
            "filtered_by_answer_hit": filtered_answer_hit,
            "retained_mcp_tool_calls": retained_mcp_tool_calls,
            "dropped_non_mcp_tool_calls": dropped_non_mcp_tool_calls,
            "orphan_tool_results": orphan_tool_results,
            "fence_wrappers_stripped": fence_wrappers_stripped,
            "truncated_observations": truncated_observations,
        },
        "tasks": dict(aggregate_task_hist),
        "raw_block_type_hist": dict(raw_block_hist),
        "options": {
            "answer_hit_only": bool(args.answer_hit_only),
            "tool_role_mode": args.tool_role_mode,
            "split_multi_tool_calls": bool(args.split_multi_tool_calls),
            "max_observation_chars": int(args.max_observation_chars),
        },
        "sft_all_dir_samples": len(sft_sample_paths),
    }

    md_lines = [
        "# Postprocess Report",
        "",
        f"- input_root: `{input_root}`",
        f"- summary_csv: `{summary_csv}`",
        f"- raw_candidates: {len(candidates)}",
        f"- accepted: {len(sft_records)}",
        f"- rejected: {len(rejected)}",
        f"- filtered_by_answer_hit: {filtered_answer_hit}",
        f"- retained_mcp_tool_calls: {retained_mcp_tool_calls}",
        f"- dropped_non_mcp_tool_calls: {dropped_non_mcp_tool_calls}",
        f"- orphan_tool_results: {orphan_tool_results}",
        f"- fence_wrappers_stripped: {fence_wrappers_stripped}",
        f"- truncated_observations: {truncated_observations}",
        "",
        "## Task Counts",
        "",
    ]
    for task in SUPPORTED_TASKS:
        md_lines.append(f"- `{task}`: {aggregate_task_hist.get(task, 0)}")
    md_lines.extend([
        "",
        "## Raw Block Types",
        "",
    ])
    for k, v in sorted(raw_block_hist.items()):
        md_lines.append(f"- `{k}`: {v}")
    md_lines.extend([
        "",
        "## Validation Summary",
        "",
        f"- ok: {validation_summary['ok']}",
        f"- total_sessions: {validation_summary['total_sessions']}",
        f"- total_sft_samples: {validation_summary['total_sft_samples']}",
        f"- react_json_parse_failed: {validation_summary['react_json_parse_failed']}",
        f"- chat_template_failed: {validation_summary['chat_template_failed']}",
    ])
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _write_json(report_json_path, validation_summary)
    _write_json(manifest_path, manifest)

    print(json.dumps({
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "sft_all_dir": str(sft_dir),
        "sft_all": str(sft_path),
        "rl_all": str(rl_path),
        "rejected": str(rej_path),
        "manifest": str(manifest_path),
        "validation_report_json": str(report_json_path),
        "validation_report_md": str(report_md_path),
        "raw_candidates": len(candidates),
        "accepted": len(sft_records),
        "rejected_count": len(rejected),
        "filtered_by_answer_hit": filtered_answer_hit,
        "retained_mcp_tool_calls": retained_mcp_tool_calls,
        "dropped_non_mcp_tool_calls": dropped_non_mcp_tool_calls,
        "orphan_tool_results": orphan_tool_results,
        "fence_wrappers_stripped": fence_wrappers_stripped,
        "truncated_observations": truncated_observations,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
