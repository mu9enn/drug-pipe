from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_PREFIX_PATTERN = re.compile(r"^mcp__[^_]+__")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} must be object")
            rows.append(obj)
    return rows


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_tool_name(tool_name: str | None) -> str:
    if not isinstance(tool_name, str):
        return ""
    raw = tool_name.strip()
    if not raw:
        return ""
    return TOOL_PREFIX_PATTERN.sub("", raw)


def is_mcp_tool_name(tool_name: str | None) -> bool:
    if not isinstance(tool_name, str):
        return False
    return tool_name.startswith("mcp__")


def strip_line_numbers(text: str) -> str:
    out = []
    for line in text.splitlines():
        out.append(re.sub(r"^\s*\d+\t", "", line))
    return "\n".join(out)


def parse_numbered_json(text: str) -> dict[str, Any] | None:
    stripped = strip_line_numbers(text).strip()
    data = safe_json_loads(stripped)
    if isinstance(data, dict):
        return data
    return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def bool_from_any(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes"}:
            return True
        if lower in {"false", "0", "no"}:
            return False
    return None
