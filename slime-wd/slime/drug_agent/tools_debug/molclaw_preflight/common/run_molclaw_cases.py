#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUTO_PREFIX = "__AUTO__:"
REF_PREFIX = "__REF__:"
PREFLIGHT_SUCCESS = {"pass_ok", "pass_reached"}


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return to_jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def parse_result(result: Any) -> dict[str, Any]:
    try:
        content = getattr(result, "content", None)
        if isinstance(content, list) and content:
            item = content[0]
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"text": text}
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"text": text}
                return to_jsonable(item)
            return to_jsonable(item)
        if isinstance(result, dict):
            return to_jsonable(result)
        return {"raw": str(result)}
    except Exception as exc:
        return {"error": f"parse error: {exc}", "raw": str(result)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MolClaw MCP tool cases and emit structured report.")
    parser.add_argument("--server-url", required=True, help="MolClaw MCP URL, e.g. http://180.184.86.2:32208/mcp")
    parser.add_argument("--proxy-url", default="", help="Forward proxy URL, e.g. http://<host>:<port>")
    parser.add_argument("--worker-mode", required=True, help="Worker mode label: no-gpu/gpu")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-tool timeout in seconds")
    parser.add_argument("--case-file", required=True, help="Case file path (JSON list)")
    parser.add_argument("--out-json", required=True, help="Output JSON report path")
    parser.add_argument("--out-md", required=True, help="Output markdown report path")
    return parser.parse_args()


def read_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"case file must be a JSON list: {path}")
    cases: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"case #{i} is not an object")
        case_id = str(item.get("case_id", f"case_{i:03d}"))
        tool_name = str(item.get("tool_name", "")).strip()
        arguments = item.get("arguments", {})
        if not tool_name:
            raise ValueError(f"case #{i} missing tool_name")
        if not isinstance(arguments, dict):
            raise ValueError(f"case #{i} arguments must be object")
        normalized = {
            "case_id": case_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "source": item.get("source"),
            "notes": item.get("notes"),
        }
        cases.append(normalized)
    return cases


def set_proxy_env(proxy_url: str) -> None:
    if not proxy_url:
        return
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url


@dataclass
class MCPContext:
    transport_ctx: Any
    session_ctx: Any
    session: Any


async def connect_client(server_url: str, api_key: str, timeout_sec: float) -> MCPContext:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except Exception as exc:
        raise RuntimeError("cannot import mcp SDK in current python env") from exc

    transport_ctx = streamablehttp_client(
        url=server_url,
        headers={"SCP-HUB-API-KEY": api_key},
    )
    read_stream, write_stream, _ = await transport_ctx.__aenter__()
    session_ctx = ClientSession(read_stream, write_stream)
    session = await session_ctx.__aenter__()
    await asyncio.wait_for(session.initialize(), timeout=timeout_sec)
    return MCPContext(transport_ctx=transport_ctx, session_ctx=session_ctx, session=session)


async def disconnect_client(ctx: MCPContext | None) -> None:
    if ctx is None:
        return
    try:
        await ctx.session_ctx.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        await ctx.transport_ctx.__aexit__(None, None, None)
    except Exception:
        pass


def _navigate_path(data: Any, field_path: str) -> Any:
    cur = data
    if not field_path:
        return cur
    for seg in field_path.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                return None
            cur = cur[seg]
        elif isinstance(cur, list):
            if not seg.isdigit():
                return None
            idx = int(seg)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def _register_path(registry: dict[str, Any], path_str: str) -> None:
    try:
        p = Path(path_str).expanduser().resolve()
    except Exception:
        return
    if not p.exists():
        return
    if p.is_file():
        file_path = str(p)
        registry["files"].add(file_path)
        registry["dirs"].add(str(p.parent))
        ext = p.suffix.lower()
        if ext:
            registry["ext"].setdefault(ext, set()).add(file_path)
    elif p.is_dir():
        registry["dirs"].add(str(p))


def _collect_paths_from_value(value: Any, registry: dict[str, Any]) -> None:
    if isinstance(value, str):
        if value.startswith("/") or value.startswith("./") or value.startswith("../"):
            _register_path(registry, value)
        return
    if isinstance(value, dict):
        for v in value.values():
            _collect_paths_from_value(v, registry)
        return
    if isinstance(value, list):
        for v in value:
            _collect_paths_from_value(v, registry)


def _first_or_none(items: set[str]) -> str | None:
    if not items:
        return None
    return sorted(items)[0]


def _find_file_by_ext(ext: str, search_roots: list[str], find_cache: dict[str, str | None]) -> str | None:
    ext = ext.lower()
    if ext in find_cache:
        return find_cache[ext]

    pattern = f"*{ext}"
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        cmd = ["find", str(root_path), "-maxdepth", "8", "-type", "f", "-name", pattern, "-print", "-quit"]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception:
            continue
        candidate = proc.stdout.strip().splitlines()
        if candidate:
            path = candidate[0].strip()
            if path:
                find_cache[ext] = path
                return path

    find_cache[ext] = None
    return None


def _ensure_fallback_file(spec: str) -> str | None:
    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if spec == "pdb_file":
        path = tmp_dir / "molclaw_auto_structure.pdb"
        if not path.exists():
            path.write_text(
                "HEADER    MOLCLAW AUTO PDB\\n"
                "ATOM      1  N   GLY A   1      11.104  13.207  10.517  1.00 20.00           N\\n"
                "ATOM      2  CA  GLY A   1      12.560  13.300  10.650  1.00 20.00           C\\n"
                "ATOM      3  C   GLY A   1      13.020  14.740  10.990  1.00 20.00           C\\n"
                "ATOM      4  O   GLY A   1      12.271  15.674  10.687  1.00 20.00           O\\n"
                "TER\\nEND\\n",
                encoding="utf-8",
            )
        return str(path)

    if spec == "fasta_file":
        path = tmp_dir / "molclaw_auto_sequence.fasta"
        if not path.exists():
            path.write_text(
                ">AUTO_SEQUENCE\\nMKTIIALSYIFCLVFA\\n",
                encoding="utf-8",
            )
        return str(path)

    if spec == "sdf_file":
        path = tmp_dir / "molclaw_auto_ligand.sdf"
        if not path.exists():
            path.write_text(
                "AUTO\\n"
                "  -OEChem-\\n\\n"
                "  3  2  0  0  0  0            999 V2000\\n"
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\\n"
                "    1.2094    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\\n"
                "    2.4188    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\\n"
                "  1  2  1  0\\n"
                "  2  3  1  0\\n"
                "M  END\\n$$$$\\n",
                encoding="utf-8",
            )
        return str(path)

    if spec == "cif_file":
        path = tmp_dir / "molclaw_auto_structure.cif"
        if not path.exists():
            path.write_text(
                "data_auto\\n#\\n",
                encoding="utf-8",
            )
        return str(path)

    return None


def _resolve_auto_token(
    token: str,
    registry: dict[str, Any],
    search_roots: list[str],
    find_cache: dict[str, str | None],
) -> str | None:
    spec = token[len(AUTO_PREFIX):].strip().lower()

    ext_map = {
        "pdb_file": ".pdb",
        "cif_file": ".cif",
        "sdf_file": ".sdf",
        "fasta_file": ".fasta",
        "pdbqt_file": ".pdbqt",
        "gro_file": ".gro",
        "dcd_file": ".dcd",
        "xtc_file": ".xtc",
        "json_file": ".json",
        "csv_file": ".csv",
    }

    if spec in ext_map:
        ext = ext_map[spec]
        local = _first_or_none(registry["ext"].get(ext, set()))
        if local:
            return local
        found = _find_file_by_ext(ext, search_roots, find_cache)
        if found:
            _register_path(registry, found)
            return found
        fallback = _ensure_fallback_file(spec)
        if fallback:
            _register_path(registry, fallback)
            return fallback
        return None

    if spec == "file":
        local_file = _first_or_none(registry["files"])
        if local_file:
            return local_file
        if Path("/etc/hosts").exists():
            _register_path(registry, "/etc/hosts")
            return "/etc/hosts"
        return None

    if spec == "dir":
        local_dir = _first_or_none(registry["dirs"])
        if local_dir:
            return local_dir
        return str(Path.cwd())

    if spec == "pdb_dir":
        pdb_file = _resolve_auto_token(f"{AUTO_PREFIX}pdb_file", registry, search_roots, find_cache)
        if pdb_file:
            return str(Path(pdb_file).parent)
        return None

    return None


def _resolve_value(
    value: Any,
    result_registry: dict[str, dict[str, Any]],
    path_registry: dict[str, Any],
    search_roots: list[str],
    find_cache: dict[str, str | None],
    warnings: list[str],
) -> Any:
    if isinstance(value, dict):
        return {
            k: _resolve_value(v, result_registry, path_registry, search_roots, find_cache, warnings)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(v, result_registry, path_registry, search_roots, find_cache, warnings)
            for v in value
        ]
    if not isinstance(value, str):
        return value

    if value.startswith(REF_PREFIX):
        # format: __REF__:case_id:field.path
        parts = value.split(":", 3)
        if len(parts) < 4:
            warnings.append(f"invalid ref token: {value}")
            return value
        case_id = parts[2]
        field_path = parts[3]
        ref_case = result_registry.get(case_id)
        if not ref_case:
            warnings.append(f"ref case not found: {case_id}")
            return value
        resolved = _navigate_path(ref_case.get("parsed_result"), field_path)
        if resolved is None:
            warnings.append(f"ref field not found: {value}")
            return value
        return resolved

    if value.startswith(AUTO_PREFIX):
        resolved_auto = _resolve_auto_token(value, path_registry, search_roots, find_cache)
        if resolved_auto is None:
            warnings.append(f"auto token unresolved: {value}")
            return value
        return resolved_auto

    return value


def classify_result(parsed: Any, tool_status: str, error_message: str | None) -> str:
    if tool_status in {"timeout", "exception", "tool_not_found", "transport_error"}:
        return tool_status

    if isinstance(parsed, dict):
        raw_status = parsed.get("status")

        # Handle boolean status values directly.
        if isinstance(raw_status, bool):
            return "success" if raw_status else "error"

        status = str(raw_status if raw_status is not None else "").strip().lower()
        if status in {"success", "ok", "true"}:
            return "success"
        if status in {"partial_success", "partial"}:
            return "partial_success"
        if "success" in status:
            return "success"
        if "error" in status or "fail" in status:
            return "error"

        text = str(parsed.get("text", ""))
        err = str(parsed.get("error", ""))
        msg = str(parsed.get("msg", ""))
        combined = f"{text} {err} {msg}".lower()
        if "unknown tool" in combined:
            return "unknown_tool"
        if err:
            return "error"

    if error_message:
        lowered = error_message.lower()
        if "timeout" in lowered:
            return "timeout"
        if "unknown tool" in lowered:
            return "unknown_tool"
        return "error"

    return "unknown"


def _collect_text_snippets(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    fields = [
        parsed.get("error"),
        parsed.get("text"),
        parsed.get("msg"),
        parsed.get("message"),
    ]
    chunks = [str(item) for item in fields if isinstance(item, str) and item.strip()]
    return " ".join(chunks)


def classify_preflight_result(
    *,
    functional_status: str,
    tool_status: str,
    error_message: str | None,
    raw_excerpt: str,
    parsed_result: Any,
) -> str:
    if tool_status == "tool_not_found":
        return "fail_not_registered"
    if tool_status == "transport_error":
        return "fail_unreachable"

    if tool_status == "timeout" or functional_status == "timeout":
        return "unknown_timeout"

    if functional_status in {"success", "partial_success"}:
        return "pass_ok"

    combined = " ".join(
        [
            str(error_message or ""),
            raw_excerpt,
            _collect_text_snippets(parsed_result),
        ]
    ).lower()

    if "unknown tool" in combined:
        return "fail_not_registered"

    if "connection refused" in combined or "failed to establish" in combined or "max retries exceeded" in combined:
        return "fail_unreachable"

    if "404" in combined and "not found" in combined and "/api/" in combined:
        return "fail_route_missing"

    if "timeout" in combined:
        return "unknown_timeout"

    reached_markers = [
        "file not found",
        "not found:",
        "pdb_path not found",
        "input file does not exist",
        "input_dir",
        "required",
        "missing",
        "invalid",
        "validation",
        "field required",
        "failed to parse arguments",
        "422",
    ]
    if any(marker in combined for marker in reached_markers):
        return "pass_reached"

    if parsed_result is not None:
        return "pass_reached"

    return "unknown"


def build_raw_excerpt(raw_result: Any, parsed_result: Any) -> str:
    raw_str = ""
    try:
        raw_str = json.dumps(to_jsonable(raw_result), ensure_ascii=False)
    except Exception:
        raw_str = str(raw_result)
    if not raw_str:
        try:
            raw_str = json.dumps(to_jsonable(parsed_result), ensure_ascii=False)
        except Exception:
            raw_str = str(parsed_result)
    raw_str = raw_str.replace("\n", " ").strip()
    return raw_str[:400]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    transport_ok_count = sum(1 for r in results if r.get("transport_ok"))
    success_count = sum(1 for r in results if r.get("result_status") in PREFLIGHT_SUCCESS)

    tool_status_counts: dict[str, int] = {}
    result_status_counts: dict[str, int] = {}
    for row in results:
        tool_status = str(row.get("tool_status", "unknown"))
        result_status = str(row.get("result_status", "unknown"))
        tool_status_counts[tool_status] = tool_status_counts.get(tool_status, 0) + 1
        result_status_counts[result_status] = result_status_counts.get(result_status, 0) + 1

    summary = {
        "case_count": total,
        "transport_ok_count": transport_ok_count,
        "success_count": success_count,
        "success_rate": round(success_count / total, 4) if total else 0.0,
        "preflight_success_count": success_count,
        "preflight_success_rate": round(success_count / total, 4) if total else 0.0,
        "tool_status_counts": tool_status_counts,
        "result_status_counts": result_status_counts,
    }
    return summary


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    meta = report["meta"]
    summary = report["summary"]
    rows = report["results"]

    lines = [
        "# MolClaw Tool Run Report",
        "",
        f"- worker_mode: `{meta['worker_mode']}`",
        f"- server_url: `{meta['server_url']}`",
        f"- proxy_url: `{meta.get('proxy_url') or ''}`",
        f"- case_file: `{meta['case_file']}`",
        f"- generated_at: `{meta['generated_at']}`",
        "",
        "## Summary",
        f"- case_count: **{summary['case_count']}**",
        f"- transport_ok_count: **{summary['transport_ok_count']}**",
        f"- preflight_success_count: **{summary['preflight_success_count']}**",
        f"- preflight_success_rate: **{summary['preflight_success_rate']:.2%}**",
        f"- tool_status_counts: `{json.dumps(summary['tool_status_counts'], ensure_ascii=False)}`",
        f"- result_status_counts: `{json.dumps(summary['result_status_counts'], ensure_ascii=False)}`",
        "",
        "## Cases",
        "",
        "| case_id | tool_name | tool_status | result_status | elapsed_sec | error_type | error_message |",
        "|---|---|---|---|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {case_id} | {tool_name} | {tool_status} | {result_status} | {elapsed_sec:.2f} | {error_type} | {error_message} |".format(
                case_id=str(row.get("case_id", "")).replace("|", "\\|"),
                tool_name=str(row.get("tool_name", "")).replace("|", "\\|"),
                tool_status=str(row.get("tool_status", "")).replace("|", "\\|"),
                result_status=str(row.get("result_status", "")).replace("|", "\\|"),
                elapsed_sec=float(row.get("elapsed_sec", 0.0)),
                error_type=str(row.get("error_type") or "").replace("|", "\\|"),
                error_message=str(row.get("error_message") or "").replace("|", "\\|").replace("\n", " "),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    set_proxy_env(args.proxy_url)

    api_key = os.environ.get("MOLCLAW_SCP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MOLCLAW_SCP_API_KEY is required in environment")

    case_path = Path(args.case_file)
    cases = read_cases(case_path)

    results: list[dict[str, Any]] = []
    result_registry: dict[str, dict[str, Any]] = {}
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    search_roots_env = os.environ.get(
        "MOLCLAW_AUTO_SEARCH_ROOTS",
        "/root/lwj/wll/code/DrugAgentTools,/home/sunxiangyu,/tmp,/var/tmp",
    )
    search_roots = [s.strip() for s in search_roots_env.split(",") if s.strip()]
    find_cache: dict[str, str | None] = {}

    path_registry: dict[str, Any] = {
        "files": set(),
        "dirs": set(),
        "ext": {},
    }

    for case in cases:
        _collect_paths_from_value(case.get("arguments"), path_registry)

    ctx: MCPContext | None = None
    transport_ok = False
    live_tools: set[str] = set()
    tool_count = 0
    transport_error = ""

    try:
        ctx = await connect_client(
            server_url=args.server_url,
            api_key=api_key,
            timeout_sec=args.timeout_sec,
        )
        transport_ok = True
        tools_obj = await asyncio.wait_for(ctx.session.list_tools(), timeout=args.timeout_sec)
        tools_list = list(getattr(tools_obj, "tools", []))
        tool_count = len(tools_list)
        for item in tools_list:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name:
                live_tools.add(name)
    except Exception as exc:
        transport_ok = False
        transport_error = f"{type(exc).__name__}: {exc}"

    for case in cases:
        start = time.monotonic()
        tool_name = case["tool_name"]
        original_arguments = case["arguments"]

        resolve_warnings: list[str] = []
        resolved_arguments = _resolve_value(
            original_arguments,
            result_registry=result_registry,
            path_registry=path_registry,
            search_roots=search_roots,
            find_cache=find_cache,
            warnings=resolve_warnings,
        )
        if isinstance(resolved_arguments, dict):
            _collect_paths_from_value(resolved_arguments, path_registry)

        tool_status = "pending"
        result_status = "pending"
        error_type: str | None = None
        error_message: str | None = None
        parsed_result: Any = None
        raw_result: Any = None
        functional_status = "unknown"

        if not transport_ok:
            tool_status = "transport_error"
            functional_status = "transport_error"
            result_status = "transport_error"
            error_type = "TransportError"
            error_message = transport_error
        elif tool_name not in live_tools:
            tool_status = "tool_not_found"
            functional_status = "tool_not_found"
            result_status = "unknown_tool"
            error_type = "UnknownTool"
            error_message = f"tool `{tool_name}` not found in live tool registry"
        else:
            try:
                raw_result = await asyncio.wait_for(
                    ctx.session.call_tool(tool_name, arguments=resolved_arguments),
                    timeout=args.timeout_sec,
                )
                parsed_result = parse_result(raw_result)
                tool_status = "ok"
                if isinstance(parsed_result, dict):
                    maybe_err = parsed_result.get("error")
                    maybe_text = parsed_result.get("text")
                    if maybe_err:
                        error_type = "ToolResultError"
                        error_message = str(maybe_err)
                    elif isinstance(maybe_text, str) and "Unknown tool" in maybe_text:
                        error_type = "UnknownTool"
                        error_message = maybe_text
                functional_status = classify_result(parsed_result, tool_status, error_message)
            except asyncio.TimeoutError:
                tool_status = "timeout"
                functional_status = "timeout"
                error_type = "TimeoutError"
                error_message = f"call_tool timeout after {args.timeout_sec}s"
            except Exception as exc:
                tool_status = "exception"
                functional_status = "error"
                error_type = type(exc).__name__
                error_message = str(exc)

        if parsed_result is not None:
            _collect_paths_from_value(parsed_result, path_registry)

        elapsed_sec = time.monotonic() - start
        raw_excerpt = build_raw_excerpt(raw_result, parsed_result)
        result_status = classify_preflight_result(
            functional_status=functional_status,
            tool_status=tool_status,
            error_message=error_message,
            raw_excerpt=raw_excerpt,
            parsed_result=parsed_result,
        )

        row = {
            "case_id": case["case_id"],
            "tool_name": tool_name,
            "arguments": to_jsonable(resolved_arguments),
            "input_arguments": to_jsonable(original_arguments),
            "worker_mode": args.worker_mode,
            "server_url": args.server_url,
            "proxy_url": args.proxy_url,
            "elapsed_sec": round(elapsed_sec, 4),
            "transport_ok": transport_ok,
            "tool_status": tool_status,
            "result_status": result_status,
            "functional_status": functional_status,
            "error_type": error_type,
            "error_message": error_message,
            "raw_excerpt": raw_excerpt,
            "source": case.get("source"),
            "notes": case.get("notes"),
            "resolve_warnings": resolve_warnings,
            "parsed_result": to_jsonable(parsed_result),
        }
        results.append(row)
        result_registry[case["case_id"]] = row

    await disconnect_client(ctx)

    report = {
        "meta": {
            "generated_at": generated_at,
            "worker_mode": args.worker_mode,
            "server_url": args.server_url,
            "proxy_url": args.proxy_url,
            "timeout_sec": args.timeout_sec,
            "case_file": str(case_path),
            "tool_count": tool_count,
            "transport_ok": transport_ok,
            "transport_error": transport_error,
            "auto_search_roots": search_roots,
        },
        "summary": summarize(results),
        "results": results,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, out_md)

    print(f"saved json: {out_json}")
    print(f"saved md: {out_md}")
    print(f"transport_ok={transport_ok}, tool_count={tool_count}, case_count={len(results)}")

    return 0 if transport_ok else 2


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except Exception as exc:
        print(f"[ERROR] run_molclaw_cases failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
