from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def _run_help(cmd: list[str], timeout_sec: float = 20.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return {
            "ok": proc.returncode == 0 and bool(text),
            "returncode": proc.returncode,
            "text": text,
            "cmd": cmd,
            "error": None,
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": None,
            "text": "",
            "cmd": cmd,
            "error": f"FileNotFoundError: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr_text = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "ok": False,
            "returncode": None,
            "text": (stdout_text + "\n" + stderr_text).strip(),
            "cmd": cmd,
            "error": f"TimeoutExpired: {exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "text": "",
            "cmd": cmd,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _has_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def _pick_flag(help_text: str, candidates: list[str]) -> str | None:
    for flag in candidates:
        if _has_flag(help_text, flag):
            return flag
    return None


def _detect_launcher(prefix_cmd: list[str], help_result: dict[str, Any]) -> dict[str, Any]:
    help_text = str(help_result.get("text") or "")
    if not help_result.get("ok"):
        return {"ok": False, "reason": help_result.get("error") or "help_not_available"}

    model_arg = _pick_flag(help_text, ["--model-path", "--model", "--model-name", "--model_name"])
    host_arg = _pick_flag(help_text, ["--host"])
    port_arg = _pick_flag(help_text, ["--port"])
    tp_arg = _pick_flag(help_text, ["--tensor-parallel-size", "--tp-size"])
    context_arg = _pick_flag(help_text, ["--context-length", "--context-len"])
    mem_arg = _pick_flag(help_text, ["--mem-fraction-static"])
    trust_arg = _pick_flag(help_text, ["--trust-remote-code"])

    required_missing = []
    if not model_arg:
        required_missing.append("model-path")
    if not port_arg:
        required_missing.append("port")
    if not tp_arg:
        required_missing.append("tensor-parallel")

    return {
        "ok": len(required_missing) == 0,
        "required_missing": required_missing,
        "model_arg": model_arg,
        "host_arg": host_arg,
        "port_arg": port_arg,
        "tp_arg": tp_arg,
        "context_arg": context_arg,
        "mem_arg": mem_arg,
        "trust_arg": trust_arg,
        "prefix_cmd": prefix_cmd,
        "help_text_len": len(help_text),
    }


def _build_cmd(
    detected: dict[str, Any],
    model_path: str,
    port: int,
    tp_size: int,
    host: str,
    context_length: int | None,
    mem_fraction_static: float | None,
    trust_remote_code: bool,
) -> list[str]:
    cmd: list[str] = list(detected["prefix_cmd"])
    cmd.extend([detected["model_arg"], str(model_path)])
    if detected.get("host_arg"):
        cmd.extend([detected["host_arg"], str(host)])
    cmd.extend([detected["port_arg"], str(port)])
    cmd.extend([detected["tp_arg"], str(tp_size)])
    if detected.get("context_arg") and context_length is not None:
        cmd.extend([detected["context_arg"], str(context_length)])
    if detected.get("mem_arg") and mem_fraction_static is not None:
        cmd.extend([detected["mem_arg"], str(mem_fraction_static)])
    if detected.get("trust_arg") and trust_remote_code:
        cmd.append(detected["trust_arg"])
    return cmd


def _error(message: str, help_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "selected_launcher": None,
        "cmd": None,
        "help_summary": help_summary or {},
        "error": {
            "category": "sglang_cli_incompatible",
            "message": message,
        },
    }


def detect_sglang_launch_command(
    model_path: str,
    port: int,
    tp_size: int,
    host: str = "0.0.0.0",
    context_length: int | None = 8192,
    mem_fraction_static: float | None = 0.80,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    serve_help = _run_help(["sglang", "serve", "--help"])
    launch_help = _run_help([sys.executable, "-m", "sglang.launch_server", "--help"])

    serve_detect = _detect_launcher(["sglang", "serve"], serve_help)
    launch_detect = _detect_launcher([sys.executable, "-m", "sglang.launch_server"], launch_help)

    help_summary = {
        "sglang_serve_available": bool(serve_help.get("ok")),
        "launch_server_available": bool(launch_help.get("ok")),
        "sglang_serve_returncode": serve_help.get("returncode"),
        "launch_server_returncode": launch_help.get("returncode"),
        "sglang_serve_missing": serve_detect.get("required_missing"),
        "launch_server_missing": launch_detect.get("required_missing"),
        "sglang_serve_help_error": serve_help.get("error"),
        "launch_server_help_error": launch_help.get("error"),
        "sglang_serve_help_excerpt": (serve_help.get("text") or "")[:800],
        "launch_server_help_excerpt": (launch_help.get("text") or "")[:800],
    }

    selected: dict[str, Any] | None = None
    selected_name = None

    # Prefer `sglang serve` only when help succeeds and required args are visible.
    if serve_detect.get("ok"):
        selected = serve_detect
        selected_name = "sglang serve"
    elif launch_detect.get("ok"):
        selected = launch_detect
        selected_name = f"{sys.executable} -m sglang.launch_server"

    if selected is None:
        return _error(
            "Cannot detect model-path argument from either sglang serve or python -m sglang.launch_server",
            help_summary=help_summary,
        )

    cmd = _build_cmd(
        detected=selected,
        model_path=model_path,
        port=port,
        tp_size=tp_size,
        host=host,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        trust_remote_code=trust_remote_code,
    )

    help_summary.update(
        {
            "selected_from": selected_name,
            "model_arg": selected.get("model_arg"),
            "tp_arg": selected.get("tp_arg"),
            "context_arg": selected.get("context_arg"),
            "mem_arg": selected.get("mem_arg"),
            "trust_arg": selected.get("trust_arg"),
        }
    )

    return {
        "ok": True,
        "selected_launcher": selected_name,
        "cmd": cmd,
        "help_summary": help_summary,
        "error": None,
    }


def detect_sglang_launch_command_json(**kwargs) -> str:
    return json.dumps(detect_sglang_launch_command(**kwargs), ensure_ascii=False, indent=2)
