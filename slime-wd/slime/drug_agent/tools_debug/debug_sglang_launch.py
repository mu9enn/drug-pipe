from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.constants import DATA_ROOT, SLIME_DRUG_RUNS_ROOT
from drug_agent.tools_debug.sglang_launcher import detect_sglang_launch_command


def _default_model_path() -> Path:
    data_root = Path(os.environ.get("DATA", "/root/slime_sxy/group-space/sunxiangyu/slime-wd/data"))
    model_122b = Path(os.environ.get("DEBUG_MODEL_PATH", str(data_root / "Qwen3.5-122B-A10B")))
    if model_122b.is_dir():
        return model_122b
    return DATA_ROOT / "Qwen3.5-27B"


def _error(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _http_get_text(url: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return True, resp.read().decode("utf-8")
    except Exception as exc:
        return False, str(exc)


def _wait_health(base_url: str, timeout_sec: float = 300.0) -> tuple[bool, str]:
    start = time.monotonic()
    last = ""
    while (time.monotonic() - start) < timeout_sec:
        ok, text = _http_get_text(f"{base_url}/health", timeout_sec=5.0)
        if ok:
            return True, text
        last = text
        time.sleep(2.0)
    return False, last or "health timeout"


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and optionally launch SGLang with CLI-compatible args")
    parser.add_argument("--model-path", type=str, default=str(_default_model_path()))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--health-timeout-sec", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-running", action="store_true", help="Block in foreground after successful startup")
    parser.add_argument("--auto-stop", action="store_true", help="Stop spawned SGLang process before exit")
    args = parser.parse_args()

    started = time.monotonic()
    output: dict[str, Any] = {
        "ok": False,
        "dry_run": bool(args.dry_run),
        "model_path": args.model_path,
        "selected_launcher": None,
        "cmd": None,
        "error": None,
        "latency_sec": 0.0,
    }

    proc: subprocess.Popen | None = None
    log_fp = None
    stop_on_exit = False
    try:
        model_path = Path(args.model_path)
        if not model_path.is_dir():
            output["error"] = _error("model_not_found", f"model path not found: {model_path}")
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1

        detected = detect_sglang_launch_command(
            model_path=str(model_path),
            port=args.port,
            tp_size=args.tp_size,
            host=args.host,
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
            trust_remote_code=(not args.no_trust_remote_code),
        )
        output["selected_launcher"] = detected.get("selected_launcher")
        output["cmd"] = detected.get("cmd")
        output["help_summary"] = detected.get("help_summary")

        if not detected.get("ok"):
            output["error"] = detected.get("error")
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1

        if args.dry_run:
            output["ok"] = True
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0

        log_path = SLIME_DRUG_RUNS_ROOT / "sglang_drug_agent_debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            detected["cmd"],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
        log_fp.close()
        log_fp = None

        base_url = f"http://127.0.0.1:{args.port}"
        ok, health_text = _wait_health(base_url, timeout_sec=args.health_timeout_sec)
        if not ok:
            output["error"] = _error("sglang_launch_failed", health_text)
            output["log_path"] = str(log_path)
            output["pid"] = proc.pid if proc else None
            stop_on_exit = True
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1

        models_ok, models_text = _http_get_text(f"{base_url}/v1/models", timeout_sec=10.0)
        process_alive = proc.poll() is None if proc else False
        output["ok"] = bool(models_ok)
        output["health"] = health_text
        output["models"] = models_text if models_ok else None
        output["log_path"] = str(log_path)
        output["pid"] = proc.pid if proc else None
        output["process_alive"] = process_alive
        if (not process_alive) and proc is not None:
            output["error"] = _error("sglang_exited_early", f"sglang exited with code {proc.returncode}")
            output["ok"] = False
        if not models_ok:
            output["error"] = _error("sglang_models_unavailable", models_text)
            stop_on_exit = True

        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))

        if args.keep_running:
            # Keep process alive for manual debugging.
            while True:
                time.sleep(10)
        return 0 if output["ok"] else 1
    except KeyboardInterrupt:
        stop_on_exit = True
        output["error"] = _error("interrupted", "keyboard interrupt")
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 130
    except Exception as exc:
        stop_on_exit = True
        output["error"] = _error("runtime_error", f"{type(exc).__name__}: {exc}")
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1
    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass

        if args.auto_stop or stop_on_exit:
            _stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
