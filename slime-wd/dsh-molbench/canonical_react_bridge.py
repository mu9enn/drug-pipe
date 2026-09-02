#!/usr/bin/env python3
"""OpenAI chat-completions facade for canonical Drug-Agent ReAct models.

DSH remains responsible for sessions and tool execution.  This bridge hides
DSH's native function-call protocol from the model, renders the exact
canonical ReAct prompt used by Drug-Agent training, and translates canonical
tool calls back into OpenAI structured calls for DSH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from drug_agent.constants import DEFAULT_SYSTEM_PROMPT
from drug_agent.protocol.prompts import format_final_contract, format_tool_catalog
from drug_agent.protocol.react_protocol import final_answer_matches_task, parse_runtime_decision


TASK_MARKER = "DSH_CANONICAL_REACT_TASK_V1\n"
MCP_PREFIX = "mcp__molclaw-scp__"
LOCAL_DSH_TO_CANONICAL = {
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "bash": "Bash",
    "grep": "Grep",
    "glob": "Glob",
}
LOCAL_CANONICAL_TO_DSH = {value: key for key, value in LOCAL_DSH_TO_CANONICAL.items()}
ROLLOUT_FORMAT_REMINDER = (
    "/no_think\n"
    "Use canonical ReAct XML. Put reasoning in <thought>...</thought>, followed by "
    "one or more <tool_call>{\"tool_name\":\"...\",\"arguments\":{...}}</tool_call> blocks, "
    "or one task-specific <final_answer>{...}</final_answer> block. "
    "Never mix tool calls and final answer in one generation."
)
LOCAL_TOOL_REMINDER = (
    "\nAvailable local tools: Read, Write, Edit, Bash, Grep, and Glob. "
    "They operate only in this task's workspace; Read, Grep, and Glob may inspect "
    "the read-only execute-molclaw-trajectory skill bundle under .dsh/skills."
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def dsh_to_canonical_name(name: Any) -> str | None:
    value = str(name or "").strip()
    if value.startswith(MCP_PREFIX):
        bare = value[len(MCP_PREFIX):].strip()
        return bare or None
    return LOCAL_DSH_TO_CANONICAL.get(value.lower())


def canonical_to_dsh_name(name: Any, name_map: dict[str, str]) -> str | None:
    value = str(name or "").strip()
    if not value:
        return None
    return name_map.get(value) or name_map.get(value.lower()) or LOCAL_CANONICAL_TO_DSH.get(value)


def extract_task(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = content_text(message.get("content"))
        position = text.find(TASK_MARKER)
        if position < 0:
            continue
        payload = json.loads(text[position + len(TASK_MARKER):].strip())
        if not isinstance(payload, dict):
            raise ValueError("bridge task marker payload must be an object")
        required = {"task_id", "task_type", "question"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"bridge task marker missing keys: {missing}")
        task_type = str(payload["task_type"]).lower()
        if task_type not in {"pf", "ac"}:
            raise ValueError(f"unsupported bridge task_type: {task_type!r}")
        payload["task_type"] = task_type
        payload["task_id"] = str(payload["task_id"])
        payload["question"] = str(payload["question"])
        return payload
    raise ValueError("request does not contain a DSH canonical-ReAct task marker")


def tool_catalog(tools: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    specs: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    for entry in tools if isinstance(tools, list) else []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") if entry.get("type") == "function" else entry
        if not isinstance(function, dict):
            continue
        dsh_name = str(function.get("name") or "").strip()
        bare_name = dsh_to_canonical_name(dsh_name)
        if not bare_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = function.get("input_schema")
        if not isinstance(parameters, dict):
            parameters = {}
        specs.append({
            "name": bare_name,
            "description": str(function.get("description") or "").strip(),
            "input_schema": parameters,
        })
        name_map[bare_name] = dsh_name
        name_map[bare_name.lower()] = dsh_name
    if not specs:
        raise ValueError("DSH request exposes no canonical MolClaw/local tools")
    return specs, name_map


def canonical_observation(tool_name: str, content: Any, *, is_error: bool = False) -> str:
    parsed_content: Any = content if isinstance(content, (dict, list)) else content_text(content)
    if isinstance(parsed_content, str):
        try:
            parsed_content = json.loads(parsed_content)
        except (json.JSONDecodeError, TypeError):
            pass
    payload = {
        "tool_name": tool_name,
        "status": "error" if is_error else "success",
        "is_error": bool(is_error),
        "content": {
            "result": None if is_error else parsed_content,
            "error": parsed_content if is_error else None,
        },
        "metadata": {},
    }
    return f'\n<observation tool_name="{tool_name}">{compact_json(payload)}</observation>\n'


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def reconstruct_trace(messages: list[dict[str, Any]]) -> str:
    call_names: dict[str, str] = {}
    chunks: list[str] = []
    marker_seen = False
    for message in messages:
        role = message.get("role")
        text = content_text(message.get("content"))
        if role == "user" and TASK_MARKER in text:
            marker_seen = True
            continue
        if not marker_seen:
            continue
        if role == "assistant":
            if text:
                chunks.append(text)
            for call in _tool_calls(message):
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                bare = dsh_to_canonical_name(function.get("name"))
                if not bare:
                    continue
                call_id = str(call.get("id") or "")
                if call_id:
                    call_names[call_id] = bare
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                chunks.append(
                    f'<tool_call>{compact_json({"tool_name": bare, "arguments": arguments})}</tool_call>'
                )
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            bare = call_names.get(call_id) or dsh_to_canonical_name(message.get("name")) or "runtime"
            is_error = bool(message.get("is_error") or message.get("isError"))
            chunks.append(canonical_observation(bare, message.get("content"), is_error=is_error))
    return "".join(chunks)


def build_prompt_text(
    tokenizer: Any,
    task: dict[str, Any],
    specs: list[dict[str, Any]],
    trace: str,
) -> str:
    system = (
        DEFAULT_SYSTEM_PROMPT
        + "\n\n"
        + ROLLOUT_FORMAT_REMINDER
        + LOCAL_TOOL_REMINDER
        + "\n"
        + format_tool_catalog(specs)
        + "\n"
        + format_final_contract(task["task_type"])
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["question"]},
    ]
    try:
        initial = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        initial = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return initial + trace


def _schema_type_ok(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    return checks.get(expected, lambda _item: True)(value)


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    if schema.get("type") == "object" and not isinstance(arguments, dict):
        return "arguments must be an object"
    for key in schema.get("required", []) if isinstance(schema.get("required"), list) else []:
        if key not in arguments:
            return f"missing required argument: {key}"
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for key, value in arguments.items():
        rule = properties.get(key)
        if not isinstance(rule, dict):
            continue
        expected = rule.get("type")
        if isinstance(expected, str) and not _schema_type_ok(value, expected):
            return f"argument {key!r} must have type {expected}"
    return None


@dataclass
class BridgeConfig:
    upstream_url: str
    model_id: str
    max_new_tokens: int
    max_prompt_tokens: int
    max_steps: int
    repeat_limit: int
    log_path: Path


class CanonicalBridge:
    def __init__(self, config: BridgeConfig, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self._log_lock = threading.Lock()
        self._repeat_lock = threading.Lock()
        self._repeat_state: dict[str, tuple[str, int]] = {}
        self._catalog_lock = threading.Lock()
        self._catalog_hash: str | None = None

    def log(self, payload: dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **payload}
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with self.config.log_path.open("a", encoding="utf-8") as handle:
                handle.write(compact_json(payload) + "\n")

    def call_upstream(self, input_ids: list[int]) -> tuple[str, str]:
        request_body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": self.config.max_new_tokens,
                "stop": ["<observation"],
                "no_stop_trim": True,
            },
        }
        request = urllib.request.Request(
            self.config.upstream_url,
            data=json.dumps(request_body).encode("utf-8"),
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang upstream HTTP {exc.code}: {detail[:1000]}") from exc
        text = str(result.get("text") or "")
        finish = (((result.get("meta_info") or {}).get("finish_reason") or {}).get("type") or "stop")
        return text, str(finish)

    def _repeat(self, task_id: str, raw: str) -> int:
        digest = hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()
        with self._repeat_lock:
            previous, count = self._repeat_state.get(task_id, ("", 0))
            count = count + 1 if previous == digest else 1
            self._repeat_state[task_id] = (digest, count)
            return count

    def complete(self, request_body: dict[str, Any]) -> dict[str, Any]:
        messages = request_body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        task = extract_task(messages)
        specs, name_map = tool_catalog(request_body.get("tools"))
        catalog_hash = hashlib.sha256(
            compact_json(sorted(specs, key=lambda item: str(item.get("name") or ""))).encode("utf-8")
        ).hexdigest()
        with self._catalog_lock:
            if self._catalog_hash is None:
                self._catalog_hash = catalog_hash
            elif self._catalog_hash != catalog_hash:
                raise RuntimeError(
                    f"DSH canonical tool catalog changed during run: "
                    f"{self._catalog_hash} != {catalog_hash}"
                )
        schemas = {str(spec["name"]): spec["input_schema"] for spec in specs}
        trace = reconstruct_trace(messages)
        assistant_turns = sum(1 for item in messages if item.get("role") == "assistant")
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        raw_outputs: list[str] = []
        synthetic_trace = ""
        decision: dict[str, Any] | None = None
        finish_type = "stop"

        for internal_step in range(8):
            if assistant_turns + internal_step >= self.config.max_steps:
                raw_outputs.append("<thought>Maximum canonical ReAct step limit reached.</thought>")
                break
            prompt_text = build_prompt_text(self.tokenizer, task, specs, trace + synthetic_trace)
            input_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if len(input_ids) > self.config.max_prompt_tokens:
                raise ValueError(
                    f"canonical prompt exceeds max_prompt_tokens: {len(input_ids)} > "
                    f"{self.config.max_prompt_tokens}"
                )
            raw, finish_type = self.call_upstream(list(input_ids))
            raw_outputs.append(raw)
            if finish_type == "length" or "<observation" in raw:
                break
            parsed = parse_runtime_decision(raw, strict_toolrl_turn=True)
            if not parsed.get("ok"):
                break
            if self._repeat(task["task_id"], raw) >= self.config.repeat_limit:
                raw_outputs.append("<thought>Repeated identical canonical decision; terminating.</thought>")
                decision = None
                break
            if parsed.get("decision_type") == "final_answer":
                final_answer = parsed.get("final_answer")
                if final_answer_matches_task(final_answer, task["task_type"]):
                    decision = parsed
                    break
                synthetic_trace += raw + canonical_observation(
                    "runtime",
                    {
                        "error_type": "FinalTaskTypeMismatch",
                        "error_message": f"final_answer.task_type must be {task['task_type']!r}",
                    },
                    is_error=True,
                )
                continue
            calls = parsed.get("tool_calls") or []
            validation_errors: list[dict[str, Any]] = []
            for call in calls:
                bare = str(call.get("tool_name") or "")
                dsh_name = canonical_to_dsh_name(bare, name_map)
                if not dsh_name:
                    validation_errors.append({"tool_name": bare, "error": "unknown or disallowed tool"})
                    continue
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                error = validate_arguments(arguments, schemas.get(bare, {}))
                if error:
                    validation_errors.append({"tool_name": bare, "error": error})
            if validation_errors:
                synthetic_trace += raw
                for error in validation_errors:
                    synthetic_trace += canonical_observation(
                        error["tool_name"] or "runtime", error, is_error=True
                    )
                continue
            decision = parsed
            break

        if decision and decision.get("decision_type") == "tool_call":
            tool_calls = []
            current_raw = raw_outputs[-1] if raw_outputs else ""
            thoughts = "".join(
                f"<thought>{body}</thought>"
                for body in re.findall(r"<thought>(.*?)</thought>", current_raw, re.DOTALL)
            )
            content = synthetic_trace + thoughts
            for index, call in enumerate(decision.get("tool_calls") or []):
                bare = str(call["tool_name"])
                dsh_name = canonical_to_dsh_name(bare, name_map)
                if not dsh_name:
                    raise RuntimeError(f"validated tool disappeared from name map: {bare}")
                tool_calls.append({
                    "id": f"bridge_{task['task_id']}_{uuid.uuid4().hex[:12]}_{index}",
                    "type": "function",
                    "function": {
                        "name": dsh_name,
                        "arguments": compact_json(call.get("arguments") or {}),
                    },
                })
            message = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            content = synthetic_trace + (raw_outputs[-1] if raw_outputs else "")
            message = {"role": "assistant", "content": content}
            finish_reason = "length" if finish_type == "length" else "stop"

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self.log({
            "request_id": request_id,
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "tool_catalog_sha256": catalog_hash,
            "assistant_turns": assistant_turns,
            "raw_outputs": raw_outputs,
            "parsed_decision": decision,
            "finish_type": finish_type,
            "openai_finish_reason": finish_reason,
            "tool_calls": message.get("tool_calls", []),
        })
        return response


class Handler(BaseHTTPRequestHandler):
    bridge: CanonicalBridge

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._json(HTTPStatus.OK, {
                "object": "list",
                "data": [{"id": self.bridge.config.model_id, "object": "model"}],
            })
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = self.bridge.complete(payload)
            if payload.get("stream"):
                self._stream(response)
            else:
                self._json(HTTPStatus.OK, response)
        except Exception as exc:
            self.bridge.log({"kind": "bridge_error", "error": f"{type(exc).__name__}: {exc}"})
            self._json(HTTPStatus.BAD_REQUEST, {
                "error": {"type": type(exc).__name__, "message": str(exc)}
            })

    def _stream(self, response: dict[str, Any]) -> None:
        choice = response["choices"][0]
        message = choice["message"]
        first = {
            "id": response["id"], "object": "chat.completion.chunk",
            "created": response["created"], "model": response["model"],
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        deltas: list[dict[str, Any]] = [first]
        content = message.get("content")
        if content:
            deltas.append({
                **{key: response[key] for key in ("id", "created", "model")},
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            })
        if message.get("tool_calls"):
            calls = []
            for index, call in enumerate(message["tool_calls"]):
                calls.append({"index": index, **call})
            deltas.append({
                **{key: response[key] for key in ("id", "created", "model")},
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}],
            })
        deltas.append({
            **{key: response[key] for key in ("id", "created", "model")},
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        })
        body = b"".join(
            f"data: {json.dumps(item, ensure_ascii=False)}\n\n".encode("utf-8") for item in deltas
        ) + b"data: [DONE]\n\n"
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:30000/generate")
    parser.add_argument("--model-id", default="qwen3.5-9b-canonical-bridge")
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-prompt-tokens", type=int, default=49152)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--repeat-limit", type=int, default=3)
    parser.add_argument("--log-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    config = BridgeConfig(
        upstream_url=args.upstream_url,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_steps=args.max_steps,
        repeat_limit=args.repeat_limit,
        log_path=args.log_path,
    )
    Handler.bridge = CanonicalBridge(config, tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(compact_json({"event": "bridge_ready", "host": args.host, "port": args.port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
