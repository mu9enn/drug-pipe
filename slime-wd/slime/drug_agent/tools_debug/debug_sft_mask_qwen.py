from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield line_no, json.loads(s)


def _preview_messages(messages: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    preview = []
    for msg in messages[:limit]:
        preview.append(
            {
                "role": msg.get("role"),
                "content": (msg.get("content") or "")[:200] if isinstance(msg.get("content"), str) else msg.get("content"),
            }
        )
    return preview


def _get_sample_id(obj: dict[str, Any], fallback: str) -> str:
    sample_id = obj.get("id")
    if isinstance(sample_id, str) and sample_id.strip():
        return sample_id
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for key in ("task_id", "id", "sample_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return fallback


def _run_one_type(
    jsonl_path: Path,
    tokenizer_path: str,
    tokenizer_type: str,
    max_samples: int | None,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=tokenizer_type)

    passed = 0
    total = 0
    first_failure = None

    for line_no, obj in _iter_jsonl(jsonl_path):
        if max_samples is not None and total >= max_samples:
            break

        total += 1
        messages = obj.get("messages")
        if not isinstance(messages, list):
            first_failure = {
                "line": line_no,
                "sample_id": _get_sample_id(obj, f"line_{line_no}"),
                "roles": [],
                "mask_generation_failed": True,
                "exception_type": "ValueError",
                "exception_message": "`messages` must be a list",
                "messages_preview": None,
            }
            break

        try:
            tools = obj.get("tools")
            if not isinstance(tools, list) or not tools:
                raise ValueError("`tools` must be a non-empty list")
            token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
            if len(token_ids) != len(loss_mask):
                raise ValueError(
                    f"token_ids/loss_mask length mismatch: {len(token_ids)} != {len(loss_mask)}"
                )
            passed += 1
        except Exception as exc:
            first_failure = {
                "line": line_no,
                "sample_id": _get_sample_id(obj, f"line_{line_no}"),
                "roles": [msg.get("role") for msg in messages],
                "mask_generation_failed": True,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "messages_preview": _preview_messages(messages),
            }
            break

    return {
        "tokenizer_type": tokenizer_type,
        "input_jsonl": str(jsonl_path),
        "tokenizer_path": tokenizer_path,
        "passed": passed,
        "total": total,
        "mask_generation_failed": first_failure is not None,
        "first_failure": first_failure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug Qwen/Qwen3.5 loss masks on structured SFT JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Structured messages + tools JSONL")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer/model path")
    parser.add_argument(
        "--tokenizer-types",
        nargs="+",
        default=["qwen", "qwen3_5"],
        help="Tokenizer types to test in order",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for quick checks")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not input_path.is_file() or input_path.suffix != ".jsonl":
        raise ValueError("--input must be a structured JSONL file")
    jsonl_path = input_path

    results = {}
    for tokenizer_type in args.tokenizer_types:
        results[tokenizer_type] = _run_one_type(
            jsonl_path=jsonl_path,
            tokenizer_path=args.tokenizer,
            tokenizer_type=tokenizer_type,
            max_samples=args.max_samples,
        )

    output = {
        "input": str(input_path),
        "effective_input_jsonl": str(jsonl_path),
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    for item in results.values():
        if item.get("mask_generation_failed"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
