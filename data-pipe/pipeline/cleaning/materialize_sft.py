from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.cleaning.deployment_tools import (
    TOOL_VISIBILITY_CHOICES,
    load_deployment_tool_set,
)
from pipeline.cleaning.io import (
    base_manifest,
    read_jsonl,
    write_json,
    write_jsonl,
    write_pretty_json,
)
from pipeline.cleaning.sft_views import semantic_to_qwen35_sft


def materialize_sft(
    input_path: Path,
    output_root: Path,
    *,
    deployment_tool_set: Path,
    system_prompt: str,
    user_prompt_prefix: str = "",
    tool_visibility: str = "all",
) -> dict[str, Any]:
    tools = load_deployment_tool_set(deployment_tool_set)
    semantic, parse_errors = read_jsonl(input_path)
    if parse_errors:
        raise ValueError(f"invalid semantic JSONL: {parse_errors}")
    sft = [
        semantic_to_qwen35_sft(
            row,
            deployment_tools=tools,
            system_prompt=system_prompt,
            user_prompt_prefix=user_prompt_prefix,
            tool_visibility=tool_visibility,
        )
        for row in semantic
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "qwen35_sft.jsonl", sft)
    write_pretty_json(output_root / "qwen35_sft.pretty.json", sft)
    manifest = {
        **base_manifest(
            step="qwen35_structured_sft_materialization",
            source=input_path.resolve(),
            repo_root=Path(__file__).resolve().parents[3],
        ),
        "input_count": len(semantic),
        "sft_count": len(sft),
        "tool_visibility": tool_visibility,
        "system_prompt_sha256": hashlib.sha256(system_prompt.strip().encode()).hexdigest(),
        "user_prompt_prefix_sha256": hashlib.sha256(user_prompt_prefix.strip().encode()).hexdigest(),
        "deployment_tool_set": str(tools.source_path),
        "deployment_tool_set_sha256": tools.sha256,
        "tool_count_histogram": dict(Counter(str(len(row["tools"])) for row in sft)),
        "outputs": {
            "jsonl": str((output_root / "qwen35_sft.jsonl").resolve()),
            "pretty_json": str((output_root / "qwen35_sft.pretty.json").resolve()),
        },
    }
    write_json(output_root / "materialization_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize semantic trajectories as structured Qwen3.5 SFT data.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--deployment-tool-set", required=True, type=Path)
    parser.add_argument("--tool-visibility", choices=TOOL_VISIBILITY_CHOICES, default="all")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--system-prompt")
    prompt.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--user-prompt-prefix-file", type=Path)
    args = parser.parse_args()
    system_prompt = args.system_prompt or args.system_prompt_file.read_text(encoding="utf-8")
    print(
        json.dumps(
            materialize_sft(
                args.input,
                args.output_root,
                deployment_tool_set=args.deployment_tool_set,
                system_prompt=system_prompt,
                user_prompt_prefix=(
                    args.user_prompt_prefix_file.read_text(encoding="utf-8")
                    if args.user_prompt_prefix_file
                    else ""
                ),
                tool_visibility=args.tool_visibility,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
