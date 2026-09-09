#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from drug_agent.toolrl.v8_dataset import materialize_decisions, sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pretty_array(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write("[\n")
            for index, row in enumerate(rows):
                if index:
                    output.write(",\n")
                rendered = json.dumps(row, ensure_ascii=False, indent=2)
                output.write("\n".join("  " + line for line in rendered.splitlines()))
            output.write("\n]\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize native Qwen ToolRL decisions from v8 semantic trajectories.")
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--qwen-sft-view", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    rows, manifest = materialize_decisions(args.semantic.resolve(), args.qwen_sft_view.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_root / "all_decisions.jsonl"
    pretty = args.output_root / "all_decisions.pretty.json"
    manifest_path = args.output_root / "materialization_manifest.json"
    _write_jsonl(jsonl, rows)
    _write_pretty_array(pretty, rows)
    manifest["outputs"] = {
        "jsonl": str(jsonl.resolve()),
        "jsonl_sha256": sha256_file(jsonl),
        "pretty_json": str(pretty.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
