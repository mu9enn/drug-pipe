from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.utils import ensure_dir, write_json


def _load_json_file(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(obj)
    return rows


def iter_records(input_path: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if input_path.is_dir():
        files = sorted([*input_path.glob("*.json"), *input_path.glob("*.jsonl")], key=lambda p: p.name)
    else:
        files = [input_path]

    for path in files:
        if path.suffix == ".json":
            yield path, _load_json_file(path)
        elif path.suffix == ".jsonl":
            for idx, obj in enumerate(_load_jsonl_file(path)):
                synthetic_path = path.with_name(f"{path.stem}__{idx:06d}.jsonl")
                yield synthetic_path, obj
        else:
            raise ValueError(f"Unsupported input file format: {path}")


def materialize_sft_jsonl(input_path: Path, output_jsonl: Path) -> dict[str, Any]:
    records = list(iter_records(input_path))
    ensure_dir(output_jsonl.parent)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    manifest = {
        "input_path": str(input_path),
        "output_jsonl": str(output_jsonl),
        "num_records": len(records),
        "source_files": [str(path) for path, _ in records[:10]],
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize ReAct SFT records from a directory of JSON files into JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input JSON directory / JSON file / JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest JSON path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    manifest = materialize_sft_jsonl(input_path, output_path)
    if args.manifest:
        write_json(Path(args.manifest).expanduser().resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
