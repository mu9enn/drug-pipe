#!/usr/bin/env python3
"""Create a metadata-correct BF16 HF view without duplicating model shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise SystemExit("source and output must differ")
    output.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.name == "config.json":
            continue
        target = output / item.name
        if target.exists() or target.is_symlink():
            continue
        if item.is_file():
            os.link(item, target)
        elif item.is_symlink():
            target.symlink_to(os.readlink(item))

    config = json.loads((source / "config.json").read_text())
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    if isinstance(config.get("text_config"), dict):
        config["text_config"]["dtype"] = "bfloat16"
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "quantization_config": config.get("quantization_config"),
                "dtype": config.get("dtype"),
                "hardlinked_files": sum(1 for path in output.iterdir() if path.is_file()) - 1,
            }
        )
    )


if __name__ == "__main__":
    main()
