#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect cleaned JSON files from llm_clean workdirs into a unified directory.",
    )
    parser.add_argument("--workdir-root", required=True, help="Root that contains per-sample workdirs.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for cleaned JSON files.")
    args = parser.parse_args()

    workdir_root = Path(args.workdir_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_dir(output_dir)

    copied = 0
    if workdir_root.is_dir():
        for src in sorted(workdir_root.glob("*/*-cleaned.json")):
            if not src.is_file():
                continue
            shutil.copy2(src, output_dir / src.name)
            copied += 1

    print(f"Copied {copied} cleaned JSON files to {output_dir}")


if __name__ == "__main__":
    main()
