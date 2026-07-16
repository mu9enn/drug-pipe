#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


LLM_CLEAN_DIR = Path(__file__).resolve().parent
PROMPT_FILE = LLM_CLEAN_DIR / "prompt.md"
COLLECT_SCRIPT = LLM_CLEAN_DIR / "collect_llm_cleaned_json.py"


def _clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _render_prompt(prompt_path: Path, source_filename: str, source_stem: str, cleaned_filename: str) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    text = text.replace("{{SOURCE_FILENAME}}", source_filename)
    text = text.replace("{{SOURCE_STEM}}", source_stem)
    text = text.replace("{{CLEANED_FILENAME}}", cleaned_filename)
    return text


def _validate_cleaned_json(cleaned_path: Path) -> None:
    obj = json.loads(cleaned_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("top_level_not_object")
    for key in ("schema_version", "id", "messages"):
        if key not in obj:
            raise ValueError(f"missing_key:{key}")
    if not isinstance(obj["messages"], list):
        raise ValueError("messages_not_list")


def _iter_source_jsons(input_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(input_dir.glob("*.json")):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("-cleaned.json") or name.endswith(".cleaned.json"):
            continue
        out.append(path)
    return out


def _run_claude(claude_bin: str, workdir: Path, rendered_prompt: str) -> int:
    proc = subprocess.run(
        [claude_bin, "--dangerously-skip-permissions", "-p", rendered_prompt],
        cwd=str(workdir),
        text=True,
        check=False,
    )
    return int(proc.returncode)


def _collect_cleaned_files(python_bin: str, workdir_root: Path, output_dir: Path) -> int:
    proc = subprocess.run(
        [
            python_bin,
            str(COLLECT_SCRIPT),
            "--workdir-root",
            str(workdir_root),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        check=False,
    )
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run llm_clean over a directory of JSON trajectories and collect cleaned outputs.",
    )
    parser.add_argument("input_dir", help="Directory containing top-level *.json trajectory files.")
    parser.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    input_arg = Path(args.input_dir)
    if not input_arg.is_dir():
        print(f"[error] input directory not found: {input_arg}", file=sys.stderr)
        return 1

    input_dir = input_arg.resolve()
    claude_bin = str(args.claude_bin)
    python_bin = str(args.python_bin)

    if not PROMPT_FILE.is_file():
        print(f"[error] prompt file not found: {PROMPT_FILE}", file=sys.stderr)
        return 1
    if not COLLECT_SCRIPT.is_file():
        print(f"[error] collect script not found: {COLLECT_SCRIPT}", file=sys.stderr)
        return 1
    if shutil.which(claude_bin) is None:
        print(f"[error] claude binary not found in PATH: {claude_bin}", file=sys.stderr)
        return 1

    workdir_root = input_dir / "cc-workdir"
    output_dir = input_dir / "cleaned"

    workdir_root.mkdir(parents=True, exist_ok=True)
    _clear_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = _iter_source_jsons(input_dir)
    total = len(source_files)
    succeeded = 0
    failed = 0

    with tqdm(source_files, desc="LLM clean", unit="sample") as progress:
        for source_path in progress:
            source_filename = source_path.name
            source_stem = source_path.stem
            cleaned_filename = f"{source_stem}-cleaned.json"
            workdir = workdir_root / source_stem
            cleaned_path = workdir / cleaned_filename

            _clear_dir(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, workdir / source_filename)

            rendered_prompt = _render_prompt(PROMPT_FILE, source_filename, source_stem, cleaned_filename)

            tqdm.write(f"[run] {source_filename} -> {cleaned_filename}")
            claude_rc = _run_claude(claude_bin, workdir, rendered_prompt)

            if not cleaned_path.is_file():
                tqdm.write(f"[error] cleaned file missing: {cleaned_path}")
                failed += 1
                continue

            try:
                _validate_cleaned_json(cleaned_path)
            except Exception:
                tqdm.write(f"[error] cleaned file is not valid JSON or missing required keys: {cleaned_path}")
                try:
                    cleaned_path.unlink()
                except FileNotFoundError:
                    pass
                failed += 1
                continue

            if claude_rc != 0:
                tqdm.write(
                    f"[warn] claude exited non-zero for {source_filename} (rc={claude_rc}), but cleaned file validated"
                )

            succeeded += 1

    collect_rc = _collect_cleaned_files(python_bin, workdir_root, output_dir)
    if collect_rc != 0:
        print(f"[error] collect step failed with rc={collect_rc}", file=sys.stderr)
        failed += 1

    print(f"Total: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Cleaned output dir: {output_dir}")

    return 1 if failed != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
