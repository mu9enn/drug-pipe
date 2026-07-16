#!/usr/bin/env python3
"""Build a CSV dataset from MolBench-E2E markdown questions."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


DEFAULT_QUESTION_IDS = (
    "E2E-Q01",
    "E2E-Q02",
    "E2E-Q03",
    "E2E-Q04",
    "E2E-Q05",
    "E2E-Q07",
    "E2E-Q08",
    "E2E-Q09",
)


def _parse_questions(raw: str) -> list[str]:
    if not raw.strip():
        return list(DEFAULT_QUESTION_IDS)
    out: list[str] = []
    for token in raw.split(","):
        qid = token.strip()
        if not qid:
            continue
        out.append(qid)
    return out


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    e2e_root = script_dir.parent
    repo_root = e2e_root.parent.parent
    default_questions_dir = repo_root / "molbench" / "MolBench-E2E" / "questions"
    default_out_csv = e2e_root / "data" / "e2e_dataset.csv"
    default_manifest = e2e_root / "data" / "dataset_manifest.json"

    parser = argparse.ArgumentParser(description="Build MolBench-E2E CSV from markdown questions.")
    parser.add_argument("--questions-dir", default=str(default_questions_dir), help="Directory containing E2E *.md files.")
    parser.add_argument(
        "--questions",
        default="",
        help="Comma-separated question ids (e.g. E2E-Q03,E2E-Q05). Default: built-in 8-question set.",
    )
    parser.add_argument("--out-csv", default=str(default_out_csv), help="Output CSV path.")
    parser.add_argument("--manifest-out", default=str(default_manifest), help="Output metadata JSON path.")
    args = parser.parse_args()

    questions_dir = Path(args.questions_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    manifest_out = Path(args.manifest_out).expanduser().resolve()
    selected_ids = _parse_questions(args.questions)

    if not questions_dir.is_dir():
        raise NotADirectoryError(questions_dir)
    if not selected_ids:
        raise ValueError("No question ids selected.")

    records: list[dict[str, str]] = []
    missing: list[str] = []
    for idx, qid in enumerate(selected_ids, start=1):
        q_path = questions_dir / f"{qid}.md"
        if not q_path.is_file():
            missing.append(str(q_path))
            continue
        question_text = q_path.read_text(encoding="utf-8")
        records.append(
            {
                "index": str(idx),
                "question_id": qid,
                "question": question_text,
                "answer": "[]",
                "source_file": str(q_path),
            }
        )

    if missing:
        raise FileNotFoundError("Missing question files:\n" + "\n".join(missing))
    if not records:
        raise RuntimeError("No records built.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question_id", "question", "answer"])
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "index": rec["index"],
                    "question_id": rec["question_id"],
                    "question": rec["question"],
                    "answer": rec["answer"],
                }
            )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "questions_dir": str(questions_dir),
        "out_csv": str(out_csv),
        "count": len(records),
        "selected_question_ids": [r["question_id"] for r in records],
        "records": [{"index": r["index"], "question_id": r["question_id"], "source_file": r["source_file"]} for r in records],
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"OUT_CSV={out_csv}")
    print(f"MANIFEST={manifest_out}")


if __name__ == "__main__":
    main()
