from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from drug_agent.gad.discriminator import GADDiscriminator
from drug_agent.utils import read_jsonl


def _valid_pairs(rows):
    valid = []
    seen_sample_ids = set()
    for row in rows:
        if not (
            isinstance(row.get("state_messages"), list)
            and isinstance(row.get("teacher_response"), str)
            and isinstance(row.get("student_response"), str)
            and row["student_response"].strip()
        ):
            continue
        sample_id = row.get("sample_id")
        if sample_id and sample_id in seen_sample_ids:
            continue
        if sample_id:
            seen_sample_ids.add(sample_id)
        valid.append(row)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 GAD discriminator warmup")
    parser.add_argument("--pairs", required=True, help="JSONL negative cache with aligned teacher/student responses")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    input_rows = read_jsonl(args.pairs)
    rows = _valid_pairs(input_rows)
    if not rows:
        raise ValueError("No valid aligned GAD pairs")
    print(json.dumps({"event": "gad_warmup_data", "input_rows": len(input_rows), "valid_unique_pairs": len(rows)}))
    resume_backbone = Path(args.resume) / "backbone" if args.resume else None
    init_path = str(resume_backbone) if resume_backbone and resume_backbone.is_dir() else args.model_path
    discriminator = GADDiscriminator(init_path, lr=args.lr, max_length=args.max_length)
    if args.resume:
        discriminator.load(args.resume)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    step = 0
    for epoch in range(args.epochs):
        random.Random(1234 + epoch).shuffle(rows)
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            result = discriminator.score_and_update(
                [row["state_messages"] for row in batch],
                [row["teacher_response"] for row in batch],
                [row["student_response"] for row in batch],
            )
            step += 1
            metrics = {"step": step, "epoch": epoch, **result["metrics"], "version": result["version_after"]}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if step % args.save_interval == 0:
                discriminator.save(str(output / f"step_{step:06d}"))
    discriminator.save(str(output / "latest"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
