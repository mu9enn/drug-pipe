from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_and_validate_contract(
    manifest_path: Path,
    *,
    student_checkpoint: Path,
    discriminator_checkpoint: Path,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "drug_agent_gad_warmup_v1":
        raise ValueError("invalid GAD warmup manifest schema")
    expected_student = Path(str(payload.get("generator_warmup_checkpoint") or "")).resolve()
    expected_discriminator = Path(str(payload.get("discriminator_warmup_checkpoint") or "")).resolve()
    actual_student = student_checkpoint.resolve()
    actual_discriminator = discriminator_checkpoint.resolve()
    if expected_student != actual_student:
        raise ValueError(f"student checkpoint does not match warmup manifest: {actual_student} != {expected_student}")
    if expected_discriminator != actual_discriminator:
        raise ValueError(
            f"discriminator checkpoint does not match warmup manifest: {actual_discriminator} != {expected_discriminator}"
        )
    for label, path in (
        ("student", actual_student),
        ("discriminator", actual_discriminator),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} warmup checkpoint does not exist: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paired GAD generator/discriminator warmup checkpoints")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--discriminator-checkpoint", required=True)
    args = parser.parse_args()
    payload = load_and_validate_contract(
        Path(args.manifest),
        student_checkpoint=Path(args.student_checkpoint),
        discriminator_checkpoint=Path(args.discriminator_checkpoint),
    )
    print(json.dumps({"ok": True, "manifest": args.manifest, "contract": payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
