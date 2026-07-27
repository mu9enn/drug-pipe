from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drug_agent.gad.checkpoint_contract import load_and_validate_contract


class CheckpointContractTest(unittest.TestCase):
    def test_requires_exact_paired_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student = root / "student"
            discriminator = root / "discriminator"
            student.mkdir()
            discriminator.mkdir()
            manifest = root / "warmup_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "drug_agent_gad_warmup_v1",
                        "generator_warmup_checkpoint": str(student),
                        "discriminator_warmup_checkpoint": str(discriminator),
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_and_validate_contract(
                manifest,
                student_checkpoint=student,
                discriminator_checkpoint=discriminator,
            )
            self.assertEqual(loaded["schema_version"], "drug_agent_gad_warmup_v1")
            other = root / "other"
            other.mkdir()
            with self.assertRaises(ValueError):
                load_and_validate_contract(
                    manifest,
                    student_checkpoint=other,
                    discriminator_checkpoint=discriminator,
                )


if __name__ == "__main__":
    unittest.main()
