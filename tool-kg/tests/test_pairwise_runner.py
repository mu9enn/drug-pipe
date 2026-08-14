from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

from molclaw_kg.pairwise_runner import _run_pairwise_attempt


class FakeAdjudicator:
    model_name = "fake-model"
    last_trace = {}

    def __init__(self, config) -> None:
        self.config = config

    def adjudicate(self, payload):
        return {"pair_id": payload["pair_id"]}


class PairwiseRunnerTest(unittest.TestCase):
    def test_worker_hashes_prompt_template_without_name_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir()
            skills_root = root / "skills"
            (skills_root / ".claude" / "skills").mkdir(parents=True)
            config = SimpleNamespace(
                paths=SimpleNamespace(run_dir=run_dir),
                runtime=SimpleNamespace(skills_root=skills_root),
            )
            pair = {
                "pair_id": "pair::source__to__target",
                "source_tool": "source",
                "target_tool": "target",
                "source_stage": "source_stage",
                "target_stage": "target_stage",
            }
            cards = {
                "source": {"tool_id": "source"},
                "target": {"tool_id": "target"},
            }
            ontology = SimpleNamespace(version="test-v1")

            with (
                patch("molclaw_kg.pairwise_runner.AgentCCAdjudicator", FakeAdjudicator),
                patch("molclaw_kg.pairwise_runner._prepare_pair_workdir"),
                patch("molclaw_kg.pairwise_runner.validate_adjudication_output"),
            ):
                record, cache_entry = _run_pairwise_attempt(
                    config=config,
                    pair=pair,
                    cards=cards,
                    prompt_template="system prompt from skill",
                    taxonomy_version="taxonomy-v1",
                    cache_path=run_dir / "cache.jsonl",
                    cache_lock=threading.Lock(),
                    rerun_round=0,
                    taxonomy_raw={},
                    ontology=ontology,
                    adjudication_schema={},
                )

            self.assertTrue(record["response_schema_ok"])
            self.assertNotEqual(record["cache_key"], "")
            self.assertEqual(record["response"]["pair_id"], pair["pair_id"])
            self.assertEqual(cache_entry["cache_key"], record["cache_key"])


if __name__ == "__main__":
    unittest.main()
