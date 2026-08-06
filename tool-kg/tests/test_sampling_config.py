from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from molclaw_kg.io_utils import write_jsonl
from molclaw_kg.question_sampling.canonical_io import load_canonical_sampling_inputs
from molclaw_kg.question_sampling.config import resolve_sampling_profile


class SamplingConfigTest(unittest.TestCase):
    def test_named_profile_supplies_defaults_and_explicit_overrides_win(self) -> None:
        tool_kg_root = Path(__file__).parents[1]
        config = SimpleNamespace(
            paths=SimpleNamespace(configs=tool_kg_root / "configs")
        )
        default = resolve_sampling_profile(config, "simple_default")
        self.assertEqual(default.values["mode"], "simple_toolchain_question")
        self.assertEqual(default.values["target_successes"], 20)
        self.assertEqual(default.values["semantic_repair_rounds"], 1)
        self.assertEqual(
            default.values["fanout_runtime_target"],
            {
                "distribution": "normal_exponent",
                "arithmetic_mean_minutes": 15,
                "plus_3sigma_minutes": 60,
            },
        )
        self.assertEqual(default.prompt_path.name, "generate.md")
        self.assertEqual(default.prompt_path.parent.name, "user_prompts")
        self.assertEqual(default.prompt_path.parent.parent.name, "grounded-molclaw-task-generation")

        overridden = resolve_sampling_profile(
            config,
            "simple_default",
            overrides={"target_successes": 1, "random_seed": 123},
        )
        self.assertEqual(overridden.values["target_successes"], 1)
        self.assertEqual(overridden.values["random_seed"], 123)
        self.assertEqual(default.config_sha256, overridden.config_sha256)
        self.assertEqual(default.profile_sha256, overridden.profile_sha256)
        self.assertEqual(
            overridden.cli_overrides,
            {"target_successes": 1, "random_seed": 123},
        )
        payload = overridden.manifest_payload()
        self.assertEqual(payload["resolved_sampling_config"], overridden.values)
        self.assertEqual(payload["cli_overrides"], overridden.cli_overrides)
        self.assertEqual(
            set(payload["prompt_hashes"]),
            {"generation", "json_repair", "semantic_repair"},
        )

    def test_sampling_mode_cannot_be_changed_outside_named_profile(self) -> None:
        tool_kg_root = Path(__file__).parents[1]
        config = SimpleNamespace(
            paths=SimpleNamespace(configs=tool_kg_root / "configs")
        )
        with self.assertRaisesRegex(ValueError, "profile-owned"):
            resolve_sampling_profile(
                config,
                "simple_default",
                overrides={"mode": "unsupported_mode"},
            )

    def test_stage3_loads_only_canonical_outputs_without_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            results = run_dir / "results"
            results.mkdir()
            write_jsonl(
                results / "graph.jsonl",
                [
                    {
                        "pair_id": "pair::a__to__b",
                        "source_tool": "a",
                        "target_tool": "b",
                        "edge_type": "generates_partial_input_for",
                    }
                ],
            )
            write_jsonl(
                results / "tool_catalog.jsonl",
                [{"tool_id": "a"}, {"tool_id": "b"}],
            )
            write_jsonl(
                results / "edge_decisions.jsonl",
                [
                    {
                        "pair_id": "pair::a__to__b",
                        "source_tool": "a",
                        "target_tool": "b",
                        "edge_types": [{"type": "generates_partial_input_for"}],
                        "satisfied_inputs": [
                            {
                                "source_output_slot": "result",
                                "target_input_slot": "input",
                            }
                        ],
                        "source_authority": "claude_adjudication",
                    }
                ],
            )
            graph, cards, decisions = load_canonical_sampling_inputs(run_dir)
            self.assertEqual(len(graph), 1)
            self.assertEqual(len(cards), 2)
            self.assertEqual(decisions[0]["pair_id"], "pair::a__to__b")
            self.assertEqual(
                decisions[0]["satisfied_mappings"][0]["target_input_slot"],
                "input",
            )
            self.assertFalse((run_dir / "edge_debug_sidecar.jsonl").exists())
            self.assertFalse((run_dir / "intermediate").exists())


if __name__ == "__main__":
    unittest.main()
