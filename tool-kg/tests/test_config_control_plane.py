from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from molclaw_kg.settings import build_config
from molclaw_kg.stage_taxonomy import load_stage_taxonomy


class ConfigControlPlaneTest(unittest.TestCase):
    def test_default_skills_root_is_the_repository_canonical_bundle(self) -> None:
        tool_kg_root = Path(__file__).resolve().parents[1]
        with mock.patch.object(Path, "mkdir"):
            config = build_config(tool_kg_root, run_id="canonical_skills")

        self.assertEqual(
            config.runtime.skills_root,
            tool_kg_root.parent / "molclaw-skills",
        )
        self.assertTrue((config.runtime.skills_root / "CLAUDE.md").is_file())
        self.assertTrue(
            (config.runtime.skills_root / "system_prompt_FULL.md").is_file()
        )

    def test_build_config_does_not_load_unrelated_stage_configs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = build_config(root, run_id="config_only")
            self.assertEqual(config.paths.root, root.resolve())
            self.assertFalse(hasattr(config, "stage_taxonomy_path"))
            self.assertFalse(hasattr(config, "edge_ontology"))
            self.assertFalse(hasattr(config, "semantic_types"))
            self.assertFalse(hasattr(config, "rules"))

    def test_mainline_taxonomy_contains_scheduling_policy_only(self) -> None:
        tool_kg_root = Path(__file__).parents[1]
        path = tool_kg_root / "configs" / "stage_taxonomy.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        taxonomy = load_stage_taxonomy(path)

        self.assertEqual(taxonomy.version, "stage_taxonomy_v2")
        self.assertTrue(taxonomy.tool_stage_map)
        forbidden = {
            "same_stage_policy",
            "edge_type_stage_policy",
            "pruning_policy",
            "coverage_policy",
            "expected_tool_count",
        }
        self.assertFalse(forbidden.intersection(raw))
        self.assertTrue(all("tools" not in spec for spec in raw["stages"].values()))
        self.assertTrue(
            all(
                "scheduling_stages" in spec
                and "secondary_stages" not in spec
                for spec in raw["tool_stage_map"].values()
            )
        )

    def test_legacy_taxonomy_shapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "stage_taxonomy_v1",
                        "pruning_stages": {"a": {"definition": "legacy"}},
                        "stage_order": ["a"],
                        "tool_pruning_stage_map": {"tool": "a"},
                        "allowed_stage_transitions": {"a": ["a"]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported stage taxonomy fields"):
                load_stage_taxonomy(path)

    def test_cli_exposes_only_mainline_commands(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "molclaw_kg.cli", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        help_text = result.stdout
        for removed in [
            "doc-chunks",
            "score",
            "provenance",
            "audit",
            "eval-logs",
            "manifest",
            "legacy-views",
            "legacy-export",
            "migrate-kg",
        ]:
            self.assertNotIn(removed, help_text)
        self.assertIn("finalize", help_text)
        self.assertIn("sample-questions", help_text)


if __name__ == "__main__":
    unittest.main()
