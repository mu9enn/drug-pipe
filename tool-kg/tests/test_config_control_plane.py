from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from molclaw_kg.settings import build_config
from molclaw_kg.stage_taxonomy import load_stage_taxonomy


class ConfigControlPlaneTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
