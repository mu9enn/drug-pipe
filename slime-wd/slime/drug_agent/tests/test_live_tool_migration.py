from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drug_agent.data.migrate_live_tool_catalog import migrate_records


class LiveToolMigrationTest(unittest.TestCase):
    def test_structured_name_argument_and_observation_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "id": "react_pf_test",
                "messages": [
                    {"role": "user", "content": "not an excluded prompt"},
                    {"role": "assistant", "content": '<thought>x</thought><tool_call>{"tool_name":"calculate_mol_topoiogy","arguments":{"smiles_list":["CCO"]}}</tool_call>'},
                    {"role": "user", "content": '<observation tool_name="calculate_mol_topoiogy">{"tool_name":"calculate_mol_topoiogy","status":"success"}</observation>'},
                ],
            }) + "\n")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "calculate_mol_topology",
                "input_schema": {"type": "object", "properties": {"smiles_list": {"type": "array", "items": {"type": "string"}}}, "required": ["smiles_list"], "additionalProperties": False},
            }]}) + "\n")
            report = migrate_records(source, catalog, root / "out")
            self.assertEqual(report["accepted_count"], 1)
            migrated = json.loads((root / "out/react_trajectories.jsonl").read_text())
            text = "\n".join(item["content"] for item in migrated["messages"])
            self.assertNotIn("calculate_mol_topoiogy", text)
            self.assertIn("calculate_mol_topology", text)
