from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drug_agent.data.migrate_live_tool_catalog import migrate_records
from drug_agent.data.normalize_tool_catalog import normalize_catalog


class LiveToolMigrationTest(unittest.TestCase):
    def test_runtime_catalog_replaces_captured_local_tools_without_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "catalog.json"
            source.write_text(json.dumps({"tools": [
                {"name": "fix_pdb", "input_schema": {"type": "object"}},
                {"name": "Skill", "executor": "local_sandbox", "input_schema": {}},
            ]}) + "\n")
            normalized = normalize_catalog(source, root / "normalized.json")
            names = [tool["name"] for tool in normalized["tools"]]
            self.assertEqual(normalized["mcp_tool_count"], 1)
            self.assertEqual(normalized["local_tool_count"], 6)
            self.assertEqual(normalized["tool_count"], 7)
            self.assertNotIn("Skill", names)
            self.assertIn("Read", names)
            self.assertIn("Glob", names)

    def test_gene_structure_optional_organism_is_not_materialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "id": "react_kg_gene_default",
                "messages": [
                    {"role": "system", "content": "Use real MolClaw calls and supported local file/skill calls."},
                    {"role": "user", "content": "not an excluded prompt"},
                    {"role": "assistant", "content": (
                        '<tool_call>{"tool_name":"retrieve_protein_structure_by_gene_name",'
                        '"arguments":{"gene_name":"TP53"}}</tool_call>'
                    )},
                    {"role": "user", "content": (
                        '<observation tool_name="retrieve_protein_structure_by_gene_name">'
                        '{"status":"success"}</observation>'
                    )},
                ],
            }) + "\n")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "retrieve_protein_structure_by_gene_name",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "gene_name": {"type": "string"},
                        "organism": {"type": "string", "default": "9606"},
                        "sort_by": {"type": "string", "default": "length"},
                    },
                    "required": ["gene_name"],
                    "additionalProperties": False,
                },
            }]}) + "\n")

            report = migrate_records(source, catalog, root / "out")

            self.assertEqual(report["accepted_count"], 1)
            migrated = json.loads((root / "out/react_trajectories.jsonl").read_text())
            content = migrated["messages"][2]["content"]
            self.assertNotIn("organism", content)
            self.assertNotIn("sort_by", content)
            self.assertEqual(
                migrated["messages"][0]["content"],
                "Use real MolClaw calls and supported local file calls.",
            )

    def test_stale_required_defaults_are_not_synthesized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "id": "react_kg_gene_required",
                "messages": [
                    {"role": "user", "content": "not an excluded prompt"},
                    {"role": "assistant", "content": (
                        '<tool_call>{"tool_name":"retrieve_protein_structure_by_gene_name",'
                        '"arguments":{"gene_name":"TP53"}}</tool_call>'
                    )},
                    {"role": "user", "content": (
                        '<observation tool_name="retrieve_protein_structure_by_gene_name">'
                        '{"status":"success"}</observation>'
                    )},
                ],
            }) + "\n")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "retrieve_protein_structure_by_gene_name",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "gene_name": {"type": "string"},
                        "organism": {"type": "string"},
                        "sort_by": {"type": "string"},
                    },
                    "required": ["gene_name", "organism", "sort_by"],
                    "additionalProperties": False,
                },
            }]}) + "\n")

            report = migrate_records(source, catalog, root / "out")

            self.assertEqual(report["accepted_count"], 1)
            migrated = json.loads((root / "out/react_trajectories.jsonl").read_text())
            content = migrated["messages"][1]["content"]
            self.assertNotIn("organism", content)
            self.assertNotIn("sort_by", content)

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

    def test_schema_invalid_failed_call_is_preserved_for_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            failed_call = (
                '<thought>try it</thought>'
                '<tool_call>{"tool_name":"server_file_to_base64",'
                '"arguments":{"file_name":"protein.pdb","file_base64_string":"placeholder"}}</tool_call>'
            )
            source.write_text(json.dumps({
                "id": "react_ac_failed_then_replan",
                "messages": [
                    {"role": "user", "content": "not an excluded prompt"},
                    {"role": "assistant", "content": failed_call},
                    {"role": "user", "content": (
                        '<observation tool_name="server_file_to_base64">'
                        '{"tool_name":"server_file_to_base64","status":"error","is_error":true,'
                        '"content":"file_path is required"}</observation>'
                    )},
                    {"role": "assistant", "content": (
                        '<thought>use the required path</thought>'
                        '<tool_call>{"tool_name":"server_file_to_base64",'
                        '"arguments":{"file_path":"<artifact:structure/protein.pdb>"}}</tool_call>'
                    )},
                    {"role": "user", "content": (
                        '<observation tool_name="server_file_to_base64">'
                        '{"tool_name":"server_file_to_base64","status":"success","is_error":false}'
                        '</observation>'
                    )},
                ],
            }) + "\n")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "server_file_to_base64",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            }]}) + "\n")

            report = migrate_records(source, catalog, root / "out")

            self.assertEqual(report["accepted_count"], 1)
            migrated = json.loads((root / "out/react_trajectories.jsonl").read_text())
            self.assertIn('"file_base64_string":"placeholder"', migrated["messages"][1]["content"])
            audit = json.loads((root / "out/migration_audit.jsonl").read_text())
            self.assertIn(
                "schema_invalid_failed_call_retained",
                [change["kind"] for change in audit["changes"]],
            )

    def test_schema_invalid_successful_call_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "id": "react_ac_invalid_success",
                "messages": [
                    {"role": "user", "content": "not an excluded prompt"},
                    {"role": "assistant", "content": (
                        '<tool_call>{"tool_name":"server_file_to_base64",'
                        '"arguments":{"file_name":"protein.pdb"}}</tool_call>'
                    )},
                    {"role": "user", "content": (
                        '<observation tool_name="server_file_to_base64">'
                        '{"tool_name":"server_file_to_base64","status":"success","is_error":false}'
                        '</observation>'
                    )},
                ],
            }) + "\n")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "server_file_to_base64",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            }]}) + "\n")

            report = migrate_records(source, catalog, root / "out")

            self.assertEqual(report["accepted_count"], 0)
            self.assertEqual(report["rejected_count"], 1)
            rejected = json.loads((root / "out/migration_rejected.jsonl").read_text())
            self.assertTrue(rejected["reason"].startswith("arguments_not_valid_for_live_schema"))
