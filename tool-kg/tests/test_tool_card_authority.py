from __future__ import annotations

import unittest

from pydantic import ValidationError

from molclaw_kg.models import ToolAnnotationPatch
from molclaw_kg.tool_card_builder import (
    _base_tool_card,
    _default_input_requirement_sets,
    _merge_annotation_patch,
)


class ToolCardAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "tool_id": "example_tool",
            "title": "Example Tool",
            "description": "Example MCP tool",
            "inputSchema": {
                "type": "object",
                "required": ["novel_input"],
                "properties": {
                    "novel_input": {
                        "type": "string",
                        "description": "An uncommon scientific input.",
                        "default": "source-value",
                        "enum": ["source-value", "other-value"],
                    }
                },
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "result_file": {
                        "type": "string",
                        "description": "Result artifact.",
                    }
                },
            },
        }
        self.base = _base_tool_card(
            self.row,
            primary_stage="stage_a",
            scheduling_stages=["stage_a"],
        )

    def test_unknown_schema_slot_is_preserved_and_connectable(self) -> None:
        slot = next(slot for slot in self.base.inputs if slot.name == "novel_input")
        self.assertEqual(slot.semantic_type, "unknown")
        self.assertEqual(slot.format, "unknown")
        self.assertTrue(slot.required)
        self.assertEqual(slot.default, "source-value")
        self.assertEqual(slot.enum, ["source-value", "other-value"])
        self.assertIn("novel_input", [value.name for value in self.base.connectable_inputs])

    def test_agent_patch_cannot_modify_schema_facts_or_taxonomy_stage(self) -> None:
        with self.assertRaises(ValidationError):
            ToolAnnotationPatch.model_validate(
                {
                    "tool_id": "example_tool",
                    "primary_stage": "agent_stage",
                    "slot_annotations": {},
                }
            )
        patch = ToolAnnotationPatch.model_validate(
            {
                "tool_id": "example_tool",
                "description_summary": "Annotated description.",
                "aliases": ["example"],
                "slot_annotations": {
                    "input.novel_input": {
                        "semantic_type": "novel_scientific_input",
                        "format": "custom",
                        "parameter_kind": "data",
                        "connectable": True,
                        "evidence_refs": ["snapshot::example_tool"],
                        "confidence": 0.9,
                    }
                },
                "skill_derived_slots": [],
                "skill_derived_requirement_sets": [],
                "needs_review": False,
            }
        )
        merged = _merge_annotation_patch(
            self.base,
            patch,
            fixed_primary="stage_a",
            scheduling_stages=["stage_a"],
        )
        slot = next(value for value in merged.inputs if value.name == "novel_input")
        self.assertEqual(merged.primary_stage, "stage_a")
        self.assertEqual(merged.scheduling_stages, ["stage_a"])
        self.assertEqual(slot.raw_type, "string")
        self.assertTrue(slot.required)
        self.assertEqual(slot.default, "source-value")
        self.assertEqual(slot.enum, ["source-value", "other-value"])
        self.assertEqual(slot.semantic_type, "novel_scientific_input")

    def test_annotation_cannot_create_fake_schema_slot(self) -> None:
        patch = ToolAnnotationPatch.model_validate(
            {
                "tool_id": "example_tool",
                "slot_annotations": {
                    "input.invented": {
                        "semantic_type": "protein_structure",
                        "format": "pdb",
                        "parameter_kind": "data",
                        "connectable": True,
                        "evidence_refs": ["snapshot::example_tool"],
                    }
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown schema slots"):
            _merge_annotation_patch(
                self.base,
                patch,
                fixed_primary="stage_a",
                scheduling_stages=["stage_a"],
            )

    def test_json_schema_variants_preserve_base_requirements(self) -> None:
        schema = {
            "required": ["base_input"],
            "oneOf": [
                {"required": ["input_a"]},
                {"required": ["input_b"]},
            ],
        }
        sets = _default_input_requirement_sets(self.base.inputs, schema)
        self.assertEqual(len(sets), 2)
        self.assertEqual(sets[0]["condition"], "oneOf")
        self.assertEqual(sets[0]["required_slots"], ["base_input", "input_a"])
        self.assertEqual(sets[1]["required_slots"], ["base_input", "input_b"])

    def test_json_schema_if_then_is_kept_as_deterministic_requirement(self) -> None:
        schema = {
            "required": ["mode"],
            "if": {
                "properties": {
                    "mode": {"const": "from_file"},
                }
            },
            "then": {"required": ["input_file"]},
        }
        sets = _default_input_requirement_sets(self.base.inputs, schema)
        self.assertEqual([item["set_id"] for item in sets], ["schema_default", "schema_if_then"])
        self.assertEqual(sets[0]["required_slots"], ["mode"])
        self.assertEqual(sets[1]["required_slots"], ["mode", "input_file"])
        self.assertIn('"const": "from_file"', sets[1]["condition"])


if __name__ == "__main__":
    unittest.main()
