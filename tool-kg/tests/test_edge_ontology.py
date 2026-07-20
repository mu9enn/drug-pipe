from __future__ import annotations

from pathlib import Path
import unittest

import jsonschema

from molclaw_kg.edge_ontology import (
    build_adjudication_schema,
    load_edge_ontology,
    render_ontology_prompt,
    validate_adjudication_output,
)


ONTOLOGY_PATH = Path(__file__).parents[1] / "configs" / "edge_ontology_v1.yaml"


def adjudication(
    *,
    pair_id: str = "pair::a__to__b",
    relation_status: str = "valid",
    edge_type: str | None = "generates_partial_input_for",
    confidence: float = 0.8,
) -> dict:
    edge_types = []
    if edge_type is not None:
        edge_types.append(
            {
                "type": edge_type,
                "source_slot": "result",
                "target_slot_or_precondition": "input",
                "confidence": confidence,
                "evidence_ids": ["snapshot::a"],
            }
        )
    return {
        "pair_id": pair_id,
        "relation_status": relation_status,
        "direct_transition": relation_status == "valid",
        "edge_types": edge_types,
        "negative_reason": None,
        "context": "",
        "satisfied_mappings": [],
        "unsatisfied_required_inputs": [],
        "evidence_refs": ["snapshot::a"],
        "rationale": "Grounded decision.",
        "agent_confidence": confidence,
        "agent_model": "test",
    }


class EdgeOntologyContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = load_edge_ontology(ONTOLOGY_PATH)
        self.schema = build_adjudication_schema(self.ontology)

    def test_ontology_is_prompt_and_schema_edge_type_authority(self) -> None:
        edge_ids = set(self.ontology.edge_type_ids)
        schema_ids = set(
            self.schema["properties"]["edge_types"]["items"]["properties"]["type"]["enum"]
        )
        prompt = render_ontology_prompt(self.ontology)
        prompt_ids = {edge_id for edge_id in edge_ids if f"`{edge_id}`" in prompt}
        self.assertEqual(edge_ids, schema_ids)
        self.assertEqual(edge_ids, prompt_ids)

    def test_cross_field_and_pair_validation_rejects_without_repair(self) -> None:
        invalid = [
            adjudication(edge_type=None),
            adjudication(relation_status="negative", edge_type="generates_partial_input_for"),
            adjudication(relation_status="alternative", edge_type="generates_partial_input_for"),
            adjudication(confidence=-0.1),
            adjudication(confidence=1.1),
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(jsonschema.ValidationError):
                    validate_adjudication_output(
                        value,
                        self.schema,
                        expected_pair_id="pair::a__to__b",
                    )
        with self.assertRaisesRegex(jsonschema.ValidationError, "pair_id mismatch"):
            validate_adjudication_output(
                adjudication(pair_id="pair::wrong"),
                self.schema,
                expected_pair_id="pair::a__to__b",
            )

    def test_alternative_requires_alternative_edge_type(self) -> None:
        value = adjudication(relation_status="alternative", edge_type="alternative_to")
        value["direct_transition"] = False
        validate_adjudication_output(
            value,
            self.schema,
            expected_pair_id="pair::a__to__b",
        )


if __name__ == "__main__":
    unittest.main()
