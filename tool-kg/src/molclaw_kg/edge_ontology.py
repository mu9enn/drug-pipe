from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml


@dataclass(frozen=True)
class EdgeOntology:
    path: Path
    version: str
    relation_statuses: tuple[str, ...]
    edge_types: dict[str, dict[str, Any]]

    @property
    def edge_type_ids(self) -> tuple[str, ...]:
        return tuple(self.edge_types)


def default_edge_ontology_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "edge_ontology_v1.yaml"


def load_edge_ontology(path: Path) -> EdgeOntology:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    version = str(raw.get("version") or "").strip()
    statuses = raw.get("relation_statuses")
    edge_types = raw.get("edge_types")
    if not version:
        raise ValueError("edge ontology version is required")
    if not isinstance(statuses, list) or not statuses:
        raise ValueError("edge ontology relation_statuses must be a non-empty list")
    if not isinstance(edge_types, dict) or not edge_types:
        raise ValueError("edge ontology edge_types must be a non-empty mapping")
    allowed_statuses = {str(status) for status in statuses}
    normalized: dict[str, dict[str, Any]] = {}
    for edge_id, spec in edge_types.items():
        edge_name = str(edge_id).strip()
        if not edge_name or not isinstance(spec, dict):
            raise ValueError(f"invalid edge ontology entry: {edge_id!r}")
        allowed = [str(status) for status in spec.get("allowed_statuses") or []]
        if not allowed or any(status not in allowed_statuses for status in allowed):
            raise ValueError(f"invalid allowed_statuses for edge type {edge_name}")
        definition = str(spec.get("definition") or "").strip()
        if not definition:
            raise ValueError(f"missing definition for edge type {edge_name}")
        normalized[edge_name] = {
            **spec,
            "definition": definition,
            "allowed_statuses": allowed,
            "requires_slot_mapping": bool(spec.get("requires_slot_mapping", False)),
        }
    return EdgeOntology(
        path=path.resolve(),
        version=version,
        relation_statuses=tuple(str(status) for status in statuses),
        edge_types=normalized,
    )


def render_ontology_prompt(ontology: EdgeOntology) -> str:
    lines = [
        f"Ontology version: {ontology.version}",
        "Allowed relation statuses: " + ", ".join(ontology.relation_statuses),
        "Allowed edge types:",
    ]
    for edge_id, spec in ontology.edge_types.items():
        lines.append(
            f"- `{edge_id}`: {spec['definition']} "
            f"Allowed statuses: {', '.join(spec['allowed_statuses'])}. "
            f"Requires slot mapping: {'yes' if spec['requires_slot_mapping'] else 'no'}."
        )
    lines.append("Do not invent relation statuses or edge types.")
    return "\n".join(lines)


def build_adjudication_schema(ontology: EdgeOntology) -> dict[str, Any]:
    edge_type_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "source_slot", "target_slot_or_precondition", "confidence", "evidence_ids"],
        "properties": {
            "type": {"type": "string", "enum": list(ontology.edge_type_ids)},
            "source_slot": {"type": ["string", "null"]},
            "target_slot_or_precondition": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    satisfied_mapping_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_output_slot", "target_input_slot", "semantic_match", "format_match"],
        "properties": {
            "source_output_slot": {"type": "string"},
            "target_input_slot": {"type": "string"},
            "semantic_match": {
                "type": "string",
                "enum": ["exact", "compatible", "convertible", "incompatible", "unknown"],
            },
            "format_match": {
                "type": "string",
                "enum": ["exact", "compatible", "convertible", "incompatible", "unknown"],
            },
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        },
    }
    unsatisfied_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_input_slot", "reason"],
        "properties": {
            "target_input_slot": {"type": "string"},
            "reason": {"type": "string"},
            "can_be_user_provided": {"type": "boolean"},
            "can_be_satisfied_by_other_upstream_tool": {"type": "boolean"},
        },
    }
    status_rules: list[dict[str, Any]] = [
        {
            "if": {"properties": {"relation_status": {"const": "valid"}}},
            "then": {"properties": {"edge_types": {"minItems": 1}}},
        },
        {
            "if": {"properties": {"relation_status": {"enum": ["negative", "uncertain"]}}},
            "then": {"properties": {"edge_types": {"maxItems": 0}}},
        },
        {
            "if": {"properties": {"relation_status": {"const": "alternative"}}},
            "then": {
                "properties": {
                    "edge_types": {
                        "minItems": 1,
                        "items": {
                            "allOf": [
                                edge_type_item,
                                {"properties": {"type": {"const": "alternative_to"}}},
                            ]
                        },
                    }
                }
            },
        },
    ]
    for edge_id, spec in ontology.edge_types.items():
        status_rules.append(
            {
                "if": {
                    "properties": {
                        "edge_types": {
                            "contains": {
                                "type": "object",
                                "properties": {"type": {"const": edge_id}},
                                "required": ["type"],
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "relation_status": {"enum": list(spec["allowed_statuses"])}
                    }
                },
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "pair_id",
            "relation_status",
            "direct_transition",
            "edge_types",
            "negative_reason",
            "context",
            "satisfied_mappings",
            "unsatisfied_required_inputs",
            "evidence_refs",
            "rationale",
            "agent_confidence",
            "agent_model",
        ],
        "properties": {
            "pair_id": {"type": "string"},
            "relation_status": {"type": "string", "enum": list(ontology.relation_statuses)},
            "direct_transition": {"type": "boolean"},
            "edge_types": {"type": "array", "items": edge_type_item},
            "negative_reason": {"type": ["string", "null"]},
            "context": {"type": "string"},
            "satisfied_mappings": {"type": "array", "items": satisfied_mapping_item},
            "unsatisfied_required_inputs": {"type": "array", "items": unsatisfied_item},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
            "agent_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "agent_model": {"type": "string"},
        },
        "allOf": status_rules,
    }


def validate_adjudication_output(
    value: Any,
    schema: dict[str, Any],
    *,
    expected_pair_id: str,
) -> None:
    jsonschema.validate(value, schema)
    if not isinstance(value, dict) or value.get("pair_id") != expected_pair_id:
        actual = value.get("pair_id") if isinstance(value, dict) else None
        raise jsonschema.ValidationError(
            f"pair_id mismatch: expected={expected_pair_id!r}, actual={actual!r}"
        )
