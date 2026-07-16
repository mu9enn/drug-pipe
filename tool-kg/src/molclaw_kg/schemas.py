from __future__ import annotations


_EDGE_TYPE_ITEM = {
    "type": "object",
    "required": ["type", "source_slot", "target_slot_or_precondition", "confidence", "evidence_ids"],
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "generates_full_input_for",
                "generates_partial_input_for",
                "preprocesses_for",
                "converts_format_for",
                "parameterizes_for",
                "filters_candidates_for",
                "ranks_or_scores_for",
                "validates_output_of",
                "refines_output_of",
                "reports_or_summarizes",
                "alternative_to",
            ],
        },
        "source_slot": {"type": ["string", "null"]},
        "target_slot_or_precondition": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_SATISFIED_MAPPING_ITEM = {
    "type": "object",
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

_UNSATISFIED_ITEM = {
    "type": "object",
    "required": ["target_input_slot", "reason"],
    "properties": {
        "target_input_slot": {"type": "string"},
        "reason": {"type": "string"},
        "can_be_user_provided": {"type": "boolean"},
        "can_be_satisfied_by_other_upstream_tool": {"type": "boolean"},
    },
}

ADJUDICATION_SCHEMA = {
    "type": "object",
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
        "relation_status": {
            "type": "string",
            "enum": ["valid", "negative", "uncertain", "alternative"],
        },
        "direct_transition": {"type": "boolean"},
        "edge_types": {
            "type": "array",
            "items": _EDGE_TYPE_ITEM,
        },
        "negative_reason": {"type": ["string", "null"]},
        "context": {"type": "string"},
        "satisfied_mappings": {"type": "array", "items": _SATISFIED_MAPPING_ITEM},
        "unsatisfied_required_inputs": {"type": "array", "items": _UNSATISFIED_ITEM},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "agent_confidence": {"type": "number"},
        "agent_model": {"type": "string"},
    },
}
