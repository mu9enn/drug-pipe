from __future__ import annotations


WORKFLOW_EDGE_SCHEMA = {
    "type": "object",
    "required": ["source_tool", "target_tool", "support_source", "support_ref"],
    "properties": {
        "source_tool": {"type": "string", "minLength": 1},
        "target_tool": {"type": "string", "minLength": 1},
        "support_source": {"type": "string", "enum": ["toolkg"]},
        "support_ref": {"type": "string", "minLength": 1},
        "source_output_slot": {"type": ["string", "null"]},
        "target_input_slot": {"type": ["string", "null"]},
    },
}


QUESTION_SAMPLER_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "status",
        "reject_reason",
        "public_question_text",
        "question_payload",
        "grounded_initial_inputs",
        "grounding_refs",
        "workflow_proposal",
        "edge_support_claims",
        "llm_message_intents",
        "tool_necessity",
        "scientific_task_rationale",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["success", "reject"]},
        "reject_reason": {"type": ["string", "null"]},
        "public_question_text": {"type": "string"},
        "question_payload": {"type": "object"},
        "grounded_initial_inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "value", "semantic_type", "format", "grounding_record_id"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "value": {},
                    "semantic_type": {"type": "string", "minLength": 1},
                    "format": {"type": "string", "minLength": 1},
                    "grounding_record_id": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
            },
        },
        "grounding_refs": {"type": "array", "items": {"type": "string"}},
        "workflow_proposal": {
            "type": "object",
            "required": ["tools", "edges", "final_deliverable"],
            "properties": {
                "tools": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "edges": {"type": "array", "items": WORKFLOW_EDGE_SCHEMA},
                "final_deliverable": {"type": "string", "minLength": 1},
            },
        },
        "edge_support_claims": {"type": "array", "items": WORKFLOW_EDGE_SCHEMA},
        "llm_message_intents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["role", "message_intent"],
                "properties": {
                    "role": {"type": "string", "enum": ["plan", "parameterize", "interpret", "route", "summarize", "repair"]},
                    "tool_id": {"type": ["string", "null"]},
                    "message_intent": {"type": "string", "minLength": 1},
                },
            },
        },
        "tool_necessity": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["tool_id", "necessary", "reason"],
                "properties": {
                    "tool_id": {"type": "string", "minLength": 1},
                    "necessary": {"type": "boolean"},
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        },
        "scientific_task_rationale": {"type": "string"},
    },
}


SIMPLE_QUESTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "public_question_text", "question_payload", "rationale"],
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["success", "reject"]},
        "public_question_text": {"type": "string"},
        "question_payload": {"type": "object"},
        "rationale": {"type": "string"},
    },
}
