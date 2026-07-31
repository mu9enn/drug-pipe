from __future__ import annotations


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
