from __future__ import annotations

ACTION_TOOL_CALL = "tool_call"
ACTION_FINAL_ANSWER = "final_answer"

ALLOWED_ACTION_TYPES = {ACTION_TOOL_CALL, ACTION_FINAL_ANSWER}

REQUIRED_TOOL_CALL_FIELDS = ("type", "tool_name", "arguments")
REQUIRED_FINAL_ANSWER_FIELDS = ("type", "answer")

REQUIRED_FINAL_ANSWER_SUBFIELDS = ("summary", "evidence", "result")

ACTION_FORMAT_DOC = (
    "Only output one JSON object per turn.\n"
    "Tool call: {\"type\":\"tool_call\",\"tool_name\":\"...\",\"arguments\":{...}}\n"
    "Final answer: {\"type\":\"final_answer\",\"answer\":{\"summary\":\"...\",\"evidence\":[],\"result\":{},\"ranked_molecules\":[]}}"
)
