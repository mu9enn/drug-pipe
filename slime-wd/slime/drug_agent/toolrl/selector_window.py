"""Lossy comparison copies only; never modify a training record."""
import copy
import hashlib
import json
import re

from drug_agent.toolrl.v8_dataset import stable_json

VERSION = "task_current_recent_interactions_v1"


def window_document(document, count_tokens, limit=32768, payload_chars=8192):
    doc = copy.deepcopy(document)
    history = doc["history_before_current_decision"]
    edits = []

    def payload(value, key, path, index):
        if isinstance(value, dict):
            return {k: payload(v, k, path + [k], index) for k, v in value.items()}
        if isinstance(value, list):
            return [payload(v, key, path + [i], index) for i, v in enumerate(value)]
        if not isinstance(value, str) or len(value) <= payload_chars:
            return value
        binary = key.lower() in {"base64", "file_base64", "image_base64", "base64_data", "b64_json"} or bool(re.match(r"^data:[^;,]+;base64,", value))
        body = key.lower() in {"file_content", "file_contents", "log", "logs", "stdout", "stderr"}
        if not binary and not body:
            return value
        edits.append({"message_index": index, "field": path, "chars": len(value),
                      "sha256": hashlib.sha256(value.encode()).hexdigest(),
                      "reason": "explicit_base64" if binary else "explicit_history_body_limit"})
        if binary:
            return "[Binary base64 payload omitted from comparison copy]"
        half = payload_chars // 2
        return value[:half] + "\n[Historical body omitted between these excerpts]\n" + value[-half:]

    for i, message in enumerate(history):
        if message.get("role") != "tool" or message.get("name") == "skill":
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except ValueError:
                # Unknown prose, code and skill text are not assumed to be noise.
                continue
            previous = len(edits)
            parsed = payload(parsed, "", [], i)
            if len(edits) != previous:
                message["content"] = stable_json(parsed)

    pinned = {i for i, m in enumerate(history) if m.get("role") in {"system", "user", "developer"}}
    units = []
    for i, message in enumerate(history):
        if i in pinned:
            continue
        if message.get("role") == "assistant":
            units.append([])
        if not units:
            raise ValueError("Historical tool observation has no assistant interaction")
        units[-1].append(i)
    for unit in units:
        calls = history[unit[0]].get("tool_calls", [])
        call_ids = [c["id"] for c in calls]
        result_ids = [history[i]["tool_call_id"] for i in unit[1:] if history[i].get("role") == "tool"]
        if sorted(call_ids) != sorted(result_ids):
            raise ValueError("Incomplete historical call/result unit")

    def render(start):
        indices = pinned | {i for unit in units[start:] for i in unit}
        doc["history_before_current_decision"] = [m for i, m in enumerate(history) if i in indices]
        return stable_json(doc)

    text = render(0)
    tokens = count_tokens(text)
    start = 0
    if tokens > limit:
        # Choose a complete contiguous suffix. No token slicing or skipped holes.
        lo, hi = 0, len(units)
        while lo < hi:
            mid = (lo + hi) // 2
            if count_tokens(render(mid)) <= limit:
                hi = mid
            else:
                lo = mid + 1
        start = lo
        text = render(start)
        tokens = count_tokens(text)
    reason = None
    if tokens > limit:
        reason = "fixed_task_current_definitions_over_limit"
    elif units and start == len(units):
        reason = "latest_complete_interaction_does_not_fit"
    omitted = [i for unit in units[:start] for i in unit]
    return text, tokens, {"version": VERSION, "protected_reason": reason,
                          "omitted_message_indices": omitted,
                          "retained_interactions": len(units) - start,
                          "payload_edits": edits, "lossy": bool(omitted or edits)}
