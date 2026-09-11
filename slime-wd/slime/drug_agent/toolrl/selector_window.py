"""Lossy comparison copies only; never modify a training record."""
import copy

from drug_agent.toolrl.v8_dataset import stable_json

VERSION = "task_current_recent_assistants_no_results_v2"


def window_document(document, count_tokens, limit=32768):
    doc = copy.deepcopy(document)
    history = doc["history_before_current_decision"]
    # Intake verifies and saves the common system text separately. Tool results,
    # including skill results, never enter the comparison text.
    excluded = {i for i, m in enumerate(history) if m.get("role") in {"system", "tool"}}
    pinned = {i for i, m in enumerate(history) if m.get("role") in {"user", "developer"}}
    units = []
    for i, message in enumerate(history):
        if i in pinned or i in excluded:
            continue
        if message.get("role") != "assistant":
            raise ValueError("Unsupported history role")
        units.append([i])

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
        reason = "latest_complete_assistant_does_not_fit"
    omitted = [i for unit in units[:start] for i in unit]
    return text, tokens, {"version": VERSION, "protected_reason": reason,
                          "omitted_message_indices": omitted,
                          "excluded_system_message_indices": sorted(i for i in excluded if history[i]["role"] == "system"),
                          "excluded_tool_result_indices": sorted(i for i in excluded if history[i]["role"] == "tool"),
                          "retained_assistant_messages": len(units) - start,
                          "lossy": bool(omitted or excluded)}
