from __future__ import annotations


def decide_final_status(
    *,
    execution_valid: bool,
    task_answer_valid: bool,
    training_trace_valid: bool,
) -> dict[str, object]:
    """Assign accepted/rejected from the three Python gates only."""
    rejection_reasons: list[str] = []
    if not execution_valid:
        rejection_reasons.append("execution_invalid")
    if not task_answer_valid:
        rejection_reasons.append("task_answer_invalid")
    if not training_trace_valid:
        rejection_reasons.append("training_trace_invalid")
    if rejection_reasons:
        return {
            "final_status": "rejected",
            "reasons": rejection_reasons,
            "authority": "final_acceptance_gate",
        }
    return {
        "final_status": "accepted",
        "reasons": [],
        "authority": "final_acceptance_gate",
    }
