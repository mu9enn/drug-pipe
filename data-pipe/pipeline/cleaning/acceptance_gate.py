from __future__ import annotations

from typing import Any


def decide_final_status(
    *,
    execution_valid: bool,
    task_answer_valid: bool,
    training_trace_valid: bool,
    llm_clean_status: str,
    llm_clean_findings: list[str],
    hard_clean_findings: list[str],
    llm_clean_required: bool = False,
) -> dict[str, Any]:
    """The sole authority for accepted/rejected/quarantine."""
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

    quarantine_reasons: list[str] = []
    if llm_clean_status in {"failed", "unsafe_rewrite"}:
        quarantine_reasons.append(f"llm_clean_{llm_clean_status}")
    if llm_clean_required and llm_clean_status != "cleaned":
        quarantine_reasons.append("llm_clean_required_but_not_completed")
    quarantine_reasons.extend(f"llm_clean:{finding}" for finding in llm_clean_findings)
    quarantine_reasons.extend(f"hard_clean:{finding}" for finding in hard_clean_findings)
    if quarantine_reasons:
        return {
            "final_status": "quarantine",
            "reasons": list(dict.fromkeys(quarantine_reasons)),
            "authority": "final_acceptance_gate",
        }
    return {
        "final_status": "accepted",
        "reasons": [],
        "authority": "final_acceptance_gate",
    }
