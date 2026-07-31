from __future__ import annotations

from typing import Any


def _score(sample: Any) -> float:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    trace = metadata.get("drug_agent_trace") if isinstance(metadata.get("drug_agent_trace"), dict) else {}
    return 1.0 if trace.get("done_reason") == "final_answer" else 0.0


async def reward(args, sample_or_samples, **kwargs):
    """Neutral online-eval reward; official benchmark metrics remain authoritative."""
    if isinstance(sample_or_samples, list):
        return [_score(sample) for sample in sample_or_samples]
    return _score(sample_or_samples)
