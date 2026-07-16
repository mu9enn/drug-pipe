from __future__ import annotations

DEFAULT_SERVER_URL = ""
DEFAULT_LOGS_ROOT = "results/used_molclaw_tool"

TRANSITION_EDGE_TYPES = {
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
}

RELATIONAL_EDGE_TYPES = {
    "alternative_to",
}
