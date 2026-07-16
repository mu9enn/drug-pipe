from __future__ import annotations

import os
from pathlib import Path


def _default_group_space() -> Path:
    env = os.environ.get("GROUP_SPACE")
    if env:
        return Path(env)

    root_candidate = Path("/root/slime_sxy/group-space/sunxiangyu")
    home_candidate = Path("/home/sunxiangyu/slime_sxy/group-space/sunxiangyu")
    if root_candidate.exists():
        return root_candidate
    if home_candidate.exists():
        return home_candidate
    return root_candidate


GROUP_SPACE = _default_group_space()
WD = Path(os.environ.get("WD", str(GROUP_SPACE / "slime-wd")))
OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", str(WD / "outputs")))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("DATA", str(WD / "data"))))
PIPELINED_DATA = Path(os.environ.get("PIPELINED_DATA", str(DATA_ROOT / "pipelined_data")))

SLIME_DRUG_DATA_ROOT = Path(os.environ.get("DRUG_AGENT_DATA_ROOT", str(OUTPUTS_ROOT / "slime_drug_agent_data")))
SLIME_DRUG_RUNS_ROOT = Path(os.environ.get("DRUG_AGENT_RUNS_ROOT", str(OUTPUTS_ROOT / "slime_drug_agent_runs")))
GRPO_OUT_ROOT = SLIME_DRUG_DATA_ROOT / "grpo"
SFT_OUT_ROOT = SLIME_DRUG_DATA_ROOT / "sft"

SCHEMA_REPORT_DEFAULT = Path(os.environ.get("DRUG_AGENT_SCHEMA_REPORT", str(OUTPUTS_ROOT / "pipelined_data_schema_report.md")))
SKIPPED_REPORT_NAME = "skipped_report.jsonl"

SFT_OUTPUTS = PIPELINED_DATA / "sft_outputs"
SFT_OUTPUTS_ANSWER_HIT = PIPELINED_DATA / "sft_outputs_answer_hit"
USAGE_SUMMARY_CSV = PIPELINED_DATA / "molclaw_usage_summary.csv"

RAW_TASK_TYPES = ("ac", "pf", "vs")

DEFAULT_MAX_STEPS = 6
DEFAULT_RUN_NAME = "drug_agent_debug"

DEFAULT_SYSTEM_PROMPT = (
    "You are a drug discovery agent. "
    "You must output exactly one JSON object per turn, either a tool_call or final_answer. "
    "Do not output markdown code fences, XML, or natural language wrappers around JSON."
)
