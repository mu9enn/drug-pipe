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

DRUG_AGENT_DATA_ROOT = Path(os.environ.get("DRUG_AGENT_DATA_ROOT", str(OUTPUTS_ROOT / "slime_drug_agent_data")))
DRUG_AGENT_RUNS_ROOT = Path(os.environ.get("DRUG_AGENT_RUNS_ROOT", str(OUTPUTS_ROOT / "slime_drug_agent_runs")))
CANONICAL_REACT_DATA = DRUG_AGENT_DATA_ROOT / "react_trajectories.jsonl"
DRUG_AGENT_WORKSPACES_ROOT = Path(
    os.environ.get("DRUG_AGENT_WORKSPACES_ROOT", str(DRUG_AGENT_RUNS_ROOT / "online_workspaces"))
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DRUG_AGENT_L1_SKILLS_ROOT = Path(
    os.environ.get(
        "DRUG_AGENT_L1_SKILLS_ROOT",
        str(_REPO_ROOT / "data-pipe" / "skills" / "skills_full" / ".claude" / "skills" / "L1_tools"),
    )
)

DEFAULT_RUN_NAME = "drug_agent_debug"

DEFAULT_SYSTEM_PROMPT = (
    "You are a drug discovery agent. "
    "You must output exactly one JSON object per turn, either a tool_call or final_answer. "
    "Online tasks may provide MolClaw tools plus sandboxed Read, Write, Edit, Bash, Grep, Glob, "
    "and L1 Skill tools. Local file operations are confined to the task workspace, and L1 skills "
    "are read-only. "
    "Do not output markdown code fences, XML, or natural language wrappers around JSON."
)
