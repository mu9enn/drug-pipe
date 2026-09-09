#!/usr/bin/env python3
"""Install the fixed structured-v8 MolClaw evaluation preset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PRESET_ID = "molclaw-v8-eval"
SKILL_PRESET_ID = PRESET_ID
DEFAULT_SYSTEM_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "data-pipe/pipeline/cleaning/prompts/qwen35_system.md"
)


def composition(system_prompt: str) -> str:
    indented = "\n".join(f"      {line}" for line in system_prompt.rstrip().splitlines())
    return f"""- id: persona
  name: '@deepseek-ai/dsh-persona'
  config:
    text: |-
{indented}
    complete: true
    includeRuntimeContext: false

- id: tool-bash
  name: '@deepseek-ai/dsh-tool-bash'
  disabled: !!js process.platform === 'win32'

- id: tool-pwsh
  name: '@deepseek-ai/dsh-tool-pwsh'
  disabled: !!js process.platform !== 'win32'

- id: tool-fs
  name: '@deepseek-ai/dsh-tool-fs'
  config:
    readImage: false

- id: tool-fs-search
  name: '@deepseek-ai/dsh-tool-fs-search'
  config:
    sampleOverCapGlobResults: false

- id: skill-filesystem
  name: '@deepseek-ai/dsh-skill-filesystem'

- id: tool-skill
  name: '@deepseek-ai/dsh-tool-skill'
"""


def install(
    dsh_repo: Path,
    dsh_home: Path,
    preset_id: str = PRESET_ID,
    system_prompt_file: Path = DEFAULT_SYSTEM_PROMPT,
) -> Path:
    if preset_id != PRESET_ID:
        raise ValueError(f"unknown MolClaw evaluation preset: {preset_id}")
    if not (dsh_repo / "apps/cli/config/agent-presets/standard/agent.cordis.yml").is_file():
        raise FileNotFoundError("DSH source checkout is incomplete")
    if not system_prompt_file.is_file():
        raise FileNotFoundError(system_prompt_file)
    target = dsh_home / ".agent-presets" / preset_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "agent.cordis.yml").write_text(
        composition(system_prompt_file.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (target / "preset.yml").write_text(
        "name: MolClaw Structured V8 Evaluation\n"
        "description: Fixed v8 system prompt with MolClaw, local filesystem, shell, and skill tools only.\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsh-repo", required=True, type=Path)
    parser.add_argument(
        "--dsh-home",
        type=Path,
        default=Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh"),
    )
    parser.add_argument("--preset-id", default=PRESET_ID)
    parser.add_argument("--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()
    print(install(
        args.dsh_repo.resolve(),
        args.dsh_home.expanduser().resolve(),
        args.preset_id,
        args.system_prompt_file.expanduser().resolve(),
    ))


if __name__ == "__main__":
    main()
