from __future__ import annotations

from pathlib import Path
from typing import Any

from .claude_code_runtime import ClaudeCodeRuntime, extract_json_object
from ..settings import ProjectConfig
from ..workdir_skills import load_scene_prompt


def _parse_failure(pair_id: str, reason: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "_parse_failure": reason,
    }


class AgentCCAdjudicator:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model_name = "claude-cc-v1"
        self.last_trace: dict[str, Any] = {}
        self.runtime = ClaudeCodeRuntime(config)

    def _load_pairwise_template(self) -> str:
        return load_scene_prompt(self.config, "molclaw-tool-edge-adjudication")

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        prompt_override = payload.get("prompt_override")
        if isinstance(prompt_override, str) and prompt_override.strip():
            return prompt_override
        return "Adjudicate the directed MolClaw tool edge defined by the runtime JSON files in this workdir."

    def adjudicate(self, payload: dict[str, Any]) -> dict[str, Any]:
        pair_meta = payload.get("pair_meta") or {}
        pair_id = str(pair_meta.get("pair_id") or payload.get("pair_id") or "pair::unknown")
        prompt = self._build_prompt(payload)
        cc_workdir_name = str(payload.get("cc_workdir_name", pair_id)).strip() or pair_id
        cc_workdir = self.config.paths.run_dir / "cc_workdir" / cc_workdir_name

        run = self.runtime.run_prompt(
            prompt,
            run_label=f"pair_{pair_id}",
            system_prompt=self._load_pairwise_template(),
            add_dirs=[cc_workdir, Path(self.config.runtime.skills_root)],
            allowed_tools=f"Read,Glob,Skill,mcp__{self.runtime.mcp_server_name}",
            workdir=Path(cc_workdir),
        )
        parsed = extract_json_object(run.result_text) or extract_json_object(run.assistant_text) or extract_json_object(run.raw_stream)
        if not isinstance(parsed, dict):
            parsed = _parse_failure(pair_id, "agent_output_parse_failed_directional")

        self.last_trace = {
            "provider": run.provider,
            "provider_switch_ok": run.provider_switch_ok,
            "provider_switch_message": run.provider_switch_message,
            "command": run.command,
            "return_code": run.return_code,
            "timed_out": run.timed_out,
            "latency_sec": round(run.latency_sec, 6),
            "prompt_sha256": run.prompt_sha256,
            "system_prompt_sha256": run.system_prompt_sha256,
            "mcp_config_sha256": run.mcp_config_sha256,
            "mcp_server_name": run.mcp_server_name,
            "mcp_server_url": run.mcp_server_url,
            "workdir": run.workdir,
            "session_file": run.session_file,
            "attempt_session_files": run.attempt_session_files,
            "claude_attempts": run.claude_attempts,
            "selected_claude_attempt": run.selected_claude_attempt,
            "skills_root": str(self.config.runtime.skills_root),
            "parsed_ok": "_parse_failure" not in parsed,
        }
        return parsed
