from __future__ import annotations

from pathlib import Path
from typing import Any

from .claude_code_runtime import ClaudeCodeRuntime, extract_json_object
from ..settings import ProjectConfig


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
        p = self.config.paths.configs / "prompts" / "pairwise_adjudication_v1.md"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "You are MolClaw tool-graph pairwise adjudicator. Output strict JSON only."

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        prompt_override = payload.get("prompt_override")
        if isinstance(prompt_override, str) and prompt_override.strip():
            return prompt_override
        template = self._load_pairwise_template()
        return (
            f"{template.strip()}\n\n"
            "You are running inside one isolated workdir for exactly one directed pairwise adjudication task.\n"
            "Read local files in this directory first, then output strict JSON only.\n\n"
            "Required local files:\n"
            "- task_context.json\n"
            "- pair_spec.json\n"
            "- stage_taxonomy.json\n"
            "- edge_contract.json\n"
            "- source_manifest.json\n"
            "- output_schema.json\n"
        )

    def adjudicate(self, payload: dict[str, Any]) -> dict[str, Any]:
        pair_meta = payload.get("pair_meta") or {}
        pair_id = str(pair_meta.get("pair_id") or payload.get("pair_id") or "pair::unknown")
        prompt = self._build_prompt(payload)
        cc_workdir_name = str(payload.get("cc_workdir_name", pair_id)).strip() or pair_id
        cc_workdir = self.config.paths.run_dir / "cc_workdir" / cc_workdir_name

        run = self.runtime.run_prompt(
            prompt,
            run_label=f"pair_{pair_id}",
            add_dirs=[cc_workdir],
            allowed_tools=f"Read,Glob,mcp__{self.runtime.mcp_server_name}",
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
            "mcp_config_sha256": run.mcp_config_sha256,
            "mcp_server_name": run.mcp_server_name,
            "mcp_server_url": run.mcp_server_url,
            "workdir": run.workdir,
            "session_file": run.session_file,
            "skills_root": str(self.config.runtime.skills_root),
            "parsed_ok": "_parse_failure" not in parsed,
        }
        return parsed
