from __future__ import annotations

from pathlib import Path
from typing import Any

from .candidate_generation import generate_candidates
from .canonical_edges import build_canonical_edges
from .canonical_outputs import publish_canonical_outputs
from .io_utils import write_json
from .mcp_snapshot import run_snapshot
from .pairwise_runner import run_pairwise_adjudication
from .settings import build_config
from .tool_card_builder import build_tool_cards


def run_all(
    project_root: Path,
    run_id: str,
    api_key: str,
    server_url: str | None = None,
    skills_root: str | None = None,
    adjudication_mode: str = "claude_cc",
    max_workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    config = build_config(
        project_root=project_root,
        run_id=run_id,
        server_url=server_url,
        api_key=api_key,
        skills_root=skills_root,
        model_name=adjudication_mode,
    )

    status: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(config.paths.run_dir),
        "steps": {},
    }

    status["steps"]["snapshot"] = run_snapshot(config)
    status["steps"]["tool_cards"] = build_tool_cards(config, max_workers=max_workers, resume=resume)
    status["steps"]["candidates"] = generate_candidates(config)
    status["steps"]["pairwise_adjudication"] = run_pairwise_adjudication(
        config,
        mode=adjudication_mode,
        max_workers=max_workers,
        resume=resume,
    )
    status["steps"]["canonical_edges"] = build_canonical_edges(config)
    status["steps"]["canonical_outputs"] = publish_canonical_outputs(config)

    write_json(config.paths.run_dir / "pipeline_status.json", status)
    return status
