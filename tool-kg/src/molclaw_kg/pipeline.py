from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit_sampler import sample_for_audit
from .candidate_generation import generate_candidates
from .canonical_edges import build_canonical_edges
from .doc_chunker import chunk_skills
from .evaluate_logs import evaluate_against_logs
from .exporters import export_artifacts
from .graph_views import build_graph_views
from .io_utils import write_json
from .manifest import write_repro_manifest
from .mcp_snapshot import run_snapshot
from .pairwise_runner import run_pairwise_adjudication
from .provenance import build_provenance_sidecar
from .settings import build_config
from .tool_card_builder import build_tool_cards


def run_all(
    project_root: Path,
    run_id: str,
    api_key: str,
    server_url: str | None = None,
    skills_root: str | None = None,
    logs_root: str | None = None,
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
        logs_root=logs_root,
        model_name=adjudication_mode,
    )

    status: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(config.paths.run_dir),
        "steps": {},
    }

    status["steps"]["snapshot"] = run_snapshot(config)
    status["steps"]["doc_chunks"] = chunk_skills(config)
    status["steps"]["tool_cards"] = build_tool_cards(config, max_workers=max_workers, resume=resume)
    status["steps"]["candidates"] = generate_candidates(config)
    status["steps"]["pairwise_adjudication"] = run_pairwise_adjudication(
        config,
        mode=adjudication_mode,
        max_workers=max_workers,
        resume=resume,
    )
    status["steps"]["canonical_edges"] = build_canonical_edges(config)
    status["steps"]["graph_views"] = build_graph_views(config)
    status["steps"]["provenance"] = build_provenance_sidecar(config)
    status["steps"]["export"] = export_artifacts(config)
    status["steps"]["audit_sample"] = sample_for_audit(config)
    status["steps"]["log_evaluation"] = evaluate_against_logs(config)
    status["steps"]["repro_manifest"] = write_repro_manifest(config)

    write_json(config.paths.run_dir / "pipeline_status.json", status)
    return status
