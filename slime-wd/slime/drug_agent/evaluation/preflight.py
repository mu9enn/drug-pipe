from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from drug_agent.constants import DRUG_AGENT_L1_SKILLS_ROOT
from drug_agent.evaluation.molbench_adapter import build_molbench_dataset
from drug_agent.evaluation.official_eval import _require_pytdc, _require_rdkit
from drug_agent.evaluation.prompt_adapter import build_prompt_suite_dataset, build_single_prompt_dataset
from drug_agent.tools.runtime_env import (
    load_molclaw_environment,
    missing_molclaw_environment,
    redacted_environment_summary,
)
from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry, catalog_sha256
from drug_agent.utils import utc_now_iso, write_json


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _l1_snapshot_info(root: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "root": str(root),
        "sha256": _hash_tree(root),
        "skill_count": sum(1 for path in root.iterdir() if path.is_dir()),
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }
    snapshot_path = root.parents[2] / "L1_SNAPSHOT.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        expected = {
            "content_sha256": info["sha256"],
            "skill_count": info["skill_count"],
            "file_count": info["file_count"],
        }
        mismatches = {key: (snapshot.get(key), value) for key, value in expected.items() if snapshot.get(key) != value}
        if mismatches:
            raise ValueError(f"L1 snapshot manifest does not match shipped files: {mismatches}")
        info["manifest"] = str(snapshot_path)
    return info


def _checkpoint_info(root: Path) -> dict[str, object]:
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Not a Slime checkpoint (missing latest marker): {root}")
    raw = marker.read_text(encoding="utf-8").strip()
    try:
        iteration = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid checkpoint iteration in {marker}: {raw!r}") from exc
    iteration_dir = root / f"iter_{iteration:07d}"
    common = iteration_dir / "common.pt"
    if not common.is_file():
        raise FileNotFoundError(f"Checkpoint common.pt missing: {common}")
    return {"path": str(root), "iteration": iteration, "iteration_dir": str(iteration_dir)}


def _validate_model_assets(hf_checkpoint: Path, model_args_file: Path) -> dict[str, str]:
    if not model_args_file.is_file():
        raise FileNotFoundError(f"Model args file not found: {model_args_file}")
    required = [hf_checkpoint / "config.json"]
    tokenizer_candidates = [
        hf_checkpoint / "tokenizer.json",
        hf_checkpoint / "tokenizer.model",
        hf_checkpoint / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not any(path.is_file() for path in tokenizer_candidates):
        raise FileNotFoundError(
            f"HF tokenizer/config source is incomplete: missing={missing}, "
            f"tokenizer_candidates={[str(path) for path in tokenizer_candidates]}"
        )
    return {"hf_checkpoint": str(hf_checkpoint), "model_args_file": str(model_args_file)}


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a strict live MolClaw evaluation run")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--molbench-root")
    parser.add_argument("--molbench-suite", action="append", default=[])
    parser.add_argument("--molbench-limit-per-suite", type=int, default=0)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-suite-file")
    parser.add_argument("--task-type", default="e2e")
    parser.add_argument("--task-id", default="manual_prompt_001")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--task-timeout-sec", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-context-len", type=int, required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--model-args-file", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--tensor-model-parallel-size", type=int, required=True)
    parser.add_argument("--pipeline-model-parallel-size", type=int, required=True)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_steps < 0 or args.task_timeout_sec <= 0:
        raise ValueError(
            "max-workers and task-timeout-sec must be positive; "
            "max-steps must be non-negative (0 means unlimited)"
        )
    input_count = sum(bool(value) for value in (args.molbench_root, args.prompt_file, args.prompt_suite_file))
    if input_count != 1:
        raise ValueError("Provide exactly one of --molbench-root, --prompt-file, or --prompt-suite-file")

    load_molclaw_environment(args.env_file)
    missing = missing_molclaw_environment()
    if missing:
        raise RuntimeError(f"Missing MolClaw environment variable(s): {missing}")
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_info(Path(args.checkpoint).expanduser().resolve())
    model_assets = _validate_model_assets(
        Path(args.hf_checkpoint).expanduser().resolve(),
        Path(args.model_args_file).expanduser().resolve(),
    )
    if args.prompt_suite_file:
        evaluation_mode = "prompt_suite"
        input_manifest = build_prompt_suite_dataset(
            args.prompt_suite_file,
            run_dir,
            max_steps=args.max_steps,
        )
        input_manifest_path = run_dir / "prompt_manifest.json"
        input_counts = {"manual_prompt": input_manifest["sample_count"]}
    elif args.prompt_file:
        evaluation_mode = "single_prompt"
        input_manifest = build_single_prompt_dataset(
            args.prompt_file,
            run_dir,
            task_type=args.task_type,
            task_id=args.task_id,
            max_steps=args.max_steps,
        )
        input_manifest_path = run_dir / "prompt_manifest.json"
        input_counts = {"manual_prompt": input_manifest["sample_count"]}
    else:
        evaluation_mode = "molbench"
        _require_rdkit()
        if not args.molbench_suite or "molbench_mo" in args.molbench_suite:
            _require_pytdc()
        input_manifest = build_molbench_dataset(
            args.molbench_root,
            run_dir,
            selected_suites=args.molbench_suite,
            limit_per_suite=args.molbench_limit_per_suite,
        )
        input_manifest_path = run_dir / "benchmark_manifest.json"
        input_counts = input_manifest["counts"]
    if not DRUG_AGENT_L1_SKILLS_ROOT.is_dir():
        raise FileNotFoundError(f"L1 skills root not found: {DRUG_AGENT_L1_SKILLS_ROOT}")
    l1_snapshot = _l1_snapshot_info(DRUG_AGENT_L1_SKILLS_ROOT)

    executor = MCPToolExecutor(connect_on_init=False)
    try:
        registry = ToolRegistry(executor=executor, include_local_tools=True)
        catalog = registry.list_tools(force_refresh=True)
    finally:
        executor.close()
    mcp_catalog = [item for item in catalog if item.get("executor") != "local_sandbox"]
    if not mcp_catalog:
        raise RuntimeError("molclaw-scp list_tools returned no MCP tools")
    catalog_hash = catalog_sha256(catalog)
    write_json(run_dir / "tool_catalog.json", {"tools": catalog, "sha256": catalog_hash})

    repo_root = Path(__file__).resolve().parents[4]
    manifest = {
        "schema_version": "drug_agent_online_eval_run_v1",
        "created_at": utc_now_iso(),
        "code_commit": _git_commit(repo_root),
        "evaluation_mode": evaluation_mode,
        "checkpoint": checkpoint,
        "model_assets": model_assets,
        "input_manifest": str(input_manifest_path),
        "input_counts": input_counts,
        "tool_catalog_path": str(run_dir / "tool_catalog.json"),
        "tool_catalog_sha256": catalog_hash,
        "mcp_tool_count": len(mcp_catalog),
        "local_tool_count": len(catalog) - len(mcp_catalog),
        "l1_skills_root": str(DRUG_AGENT_L1_SKILLS_ROOT),
        "l1_skills_sha256": l1_snapshot["sha256"],
        "l1_skills_snapshot": l1_snapshot,
        "settings": {
            "max_workers": args.max_workers,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "task_timeout_sec": args.task_timeout_sec,
            "mcp_connect_timeout_sec": os.environ.get("MOLCLAW_CONNECT_TIMEOUT_SEC"),
            "mcp_list_tools_timeout_sec": os.environ.get("MOLCLAW_LIST_TOOLS_TIMEOUT_SEC"),
            "mcp_tool_timeout_sec": os.environ.get("MOLCLAW_TOOL_TIMEOUT_SEC"),
            "max_new_tokens": args.max_new_tokens,
            "max_context_len": args.max_context_len,
            "num_gpus": args.num_gpus,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "protocol": "canonical_react_xml",
            "molbench_suites": args.molbench_suite,
            "molbench_limit_per_suite": args.molbench_limit_per_suite,
        },
        "mcp_environment": redacted_environment_summary(),
    }
    if evaluation_mode == "molbench":
        manifest["benchmark_manifest"] = str(input_manifest_path)
        manifest["benchmark_counts"] = input_counts
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
