from __future__ import annotations

import re
import shutil
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonschema
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from .adjudicators.agent_cc import AgentCCAdjudicator
from .edge_ontology import (
    EdgeOntology,
    build_adjudication_schema,
    load_edge_ontology,
    render_ontology_prompt,
    validate_adjudication_output,
)
from .io_utils import append_jsonl, atomic_write_jsonl, read_jsonl, stable_hash_obj, write_json, write_jsonl
from .settings import ProjectConfig
from .runtime_state import latest_jsonl_by_key, next_attempt_dir
from .stage_taxonomy import load_stage_taxonomy, resolve_stage_taxonomy_path


def _select_adjudicator(config: ProjectConfig, mode: str):
    if mode != "claude_cc":
        raise ValueError(f"unsupported adjudication mode: {mode}")
    return AgentCCAdjudicator(config)


def _failure_response(pair_id: str, reason: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "_parse_failure": reason,
    }


def _safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return s[:120] if s else "pair"


def _load_pair_prompt_template(config: ProjectConfig) -> str:
    p = config.paths.configs / "prompts" / "pairwise_adjudication_v1.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "You are MolClaw tool-graph pairwise adjudicator. Output strict JSON only."


def _build_pair_prompt(template: str, ontology: EdgeOntology) -> str:
    rendered = template.replace("{{EDGE_ONTOLOGY}}", render_ontology_prompt(ontology))
    return (
        f"{rendered.strip()}\n\n"
        "You are running inside one isolated workdir for exactly one directed pairwise adjudication task.\n"
        "Read local files in this directory first, then output strict JSON for this direction only.\n\n"
        "Required local files:\n"
        "- task_context.json\n"
        "- pair_spec.json\n"
        "- stage_taxonomy.json\n"
        "- edge_ontology.json\n"
        "- source_manifest.json\n"
        "- output_schema.json\n"
    )


def _candidate_source_globs(tool_id: str, aliases: list[str]) -> list[str]:
    tokens = [tool_id, tool_id.replace("_", "-")] + aliases[:4]
    out = []
    for tok in tokens:
        t = str(tok).strip()
        if not t:
            continue
        out.append(f".claude/skills/L1_tools/**/*{t}*")
        out.append(f".claude/skills/L2_workflows/**/*{t}*")
    out.append(".claude/skills/L3_methodology/**/*")
    dedup = []
    for x in out:
        if x not in dedup:
            dedup.append(x)
    return dedup


def _build_pair_spec(*, row: dict[str, Any], cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = str(row["source_tool"])
    target = str(row["target_tool"])
    source_card = cards.get(source, {})
    target_card = cards.get(target, {})
    source_aliases = [str(x).strip() for x in (source_card.get("aliases") or []) if str(x).strip()]
    target_aliases = [str(x).strip() for x in (target_card.get("aliases") or []) if str(x).strip()]

    return {
        "pair_id": row["pair_id"],
        "source": {
            "tool_id": source,
            "primary_stage": source_card.get("primary_stage"),
            "aliases": source_aliases,
            "tool_card_file": "source_tool_card.json",
            "candidate_skill_globs": _candidate_source_globs(source, source_aliases),
        },
        "target": {
            "tool_id": target,
            "primary_stage": target_card.get("primary_stage"),
            "aliases": target_aliases,
            "tool_card_file": "target_tool_card.json",
            "candidate_skill_globs": _candidate_source_globs(target, target_aliases),
        },
        "pair_meta": {
            "pair_id": row["pair_id"],
            "source_tool": source,
            "target_tool": target,
            "source_stage": row.get("source_stage"),
            "target_stage": row.get("target_stage"),
            "proposal_reasons": row.get("proposal_reasons") or [],
            "recall_risk": row.get("recall_risk"),
        },
        "adjudication_goal": "Judge direct typed edge for this directed pair and record coverage/missing requirements.",
        "must_not_do": [
            "Do not decide only from generated summaries.",
            "Do not reject a direction only because some other required target inputs are missing.",
            "Do not cite pair_spec/task_context as primary evidence.",
        ],
    }


def _build_source_manifest(pair_spec: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for side in ["source", "target"]:
        tool = pair_spec[side]
        for glob in tool.get("candidate_skill_globs", []):
            entries.append(
                {
                    "path_glob": glob,
                    "reason": f"{side}_tool_candidate_skill",
                }
            )
    entries.append(
        {
            "path_glob": ".claude/skills/L2_workflows/**/*",
            "reason": "workflow-level evidence",
        }
    )
    entries.append(
        {
            "path_glob": ".claude/skills/L3_methodology/**/*",
            "reason": "methodology constraints",
        }
    )
    return {"candidate_sources": entries}


def _prepare_pair_workdir(
    *,
    config: ProjectConfig,
    workdir: Path,
    pair_spec: dict[str, Any],
    source_manifest: dict[str, Any],
    prompt: str,
    source_card: dict[str, Any],
    target_card: dict[str, Any],
    taxonomy_raw: dict[str, Any],
    ontology: EdgeOntology,
    adjudication_schema: dict[str, Any],
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)

    for item in config.runtime.skills_root.iterdir():
        dst = workdir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    write_json(workdir / "pair_spec.json", pair_spec)
    write_json(workdir / "source_manifest.json", source_manifest)
    write_json(workdir / "source_tool_card.json", source_card)
    write_json(workdir / "target_tool_card.json", target_card)
    write_json(workdir / "stage_taxonomy.json", taxonomy_raw)
    write_json(
        workdir / "edge_ontology.json",
        {
            "version": ontology.version,
            "relation_statuses": list(ontology.relation_statuses),
            "edge_types": ontology.edge_types,
        },
    )
    write_json(workdir / "output_schema.json", adjudication_schema)
    write_json(
        workdir / "task_context.json",
        {
            "task_type": "pairwise_edge_adjudication_directional",
            "pair_id": pair_spec.get("pair_id"),
            "pair_spec_file": "pair_spec.json",
            "taxonomy_file": "stage_taxonomy.json",
            "edge_ontology_file": "edge_ontology.json",
            "source_manifest_file": "source_manifest.json",
            "output_schema_file": "output_schema.json",
            "canonical_skill_root": ".claude/skills",
            "tool_card_files": ["source_tool_card.json", "target_tool_card.json"],
        },
    )
    (workdir / "prompt.txt").write_text(prompt, encoding="utf-8")


def _pair_unit_dir(config: ProjectConfig, source_tool: str, target_tool: str) -> Path:
    return config.paths.run_dir / "cc_workdir" / _safe_name(f"{source_tool}__to__{target_tool}")


def _pair_attempt_rel_name(config: ProjectConfig, source_tool: str, target_tool: str, attempt_dir: Path) -> str:
    base = config.paths.run_dir / "cc_workdir"
    return str(attempt_dir.relative_to(base))


def _flatten_adjudication_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": rec["pair_id"],
        "source_tool": rec["source_tool"],
        "target_tool": rec["target_tool"],
        "source_stage": rec.get("source_stage"),
        "target_stage": rec.get("target_stage"),
        **rec["response"],
        "response_schema_ok": rec["response_schema_ok"],
        "response_schema_error": rec["response_schema_error"],
        "cache_key": rec["cache_key"],
        "from_cache": rec["from_cache"],
        "trace": rec["trace"],
        "created_at_utc": rec["created_at_utc"],
    }


def _flatten_cache_entry_to_record(
    *,
    pair: dict[str, Any],
    cache_key: str,
    cache_entry: dict[str, Any],
    from_cache: bool,
) -> dict[str, Any]:
    response = cache_entry.get("response")
    if not isinstance(response, dict):
        response = _failure_response(str(pair["pair_id"]), "cache_missing_response")
    trace = cache_entry.get("trace") if isinstance(cache_entry.get("trace"), dict) else {}
    schema_ok = bool(trace.get("schema_ok", True))
    schema_error = trace.get("schema_error")
    rec = {
        "pair_id": str(pair["pair_id"]),
        "source_tool": str(pair["source_tool"]),
        "target_tool": str(pair["target_tool"]),
        "source_stage": pair.get("source_stage"),
        "target_stage": pair.get("target_stage"),
        "response": dict(response),
        "response_schema_ok": schema_ok,
        "response_schema_error": schema_error,
        "cache_key": cache_key,
        "from_cache": from_cache,
        "created_at_utc": cache_entry.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
        "trace": trace,
    }
    return rec


def _run_pairwise_attempt(
    *,
    config: ProjectConfig,
    pair: dict[str, Any],
    cards: dict[str, dict[str, Any]],
    prompt_template: str,
    taxonomy_version: Any,
    cache_path: Path,
    cache_lock: threading.Lock,
    rerun_round: int,
    taxonomy_raw: dict[str, Any],
    ontology: EdgeOntology,
    adjudication_schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_tool = str(pair["source_tool"])
    target_tool = str(pair["target_tool"])
    pair_id = str(pair["pair_id"])
    pair_spec = None
    source_manifest = None
    prompt = ""
    attempt_dir: Path | None = None
    workdir_str = str(config.paths.run_dir / "cc_workdir" / _safe_name(f"{source_tool}__to__{target_tool}"))
    session_file_str = str(Path(workdir_str) / "complete_session.jsonl")
    status = "ok"
    err_msg = None
    schema_ok = False
    schema_error = None
    response_obj: dict[str, Any] = _failure_response(pair_id, "worker_exception")
    trace: dict[str, Any] = {}
    cache_key = ""
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        pair_spec = _build_pair_spec(row=pair, cards=cards)
        source_manifest = _build_source_manifest(pair_spec)
        prompt = _build_pair_prompt(prompt_template, ontology)
        unit_dir = _pair_unit_dir(config, source_tool, target_tool)
        attempt_dir = next_attempt_dir(unit_dir)
        attempt_rel = _pair_attempt_rel_name(config, source_tool, target_tool, attempt_dir)
        adjudicator = AgentCCAdjudicator(config)
        _prepare_pair_workdir(
            config=config,
            workdir=attempt_dir,
            pair_spec=pair_spec,
            source_manifest=source_manifest,
            prompt=prompt,
            source_card=cards.get(source_tool, {}),
            target_card=cards.get(target_tool, {}),
            taxonomy_raw=taxonomy_raw,
            ontology=ontology,
            adjudication_schema=adjudication_schema,
        )

        payload = {
            "pair_id": pair_id,
            "pair_meta": pair_spec["pair_meta"],
            "cc_workdir_name": attempt_rel,
            "template_version": "pairwise_adjudication_agent_v3_directional",
            "prompt_override": prompt,
        }

        cache_key = stable_hash_obj(
            {
                "model": adjudicator.model_name,
                "pair_id": pair_id,
                "pair_meta": pair_spec["pair_meta"],
                "prompt": prompt,
                "source_tool_card": cards.get(source_tool, {}),
                "target_tool_card": cards.get(target_tool, {}),
                "taxonomy_version": taxonomy_version,
                "edge_ontology_version": ontology.version,
            }
        )

        raw = adjudicator.adjudicate(payload)
        trace = dict(getattr(adjudicator, "last_trace", {}) or {})

        schema_ok = True
        try:
            validate_adjudication_output(
                raw,
                adjudication_schema,
                expected_pair_id=pair_id,
            )
        except jsonschema.ValidationError as e:
            schema_ok = False
            schema_error = str(e)

        response_obj = dict(raw) if isinstance(raw, dict) else _failure_response(pair_id, "agent_output_not_object")
    except Exception as e:  # pragma: no cover - worker safety net
        status = "worker_exception"
        err_msg = f"{type(e).__name__}: {e}"
        response_obj = _failure_response(pair_id, "agent_output_parse_failed_directional")
        trace = {
            "provider": "worker_exception",
            "provider_switch_ok": False,
            "provider_switch_message": err_msg,
            "command": "",
            "return_code": 1,
            "timed_out": False,
            "latency_sec": 0.0,
            "prompt_sha256": "",
            "mcp_config_sha256": "",
            "mcp_server_name": "",
            "mcp_server_url": "",
            "workdir": workdir_str,
            "session_file": session_file_str,
            "skills_root": str(config.runtime.skills_root),
            "parsed_ok": False,
            "schema_ok": False,
            "schema_error": err_msg,
        }
        if attempt_dir is None:
            attempt_dir = Path(workdir_str)
    finally:
        if not cache_key:
            cache_key = stable_hash_obj(
                {
                    "model": "claude-cc-v1",
                    "pair_id": pair_id,
                    "pair_meta": pair.get("pair_meta", {}),
                    "prompt": prompt,
                    "source_tool_card": cards.get(source_tool, {}),
                    "target_tool_card": cards.get(target_tool, {}),
                    "taxonomy_version": taxonomy_version,
                    "edge_ontology_version": ontology.version,
                }
            )

    rec = {
        "pair_id": pair_id,
        "source_tool": source_tool,
        "target_tool": target_tool,
        "source_stage": pair.get("source_stage"),
        "target_stage": pair.get("target_stage"),
        "response": response_obj,
        "response_schema_ok": schema_ok,
        "response_schema_error": schema_error,
        "cache_key": cache_key,
        "from_cache": False,
        "created_at_utc": created_at,
        "trace": {
            "provider": trace.get("provider"),
            "provider_switch_ok": trace.get("provider_switch_ok"),
            "provider_switch_message": trace.get("provider_switch_message"),
            "command": trace.get("command"),
            "return_code": trace.get("return_code"),
            "timed_out": trace.get("timed_out"),
            "latency_sec": trace.get("latency_sec"),
            "prompt_sha256": trace.get("prompt_sha256"),
            "mcp_config_sha256": trace.get("mcp_config_sha256"),
            "mcp_server_name": trace.get("mcp_server_name"),
            "mcp_server_url": trace.get("mcp_server_url"),
            "workdir": trace.get("workdir"),
            "session_file": trace.get("session_file"),
            "skills_root": trace.get("skills_root"),
            "parsed_ok": trace.get("parsed_ok"),
            "schema_ok": schema_ok,
            "schema_error": schema_error,
        },
    }

    cache_entry = {
        "cache_key": cache_key,
        "response": response_obj,
        "trace": rec["trace"],
        "created_at_utc": rec["created_at_utc"],
    }
    try:
        append_jsonl(cache_path, cache_entry, lock=cache_lock)
    except Exception as e:  # pragma: no cover - best effort cache persistence
        err = f"{type(e).__name__}: {e}"
        rec["trace"]["cache_write_error"] = err
        cache_entry.setdefault("trace", {})["cache_write_error"] = err
    return rec, cache_entry


def _load_pair_ids_filter(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"pair ids file not found: {path}")
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    if not out:
        raise ValueError(f"pair ids file is empty after filtering comments/blanks: {path}")
    return out


def _merge_adjudications_with_existing(*, out_path: Path, updated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = read_jsonl(out_path) if out_path.exists() else []
    existing_map = {str(x.get("pair_id")): x for x in existing if x.get("pair_id")}
    for row in updated_rows:
        existing_map[str(row["pair_id"])] = row
    return [existing_map[k] for k in sorted(existing_map.keys())]


def _extract_pair_adjudication_alerts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    alerts: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    for r in rows:
        schema_ok = bool(r.get("response_schema_ok", False))
        failure_reason = str(r.get("_parse_failure") or "")
        if (not schema_ok) or failure_reason:
            rec = {
                "pair_id": r.get("pair_id"),
                "source_tool": r.get("source_tool"),
                "target_tool": r.get("target_tool"),
                "response_schema_ok": schema_ok,
                "failure_reason": failure_reason or r.get("response_schema_error"),
                "created_at_utc": r.get("created_at_utc"),
            }
            alerts.append(rec)
            pid = str(r.get("pair_id") or "").strip()
            if pid:
                pair_ids.append(pid)
    dedup_pair_ids = sorted(set(pair_ids))
    return alerts, dedup_pair_ids


def run_pairwise_adjudication(
    config: ProjectConfig,
    mode: str = "claude_cc",
    pair_ids_file: Path | None = None,
    merge_into_existing: bool = False,
    bypass_cache_for_targets: bool = False,
    max_workers: int = 1,
    resume: bool = False,
    rerun_round: int = 0,
) -> dict[str, Any]:
    cards = {x["tool_id"]: x for x in read_jsonl(config.paths.run_dir / "tool_cards.jsonl")}
    pairs_all = read_jsonl(config.paths.run_dir / "candidate_pairs.jsonl")
    out_path = config.paths.run_dir / "pair_adjudications.jsonl"
    requested_pair_ids: set[str] | None = None
    if pair_ids_file is not None:
        requested_pair_ids = _load_pair_ids_filter(pair_ids_file)
        missing = sorted(requested_pair_ids - {str(p.get("pair_id")) for p in pairs_all})
        if missing:
            raise ValueError(f"requested pair_ids missing in candidate_pairs.jsonl: {missing[:50]}")
        pairs = [p for p in pairs_all if str(p.get("pair_id")) in requested_pair_ids]
    else:
        pairs = pairs_all

    adjudicator = _select_adjudicator(config, mode)
    cache_path = config.paths.run_dir / "pairwise_cache.jsonl"
    cache_rows = read_jsonl(cache_path)
    cache = {x.get("cache_key"): x for x in cache_rows if x.get("cache_key")}
    reuse_existing = bool(resume or merge_into_existing)
    final_map: dict[str, dict[str, Any]] = latest_jsonl_by_key(out_path, "pair_id") if reuse_existing else {}

    taxonomy = load_stage_taxonomy(resolve_stage_taxonomy_path(config.paths.root))
    ontology = load_edge_ontology(config.paths.configs / "edge_ontology_v1.yaml")
    adjudication_schema = build_adjudication_schema(ontology)
    pruned: list[dict[str, Any]] = []
    stage_kept = list(pairs)

    write_jsonl(config.paths.run_dir / "pair_pruned_by_stage.jsonl", pruned)
    write_json(
        config.paths.run_dir / "pair_pruned_by_stage_meta.json",
        {
            "stage_pruning_mode": "disabled_semantic_authority",
            "input_pair_count": len(pairs),
            "pruned_pair_count": 0,
            "kept_pair_count": len(stage_kept),
            "deterministic_alternative_count": 0,
            "note": "Candidate scheduling already applied taxonomy; edge semantics are decided only by Claude adjudication.",
        },
    )

    prompt_template = _load_pair_prompt_template(config)
    target_pair_ids = requested_pair_ids or set()
    cache_lock = threading.Lock()
    scheduled_pairs: list[dict[str, Any]] = []
    cache_hit_rows: list[dict[str, Any]] = []
    skipped_existing = 0
    cache_hit_count = 0

    for p in tqdm(stage_kept, desc="adjudicate-pairs", unit="pair"):
        pair_id = str(p["pair_id"])
        source_tool = str(p["source_tool"])
        target_tool = str(p["target_tool"])
        pair_spec = _build_pair_spec(row=p, cards=cards)
        prompt = _build_pair_prompt(prompt_template, ontology)
        cache_key = stable_hash_obj(
            {
                "model": adjudicator.model_name,
                "pair_id": pair_id,
                "pair_meta": pair_spec["pair_meta"],
                "prompt": prompt,
                "source_tool_card": cards.get(source_tool, {}),
                "target_tool_card": cards.get(target_tool, {}),
                "taxonomy_version": taxonomy.version,
                "edge_ontology_version": ontology.version,
            }
        )
        bypass_cache = bool(bypass_cache_for_targets and target_pair_ids and pair_id in target_pair_ids)

        if (not bypass_cache) and (cache_key in cache):
            rec = _flatten_cache_entry_to_record(pair=p, cache_key=cache_key, cache_entry=cache[cache_key], from_cache=True)
            cache_hit_rows.append(rec)
            final_map[pair_id] = _flatten_adjudication_record(rec)
            cache_hit_count += 1
            continue

        if resume and (pair_id in final_map) and (not bypass_cache):
            skipped_existing += 1
            continue

        scheduled_pairs.append({**p, "_cache_key": cache_key, "_bypass_cache": bypass_cache})

    actual_calls = 0
    progress_rows: list[dict[str, Any]] = []
    if scheduled_pairs:
        max_workers = max(1, int(max_workers or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_pairwise_attempt,
                    config=config,
                    pair=pair,
                    cards=cards,
                    prompt_template=prompt_template,
                    taxonomy_version=taxonomy.version,
                    cache_path=cache_path,
                    cache_lock=cache_lock,
                    rerun_round=rerun_round,
                    taxonomy_raw=taxonomy.raw,
                    ontology=ontology,
                    adjudication_schema=adjudication_schema,
                )
                for pair in scheduled_pairs
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="adjudicate-pairs", unit="pair"):
                rec, cache_entry = fut.result()
                actual_calls += 1
                pair_id = rec["pair_id"]
                cache[rec["cache_key"]] = cache_entry
                final_map[pair_id] = _flatten_adjudication_record(rec)
                progress_rows.append(rec)

    # Preserve cache hits, progress rows, and any existing rows for untouched pairs.
    for rec in cache_hit_rows:
        final_map[rec["pair_id"]] = _flatten_adjudication_record(rec)
    for rec in progress_rows:
        final_map[rec["pair_id"]] = _flatten_adjudication_record(rec)

    final_rows = [final_map[k] for k in sorted(final_map.keys())]
    atomic_write_jsonl(out_path, final_rows)

    alerts, alert_pair_ids = _extract_pair_adjudication_alerts(final_rows)
    alerts_path = config.paths.run_dir / "pair_adjudication_alerts.jsonl"
    atomic_write_jsonl(alerts_path, alerts)
    rerun_targets_path = config.paths.run_dir / "pair_adjudication_rerun_targets.txt"
    rerun_targets_path.write_text("".join(f"{x}\n" for x in alert_pair_ids), encoding="utf-8")
    by_reason: dict[str, int] = defaultdict(int)
    for a in alerts:
        reason = str(a.get("failure_reason") or ("schema_not_ok" if not a.get("response_schema_ok", True) else "other"))
        by_reason[reason] += 1
    write_json(
        config.paths.run_dir / "pair_adjudication_alerts_meta.json",
        {
            "alert_count": len(alerts),
            "alert_pair_count": len(alert_pair_ids),
            "status_breakdown": dict(sorted(by_reason.items())),
            "alerts_path": str(alerts_path),
            "rerun_targets_path": str(rerun_targets_path),
            "subset_run": requested_pair_ids is not None,
            "subset_pair_count_requested": len(requested_pair_ids or []),
            "pair_ids_file": str(pair_ids_file) if pair_ids_file else None,
            "merge_into_existing": bool(merge_into_existing),
            "bypass_cache_for_targets": bool(bypass_cache_for_targets),
            "resume": bool(resume),
            "max_workers": int(max_workers or 1),
            "rerun_round": int(rerun_round or 0),
        },
    )

    relation_status_count = Counter()
    schema_ok_count = 0
    for x in final_rows:
        rel = str(x.get("relation_status") or "schema_failure")
        relation_status_count[rel] += 1
        if x.get("response_schema_ok"):
            schema_ok_count += 1

    summary = {
        "pair_count_input": len(pairs),
        "pair_count_pruned": len(pruned),
        "pair_count_adjudicated": len(final_rows),
        "pair_count_unique_calls": len(stage_kept),
        "pair_count_actual_model_calls": actual_calls,
        "pair_count_cache_hits": cache_hit_count,
        "pair_count_skipped_existing": skipped_existing,
        "deterministic_alternative_count": 0,
        "stage_pruning_mode": "disabled_semantic_authority",
        "relation_status_count": dict(sorted(relation_status_count.items())),
        "schema_ok_count": schema_ok_count,
        "model": adjudicator.model_name,
        "output": str(out_path),
        "alert_count": len(alerts),
        "alerts_path": str(alerts_path),
        "rerun_targets_path": str(rerun_targets_path),
        "subset_run": requested_pair_ids is not None,
        "merge_into_existing": bool(merge_into_existing),
        "bypass_cache_for_targets": bool(bypass_cache_for_targets),
        "resume": bool(resume),
        "max_workers": int(max_workers or 1),
        "rerun_round": int(rerun_round or 0),
    }
    write_json(config.paths.run_dir / "pair_adjudication_meta.json", summary)
    return summary
