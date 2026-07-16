from __future__ import annotations

import json
import shutil
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import ValidationError
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from .adjudicators.claude_code_runtime import ClaudeCodeRuntime, extract_json_object, safe_name
from .io_utils import append_jsonl, atomic_write_jsonl, read_jsonl, sha256_text, stable_hash_obj, write_json, write_jsonl
from .models import Slot, ToolCard
from .settings import ProjectConfig
from .runtime_state import next_attempt_dir
from .stage_taxonomy import load_stage_taxonomy


def _infer_format(text: str) -> str:
    t = text.lower()
    if "pdbqt" in t:
        return "pdbqt"
    if "pdb" in t:
        return "pdb"
    if "sdf" in t:
        return "sdf"
    if "mol2" in t:
        return "mol2"
    if "smiles" in t or "smi" in t:
        return "smiles"
    if "csv" in t:
        return "csv"
    if "json" in t:
        return "json"
    if "base64" in t:
        return "base64"
    return "unknown"


def _infer_semantic(name: str, desc: str) -> str:
    s = f"{name} {desc}".lower()
    if "sequence" in s:
        return "protein_sequence"
    if "admet" in s:
        return "admet_profile"
    if "smiles" in s and "list" in s:
        return "ligand_smiles_list"
    if "smiles" in s:
        return "ligand_smiles"
    if "pocket" in s and ("center" in s or "box" in s):
        return "pocket_box"
    if "docking" in s and ("pose" in s or "pdbqt" in s):
        return "docking_pose"
    if "score" in s or "affinity" in s:
        return "docking_score"
    if "pdb" in s and ("fix" in s or "repair" in s):
        return "repaired_pdb"
    if "pdb" in s:
        return "protein_structure_pdb"
    if "candidate" in s or "filter" in s:
        return "candidate_set"
    if "rank" in s:
        return "ranking_table"
    if "visual" in s or "image" in s or "plot" in s:
        return "image_plot"
    return "unknown"


def _infer_parameter_kind(name: str, semantic_type: str, raw_type: str) -> str:
    n = name.lower()
    if n in {"dry_run", "overwrite", "force", "verbose"}:
        return "control"
    if n in {"mode", "samples", "num_samples", "batch_size", "seed", "temperature", "top_k"}:
        return "config"
    if semantic_type not in {"unknown", "image_plot"}:
        return "data"
    if raw_type in {"string", "array", "object"} and any(
        kw in n for kw in ("path", "file", "pdb", "cif", "smiles", "sequence", "ligand", "protein", "pocket")
    ):
        return "data"
    return "unknown"


def _slot_from_schema(name: str, spec: dict[str, Any], required: bool, source: str) -> Slot:
    raw_type = str(spec.get("type", "unknown"))
    desc = str(spec.get("description", ""))
    enum_vals = spec.get("enum")
    default_val = spec.get("default")
    extra_parts: list[str] = []
    if isinstance(enum_vals, list) and enum_vals:
        extra_parts.append(f"enum={enum_vals[:20]}")
    if default_val is not None:
        extra_parts.append(f"default={default_val}")
    if extra_parts:
        desc = f"{desc} [{' ; '.join(extra_parts)}]".strip()

    semantic = _infer_semantic(name, desc)
    fmt = _infer_format(f"{name} {desc}")
    card = "list" if raw_type == "array" else "single" if raw_type in {"string", "number", "integer", "boolean"} else "unknown"
    kind = _infer_parameter_kind(name, semantic, raw_type)
    requirement_status = "required" if required else "optional"
    return Slot(
        name=name,
        raw_type=raw_type,
        semantic_type=semantic,
        format=fmt,
        parameter_kind=kind,  # type: ignore[arg-type]
        requirement_status=requirement_status,  # type: ignore[arg-type]
        required=required,
        description=desc,
        source=source,  # type: ignore[arg-type]
        confidence=0.9 if source.endswith("schema") else 0.55,
        cardinality=card,  # type: ignore[arg-type]
    )


def _flatten_schema_slots(name: str, spec: dict[str, Any], required: bool, source: str) -> list[Slot]:
    out = [_slot_from_schema(name, spec, required, source)]
    typ = str(spec.get("type", "unknown"))
    if typ == "object":
        props = spec.get("properties") or {}
        req = set(spec.get("required") or [])
        if isinstance(props, dict):
            for child_name, child_spec in props.items():
                if not isinstance(child_spec, dict):
                    continue
                full = f"{name}.{child_name}"
                out.extend(_flatten_schema_slots(full, child_spec, child_name in req, source))
    elif typ == "array":
        items = spec.get("items")
        if isinstance(items, dict) and items.get("type") == "object":
            props = items.get("properties") or {}
            req = set(items.get("required") or [])
            if isinstance(props, dict):
                for child_name, child_spec in props.items():
                    if not isinstance(child_spec, dict):
                        continue
                    full = f"{name}[*].{child_name}"
                    out.extend(_flatten_schema_slots(full, child_spec, child_name in req, source))
    return out


def _deterministic_outputs(row: dict[str, Any]) -> list[Slot]:
    out: list[Slot] = []
    out_schema = row.get("outputSchema")
    if isinstance(out_schema, dict):
        props = out_schema.get("properties") or {}
        required = set(out_schema.get("required") or [])
        for name, spec in props.items():
            if isinstance(spec, dict):
                out.extend(_flatten_schema_slots(name, spec, name in required, "output_schema"))
    return out


def _build_connectable_inputs(inputs: list[Slot], preconditions: list[Slot]) -> list[Slot]:
    out: list[Slot] = []
    for s in [*inputs, *preconditions]:
        if s.parameter_kind == "control":
            continue
        if s.semantic_type == "unknown" and s.format == "unknown":
            continue
        out.append(Slot.model_validate(s.model_dump()))
    return out


def _build_connectable_outputs(outputs: list[Slot], side_effects: list[Slot]) -> list[Slot]:
    out: list[Slot] = []
    for s in [*outputs, *side_effects]:
        if s.semantic_type == "unknown" and s.format == "unknown":
            continue
        out.append(Slot.model_validate(s.model_dump()))
    return out


def _default_input_requirement_sets(inputs: list[Slot]) -> list[dict[str, Any]]:
    required_data = [x.name for x in inputs if x.required and x.parameter_kind in {"data", "unknown"}]
    optional_data = [x.name for x in inputs if (not x.required) and x.parameter_kind in {"data", "unknown"}]
    defaulted = [x.name for x in inputs if x.parameter_kind in {"config", "control"}]
    return [
        {
            "set_id": "default_execution",
            "condition": "default",
            "required_slots": required_data,
            "optional_slots": optional_data,
            "defaulted_slots": defaulted,
            "execution_meaning": "default",
        }
    ]


def _base_tool_card(row: dict[str, Any], primary_stage: str, secondary_stages: list[str]) -> ToolCard:
    tool_id = row.get("tool_id") or row.get("name")
    if not tool_id:
        raise ValueError("snapshot row missing tool_id/name")
    title = row.get("title") or tool_id
    desc = str(row.get("description", "") or "")
    input_schema = row.get("inputSchema") if isinstance(row.get("inputSchema"), dict) else {}
    props = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])

    inputs: list[Slot] = []
    for name, spec in props.items():
        if isinstance(spec, dict):
            inputs.append(_slot_from_schema(name, spec, name in required, "input_schema"))

    output_slots = _deterministic_outputs(row)
    preconditions: list[Slot] = []
    side_effects: list[Slot] = []
    connectable_inputs = _build_connectable_inputs(inputs, preconditions)
    connectable_outputs = _build_connectable_outputs(output_slots, side_effects)

    return ToolCard(
        tool_id=str(tool_id),
        title=str(title),
        description_summary=desc.strip()[:1000],
        primary_stage=primary_stage,
        secondary_stages=secondary_stages,
        aliases=[],
        inputs=inputs,
        outputs=output_slots,
        connectable_inputs=connectable_inputs,
        connectable_outputs=connectable_outputs,
        input_requirement_sets=_default_input_requirement_sets(inputs),
        preconditions=preconditions,
        side_effects=side_effects,
        needs_review=False,
    )


def _tool_terms(card: ToolCard) -> list[str]:
    terms = {
        card.tool_id.lower(),
        card.tool_id.lower().replace("-", "_"),
        card.title.lower().strip(),
    }
    for a in card.aliases:
        x = a.strip().lower()
        if x:
            terms.add(x)
    return sorted(t for t in terms if len(t) >= 3)


def _build_doc_index(cards: list[ToolCard], chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cleaned_chunks = []
    for c in chunks:
        text = str(c.get("text", ""))
        cleaned_chunks.append(
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "path": c.get("path"),
                "heading_path": c.get("heading_path", []),
                "text": text,
                "text_l": text.lower(),
            }
        )
    out: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        terms = _tool_terms(card)
        hits = []
        for c in cleaned_chunks:
            txt = c["text_l"]
            if any(t in txt for t in terms):
                hits.append(c)
        out[card.tool_id] = hits
    return out


def _load_prompt_template(config: ProjectConfig) -> str:
    prompt_path = config.paths.configs / "prompts" / "tool_card_agent_v1.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "You are constructing a semantically enriched tool-card for a computational drug discovery "
        "and biomolecular modeling MCP tool. Output strict JSON only."
    )


def _build_tool_card_prompt(template: str, *, fixed_primary_stage: str) -> str:
    return (
        f"{template.strip()}\n\n"
        "You are running inside one isolated workdir for exactly one tool-card task.\n"
        "Read local files in this directory first, then produce final strict JSON.\n\n"
        "Required local files:\n"
        "- task_context.json\n"
        "- tool_snapshot_row.json\n"
        "- deterministic_base_tool_card.json\n"
        "- stage_taxonomy.json\n"
        "- doc_context.jsonl\n\n"
        f"Hard constraint: primary_stage must equal fixed_primary_stage in task_context.json (expected: {fixed_primary_stage}).\n"
    )


def _prepare_toolcard_workdir(
    *,
    config: ProjectConfig,
    workdir: Path,
    base_card: ToolCard,
    row: dict[str, Any],
    fixed_primary_stage: str,
    allowed_stages: list[str],
    doc_context: list[dict[str, Any]],
    prompt: str,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)

    for item in config.runtime.skills_root.iterdir():
        dst = workdir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    write_json(workdir / "tool_snapshot_row.json", row)
    write_json(workdir / "deterministic_base_tool_card.json", base_card.model_dump())
    write_json(workdir / "stage_taxonomy.json", config.stage_taxonomy)
    write_json(
        workdir / "task_context.json",
        {
            "tool_id": base_card.tool_id,
            "fixed_primary_stage": fixed_primary_stage,
            "allowed_stages": allowed_stages,
            "doc_context_file": "doc_context.jsonl",
            "taxonomy_file": "stage_taxonomy.json",
            "snapshot_file": "tool_snapshot_row.json",
            "base_card_file": "deterministic_base_tool_card.json",
        },
    )
    write_jsonl(
        workdir / "doc_context.jsonl",
        [
            {
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "path": x.get("path"),
                "heading_path": x.get("heading_path", []),
                "text": str(x.get("text", "")),
            }
            for x in doc_context
        ],
    )
    (workdir / "prompt.txt").write_text(prompt, encoding="utf-8")


def _run_toolcard_attempt(
    *,
    config: ProjectConfig,
    base: ToolCard,
    row: dict[str, Any],
    fixed_primary: str,
    allowed_stages: set[str],
    doc_hits: list[dict[str, Any]],
    template: str,
    progress_path: Path,
    progress_lock: threading.Lock,
    rerun_round: int,
) -> dict[str, Any]:
    tool_id = base.tool_id
    status = "ok"
    err_msg = None
    built: ToolCard | None = None
    debug_extra: dict[str, Any] = {}
    run = None
    attempt_dir: Path | None = None
    cache_key = ""
    prompt = ""

    try:
        unit_dir = _toolcard_unit_dir(config, tool_id)
        attempt_dir = next_attempt_dir(unit_dir)
        prompt = _build_tool_card_prompt(template, fixed_primary_stage=fixed_primary)
        _prepare_toolcard_workdir(
            config=config,
            workdir=attempt_dir,
            base_card=base,
            row=row,
            fixed_primary_stage=fixed_primary,
            allowed_stages=sorted(allowed_stages),
            doc_context=doc_hits,
            prompt=prompt,
        )
        payload_for_cache = {
            "template_version": "tool_card_agent_v1",
            "tool_id": tool_id,
            "prompt_sha256": sha256_text(prompt),
            "base_card": base.model_dump(),
            "rerun_round": int(rerun_round or 0),
        }
        cache_key = stable_hash_obj(payload_for_cache)

        runtime = ClaudeCodeRuntime(config)
        run = runtime.run_prompt(
            prompt,
            run_label=f"toolcard_{tool_id}",
            add_dirs=[attempt_dir],
            allowed_tools=f"Read,Glob,mcp__{runtime.mcp_server_name}",
            workdir=attempt_dir,
        )

        parsed = extract_json_object(run.result_text) or extract_json_object(run.assistant_text) or extract_json_object(run.raw_stream)
        if parsed is None:
            repair_prompt = _build_parse_repair_prompt()
            (attempt_dir / "repair_prompt.txt").write_text(repair_prompt, encoding="utf-8")
            repair_run = runtime.run_prompt(
                repair_prompt,
                run_label=f"toolcard_repair_{tool_id}",
                add_dirs=[attempt_dir],
                allowed_tools="Read,Glob",
                workdir=attempt_dir,
            )
            parsed = (
                extract_json_object(repair_run.result_text)
                or extract_json_object(repair_run.assistant_text)
                or extract_json_object(repair_run.raw_stream)
            )

        if parsed is None:
            status = "parse_failed"
            err_msg = "agent output parse failed"
        else:
            parsed_primary_stage = str(parsed.get("primary_stage", fixed_primary)).strip()
            parsed_secondary_stages = parsed.get("secondary_stages", list(base.secondary_stages))
            invalid_secondary_stage = None
            if isinstance(parsed_secondary_stages, list):
                for x in parsed_secondary_stages:
                    sx = str(x).strip()
                    if sx and sx not in allowed_stages:
                        invalid_secondary_stage = sx
                        break

            main_payload, debug_extra = _extract_toolcard_model_payload(parsed, base)

            try:
                built = ToolCard.model_validate(main_payload)
                if parsed_primary_stage != fixed_primary:
                    status = "stage_mismatch"
                    err_msg = f"primary_stage mismatch: parsed={parsed_primary_stage}, fixed={fixed_primary}"
                    built = None
                elif invalid_secondary_stage is not None:
                    status = "secondary_stage_invalid"
                    err_msg = f"invalid secondary_stage: {invalid_secondary_stage}"
                    built = None
                else:
                    built.primary_stage = fixed_primary
                    built.secondary_stages = _normalize_secondary_stages(built.secondary_stages or list(base.secondary_stages), allowed_stages, fixed_primary)
                    if not built.connectable_inputs:
                        built.connectable_inputs = _build_connectable_inputs(built.inputs, built.preconditions)
                    if not built.connectable_outputs:
                        built.connectable_outputs = _build_connectable_outputs(built.outputs, built.side_effects)
                    if not built.input_requirement_sets:
                        built.input_requirement_sets = _default_input_requirement_sets(built.inputs)
            except ValidationError as e:
                status = "validation_failed"
                err_msg = str(e)
    except Exception as e:  # pragma: no cover - safety net for worker failures
        status = "worker_exception"
        err_msg = f"{type(e).__name__}: {e}"
        built = None
        debug_extra = {"worker_exception": type(e).__name__}

    fallback_applied = False
    if built is None:
        fallback_applied = True
        card_model = _fallback_from_alert(base)
    else:
        card_model = built

    attempt_dir_str = str(attempt_dir) if attempt_dir is not None else str(_toolcard_unit_dir(config, tool_id))
    workdir_str = getattr(run, "workdir", attempt_dir_str) if run is not None else attempt_dir_str
    session_file_str = getattr(run, "session_file", str(Path(attempt_dir_str) / "complete_session.jsonl")) if run is not None else str(Path(attempt_dir_str) / "complete_session.jsonl")
    created_at = datetime.now(timezone.utc).isoformat()
    debug_row = {
        "tool_id": tool_id,
        "status": status,
        "error": err_msg,
        "cache_key": cache_key,
        "fallback_applied": fallback_applied,
        "workdir": workdir_str,
        "session_file": session_file_str,
        "prompt_file": str(Path(attempt_dir_str) / "prompt.txt"),
        "doc_context_hits": len(doc_hits),
        "agent_extra": debug_extra,
        "created_at_utc": created_at,
        "rerun_round": int(rerun_round or 0),
        "attempt_dir": attempt_dir_str,
    }
    alert_row: dict[str, Any] | None = None
    if status != "ok":
        alert_row = {
            "tool_id": tool_id,
            "status": status,
            "error": err_msg,
            "cache_key": cache_key,
            "workdir": workdir_str,
            "session_file": session_file_str,
            "prompt_file": str(Path(attempt_dir_str) / "prompt.txt"),
            "created_at_utc": created_at,
            "fallback_applied": True,
            "rerun_round": int(rerun_round or 0),
            "attempt_dir": attempt_dir_str,
        }

    progress_row = {
        "tool_id": tool_id,
        "status": status,
        "error": err_msg,
        "cache_key": cache_key,
        "fallback_applied": fallback_applied,
        "workdir": workdir_str,
        "session_file": session_file_str,
        "prompt_file": str(Path(attempt_dir_str) / "prompt.txt"),
        "doc_context_hits": len(doc_hits),
        "agent_extra": debug_extra,
        "created_at_utc": created_at,
        "rerun_round": int(rerun_round or 0),
        "attempt_dir": attempt_dir_str,
        "card": card_model.model_dump(),
        "debug_row": debug_row,
        "alert_row": alert_row,
    }
    try:
        append_jsonl(progress_path, progress_row, lock=progress_lock)
    except Exception as e:  # pragma: no cover - best effort progress persistence
        debug_row["progress_write_error"] = f"{type(e).__name__}: {e}"

    return {
        "tool_id": tool_id,
        "card": card_model.model_dump(),
        "debug_row": debug_row,
        "alert_row": alert_row,
        "status": status,
        "error": err_msg,
        "fallback_applied": fallback_applied,
        "cache_key": cache_key,
        "attempt_dir": attempt_dir_str,
        "workdir": workdir_str,
        "session_file": session_file_str,
    }


def _normalize_secondary_stages(stages: list[str], allowed: set[str], primary: str) -> list[str]:
    out: list[str] = []
    for x in stages:
        s = str(x).strip()
        if not s or s == primary or s not in allowed:
            continue
        if s not in out:
            out.append(s)
    return out


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
                continue
            if isinstance(item, dict):
                cand = item.get("description") or item.get("name") or item.get("text")
                if isinstance(cand, str) and cand.strip():
                    out.append(cand.strip())
                else:
                    out.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
                continue
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if value is None:
        return []
    s = str(value).strip()
    return [s] if s else []


def _normalize_parsed_tool_card_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(parsed)
    fixed["aliases"] = _to_str_list(fixed.get("aliases"))
    fixed["secondary_stages"] = _to_str_list(fixed.get("secondary_stages"))
    return fixed


def _build_parse_repair_prompt() -> str:
    return (
        "Read complete_session.jsonl in current directory and output one strict JSON object only. "
        "No markdown, no code fences, no explanation."
    )


def _fallback_from_alert(base: ToolCard) -> ToolCard:
    fallback = ToolCard.model_validate(base.model_dump())
    fallback.needs_review = True
    return fallback


def _toolcard_progress_path(config: ProjectConfig) -> Path:
    return config.paths.run_dir / "tool_card_progress.jsonl"


def _toolcard_unit_dir(config: ProjectConfig, tool_id: str) -> Path:
    return config.paths.run_dir / "cc_workdir" / f"toolcard__{safe_name(tool_id)}"


def _load_toolcard_state_maps(
    *,
    config: ProjectConfig,
    progress_path: Path,
    use_existing: bool,
) -> tuple[dict[str, ToolCard], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cards_map: dict[str, ToolCard] = {}
    debug_map: dict[str, dict[str, Any]] = {}
    alert_map: dict[str, dict[str, Any]] = {}

    def ingest_card_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            tid = str(row.get("tool_id") or "").strip()
            if not tid:
                continue
            try:
                cards_map[tid] = ToolCard.model_validate(row)
            except Exception:
                continue

    def ingest_debug_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            tid = str(row.get("tool_id") or "").strip()
            if not tid:
                continue
            debug_map[tid] = row
            status = str(row.get("status") or "").strip()
            if status and status != "ok":
                alert_map[tid] = {
                    "tool_id": tid,
                    "status": status,
                    "error": row.get("error"),
                    "cache_key": row.get("cache_key"),
                    "workdir": row.get("workdir"),
                    "session_file": row.get("session_file"),
                    "prompt_file": row.get("prompt_file"),
                    "created_at_utc": row.get("created_at_utc"),
                    "fallback_applied": bool(row.get("fallback_applied", False)),
                }
            elif tid in alert_map:
                alert_map.pop(tid, None)

    if use_existing:
        out_path = config.paths.run_dir / "tool_cards.jsonl"
        debug_path = config.paths.run_dir / "tool_cards_debug.jsonl"
        alerts_path = config.paths.run_dir / "tool_card_alerts.jsonl"
        if out_path.exists():
            ingest_card_rows(read_jsonl(out_path))
        if debug_path.exists():
            ingest_debug_rows(read_jsonl(debug_path))
        if alerts_path.exists():
            for row in read_jsonl(alerts_path):
                tid = str(row.get("tool_id") or "").strip()
                if tid:
                    alert_map[tid] = row

    if progress_path.exists():
        for row in read_jsonl(progress_path):
            tid = str(row.get("tool_id") or "").strip()
            if not tid:
                continue
            card = row.get("card")
            debug_row = row.get("debug_row")
            alert_row = row.get("alert_row")
            if isinstance(card, dict):
                try:
                    cards_map[tid] = ToolCard.model_validate(card)
                except Exception:
                    pass
            if isinstance(debug_row, dict):
                debug_map[tid] = debug_row
                status = str(debug_row.get("status") or "").strip()
                if status and status != "ok":
                    if isinstance(alert_row, dict):
                        alert_map[tid] = alert_row
                else:
                    alert_map.pop(tid, None)
            else:
                status = str(row.get("status") or "").strip()
                if status and status != "ok":
                    if isinstance(alert_row, dict):
                        alert_map[tid] = alert_row
                elif tid in alert_map and status == "ok":
                    alert_map.pop(tid, None)

    return cards_map, debug_map, alert_map


def _load_tool_ids_filter(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"tool ids file not found: {path}")
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    if not out:
        raise ValueError(f"tool ids file is empty after filtering comments/blanks: {path}")
    return out


def _merge_tool_cards_with_existing(*, out_path: Path, updated_cards: list[ToolCard]) -> list[ToolCard]:
    existing_rows = read_jsonl(out_path) if out_path.exists() else []
    existing_map: dict[str, ToolCard] = {}
    for row in existing_rows:
        try:
            c = ToolCard.model_validate(row)
            existing_map[c.tool_id] = c
        except Exception:
            continue
    for card in updated_cards:
        existing_map[card.tool_id] = card
    return [existing_map[k] for k in sorted(existing_map.keys())]


def _merge_debug_rows_with_existing(*, out_path: Path, updated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_rows = read_jsonl(out_path) if out_path.exists() else []
    existing_map: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        tid = str(row.get("tool_id") or "").strip()
        if tid:
            existing_map[tid] = row
    for row in updated_rows:
        tid = str(row.get("tool_id") or "").strip()
        if tid:
            existing_map[tid] = row
    return [existing_map[k] for k in sorted(existing_map.keys())]


def _extract_toolcard_model_payload(parsed: dict[str, Any], base: ToolCard) -> tuple[dict[str, Any], dict[str, Any]]:
    p = _normalize_parsed_tool_card_payload(parsed)
    main = {
        "tool_id": p.get("tool_id", base.tool_id),
        "title": p.get("title", base.title),
        "description_summary": p.get("description_summary", base.description_summary),
        "primary_stage": p.get("primary_stage", base.primary_stage),
        "secondary_stages": p.get("secondary_stages", list(base.secondary_stages)),
        "aliases": p.get("aliases", []),
        "inputs": p.get("inputs", base.inputs),
        "outputs": p.get("outputs", base.outputs),
        "connectable_inputs": p.get("connectable_inputs", base.connectable_inputs),
        "connectable_outputs": p.get("connectable_outputs", base.connectable_outputs),
        "input_requirement_sets": p.get("input_requirement_sets", base.input_requirement_sets),
        "preconditions": p.get("preconditions", base.preconditions),
        "side_effects": p.get("side_effects", base.side_effects),
        "needs_review": bool(p.get("needs_review", False)),
    }

    known = set(main.keys())
    extras = {k: v for k, v in p.items() if k not in known}
    return main, extras


def build_tool_cards(
    config: ProjectConfig,
    snapshot_path: Path | None = None,
    tool_ids_file: Path | None = None,
    merge_into_existing: bool = False,
    max_workers: int = 1,
    resume: bool = False,
    rerun_round: int = 0,
) -> dict[str, Any]:
    if snapshot_path is None:
        snapshot_path = config.paths.run_dir / "tool_snapshot.jsonl"
    rows = read_jsonl(snapshot_path)
    if not rows:
        raise RuntimeError(f"tool snapshot is empty: {snapshot_path}")

    tool_ids_filter: set[str] | None = None
    if tool_ids_file is not None:
        tool_ids_filter = _load_tool_ids_filter(tool_ids_file)
        rows = [r for r in rows if str((r.get("tool_id") or r.get("name") or "")).strip() in tool_ids_filter]
        missing = sorted(tool_ids_filter - {str((r.get("tool_id") or r.get("name") or "")).strip() for r in rows})
        if missing:
            raise ValueError(f"requested tool_ids missing in snapshot: {missing}")

    taxonomy = load_stage_taxonomy(config.stage_taxonomy_path)
    tool_ids = [str((r.get("tool_id") or r.get("name") or "")).strip() for r in rows]
    if tool_ids_filter is None:
        taxonomy.validate_tool_coverage(tool_ids)
    allowed_stages = taxonomy.allowed_stages()

    base_cards: list[ToolCard] = []
    by_tool_row: dict[str, dict[str, Any]] = {}
    for row in rows:
        tool_id = str((row.get("tool_id") or row.get("name") or "")).strip()
        if not tool_id:
            continue
        primary = taxonomy.get_primary_stage(tool_id)
        secondaries = taxonomy.get_secondary_stages(tool_id)
        base = _base_tool_card(row, primary_stage=primary, secondary_stages=secondaries)
        base_cards.append(base)
        by_tool_row[tool_id] = row

    chunks = read_jsonl(config.paths.run_dir / "doc_chunks.jsonl")
    doc_index = _build_doc_index(base_cards, chunks)
    template = _load_prompt_template(config)
    tool_rows = sorted(base_cards, key=lambda x: x.tool_id)
    progress_path = _toolcard_progress_path(config)
    reuse_existing = bool(resume or merge_into_existing)
    cards_map, debug_map, alert_map = _load_toolcard_state_maps(
        config=config,
        progress_path=progress_path,
        use_existing=reuse_existing,
    )
    if not reuse_existing:
        cards_map = {}
        debug_map = {}
        alert_map = {}

    if tool_ids_filter is None and resume:
        pending_cards = [base for base in tool_rows if base.tool_id not in cards_map]
    else:
        pending_cards = [base for base in tool_rows if tool_ids_filter is None or base.tool_id in tool_ids_filter]

    progress_lock = threading.Lock()
    scheduled_count = len(pending_cards)
    skipped_count = len(tool_rows) - scheduled_count

    if scheduled_count > 0:
        max_workers = max(1, int(max_workers or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for base in pending_cards:
                tool_id = base.tool_id
                row = by_tool_row[tool_id]
                fixed_primary = taxonomy.get_primary_stage(tool_id)
                doc_hits = doc_index.get(tool_id, [])
                futures.append(
                    executor.submit(
                        _run_toolcard_attempt,
                        config=config,
                        base=base,
                        row=row,
                        fixed_primary=fixed_primary,
                        allowed_stages=allowed_stages,
                        doc_hits=doc_hits,
                        template=template,
                        progress_path=progress_path,
                        progress_lock=progress_lock,
                        rerun_round=rerun_round,
                    )
                )
            for fut in tqdm(as_completed(futures), total=len(futures), desc="tool-cards", unit="tool"):
                result = fut.result()
                tool_id = result["tool_id"]
                cards_map[tool_id] = ToolCard.model_validate(result["card"])
                debug_map[tool_id] = result["debug_row"]
                if result["alert_row"] is not None:
                    alert_map[tool_id] = result["alert_row"]
                else:
                    alert_map.pop(tool_id, None)

    final_cards = [cards_map[k] for k in sorted(cards_map.keys())]
    final_debug_rows = [debug_map[k] for k in sorted(debug_map.keys()) if k in debug_map]
    alerts = [alert_map[k] for k in sorted(alert_map.keys())]

    if len(final_cards) != len(cards_map):
        raise RuntimeError(f"tool card count mismatch: expected={len(cards_map)} got={len(final_cards)}")

    out_path = config.paths.run_dir / "tool_cards.jsonl"
    debug_path = config.paths.run_dir / "tool_cards_debug.jsonl"
    alerts_path = config.paths.run_dir / "tool_card_alerts.jsonl"
    rerun_path = config.paths.run_dir / "tool_card_rerun_targets.txt"
    atomic_write_jsonl(out_path, [c.model_dump() for c in final_cards])
    atomic_write_jsonl(debug_path, final_debug_rows)
    atomic_write_jsonl(alerts_path, alerts)
    rerun_path.write_text("".join(f"{x}\n" for x in sorted(alert_map.keys())), encoding="utf-8")

    status_breakdown = dict(sorted(Counter(str(x.get("status") or "unknown") for x in alerts).items()))
    write_json(
        config.paths.run_dir / "tool_card_alerts_meta.json",
        {
            "alert_count": len(alerts),
            "tool_ids": sorted(alert_map.keys()),
            "status_breakdown": status_breakdown,
            "alerts_path": str(alerts_path),
            "rerun_targets_path": str(rerun_path),
            "progress_path": str(progress_path),
            "resume": bool(resume),
            "max_workers": int(max_workers or 1),
            "skipped_tool_count": skipped_count,
            "scheduled_tool_count": scheduled_count,
        },
    )

    stages = sorted({c.primary_stage for c in final_cards})
    needs_review_count = sum(1 for c in final_cards if c.needs_review)
    agent_success_count = sum(1 for row in debug_map.values() if str(row.get("status") or "") == "ok")
    agent_failure_count = sum(1 for row in debug_map.values() if str(row.get("status") or "") != "ok")
    fallback_count = sum(1 for row in debug_map.values() if bool(row.get("fallback_applied")))
    write_json(
        config.paths.run_dir / "tool_cards_meta.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool_count": len(final_cards),
            "path": str(out_path),
            "debug_path": str(debug_path),
            "progress_path": str(progress_path),
            "stages": stages,
            "stage_taxonomy_path": str(config.stage_taxonomy_path),
            "stage_taxonomy_schema_version": taxonomy.raw.get("schema_version"),
            "mapped_tool_count": len(taxonomy.tool_stage_map),
            "agent_success_count": agent_success_count,
            "agent_failure_count": agent_failure_count,
            "fallback_count": fallback_count,
            "alert_count": len(alerts),
            "needs_review_count": needs_review_count,
            "run_completed_with_alerts": len(alerts) > 0,
            "subset_run": tool_ids_filter is not None,
            "subset_tool_count": len(tool_ids_filter or []),
            "tool_ids_file": str(tool_ids_file) if tool_ids_file else None,
            "merge_into_existing": bool(merge_into_existing),
            "resume": bool(resume),
            "max_workers": int(max_workers or 1),
            "rerun_round": int(rerun_round or 0),
        },
    )
    return {
        "tool_count": len(final_cards),
        "output": str(out_path),
        "debug_output": str(debug_path),
        "progress_path": str(progress_path),
        "stage_taxonomy_path": str(config.stage_taxonomy_path),
        "alert_count": len(alerts),
        "alerts_path": str(alerts_path),
        "rerun_targets_path": str(rerun_path),
        "subset_run": tool_ids_filter is not None,
        "merge_into_existing": bool(merge_into_existing),
        "resume": bool(resume),
        "max_workers": int(max_workers or 1),
        "rerun_round": int(rerun_round or 0),
    }
