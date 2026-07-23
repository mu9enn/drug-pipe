from __future__ import annotations

import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from ..adjudicators.claude_code_runtime import ClaudeCodeRuntime, extract_json_object, safe_name
from ..io_utils import write_json, write_jsonl
from ..science_kb import ScienceKB
from ..settings import ProjectConfig
from .schemas import SIMPLE_QUESTION_OUTPUT_SCHEMA
from .canonical_io import canonical_task, load_canonical_sampling_inputs, update_manifest_tasks


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_simple_output(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["output_not_json_object"]
    try:
        jsonschema.validate(obj, SIMPLE_QUESTION_OUTPUT_SCHEMA)
    except jsonschema.ValidationError as exc:
        return [f"schema_invalid:{exc.message}"]
    if obj["status"] == "success":
        if not str(obj["public_question_text"]).strip():
            return ["success_public_question_empty"]
        if not obj["question_payload"]:
            return ["success_question_payload_empty"]
        for key in ["task", "inputs", "expected_output"]:
            if key not in obj["question_payload"]:
                return [f"success_question_payload_missing:{key}"]
    elif not str(obj["rationale"]).strip():
        return ["reject_rationale_empty"]
    return []


def _valid_edge_pool(graph: list[dict[str, Any]], cards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        edge for edge in graph
        if edge.get("relation_status") == "valid"
        and str(edge.get("source_tool")) in cards
        and str(edge.get("target_tool")) in cards
        and str(edge.get("source_tool")) != str(edge.get("target_tool"))
    ]


def sample_hidden_toolchain(
    edges: list[dict[str, Any]],
    cards: dict[str, dict[str, Any]],
    min_hops: int,
    max_hops: int,
    rng: random.Random,
    retries: int = 100,
) -> dict[str, Any] | None:
    pool = _valid_edge_pool(edges, cards)
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in pool:
        adjacency[str(edge["source_tool"])].append(edge)
    if not pool:
        return None
    for _ in range(retries):
        hops = rng.randint(min_hops, max_hops)
        first = rng.choice(pool)
        current = str(first["target_tool"])
        nodes, picked = [str(first["source_tool"]), current], [first]
        for _hop in range(1, hops):
            options = [
                edge for edge in adjacency.get(current, [])
                if str(edge["target_tool"]) not in nodes
            ]
            if not options:
                break
            edge = rng.choice(options)
            picked.append(edge)
            current = str(edge["target_tool"])
            nodes.append(current)
        if len(picked) == hops:
            return {
                "hidden_toolchain_nodes": nodes,
                "hidden_toolchain_edges": [_edge_context(edge) for edge in picked],
                "walk_hops": hops,
            }
    return None


def _edge_context(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_tool": edge.get("source_tool"),
        "target_tool": edge.get("target_tool"),
        "edge_type": edge.get("edge_type"),
        "relation_status": edge.get("relation_status"),
        "pair_id": edge.get("pair_id", ""),
        "confidence": edge.get("confidence", edge.get("confidence_calibrated")),
        "view": edge.get("view"),
    }


def _slot_context(slot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: slot.get(key)
        for key in ["name", "semantic_type", "format", "required", "parameter_kind", "description"]
        if slot.get(key) not in (None, "", [])
    }


def _tool_context(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": card.get("tool_id"),
        "description": card.get("description_summary"),
        "inputs": [_slot_context(x) for x in card.get("connectable_inputs") or card.get("inputs") or []],
        "outputs": [_slot_context(x) for x in card.get("connectable_outputs") or card.get("outputs") or []],
    }


def _decision_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            out[pair_id] = row
    return out


def _edge_evidence(edge: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = decisions.get(str(edge.get("pair_id") or ""), {})
    return {
        **_edge_context(edge),
        "satisfied_mappings": row.get("satisfied_mappings") or [],
    }


def _seed_target_id(seed: dict[str, Any]) -> str:
    return str(seed.get("uniprot_accession") or seed.get("protein_id") or "")


def _seed_compound_id(seed: dict[str, Any]) -> str:
    return str(seed.get("compound_id") or seed.get("canonical_smiles") or "")


def select_grounding_seed(
    records: list[dict[str, Any]],
    seen_targets: Counter[str],
    seen_compounds: Counter[str],
    max_repeat_target: int,
    max_repeat_compound: int,
    rng: random.Random,
) -> dict[str, Any]:
    if not records:
        return {}
    eligible = [
        row for row in records
        if seen_targets[_seed_target_id(row)] < max_repeat_target
        and seen_compounds[_seed_compound_id(row)] < max_repeat_compound
    ]
    if not eligible:
        minimum = min(
            seen_targets[_seed_target_id(row)] + seen_compounds[_seed_compound_id(row)]
            for row in records
        )
        eligible = [
            row for row in records
            if seen_targets[_seed_target_id(row)] + seen_compounds[_seed_compound_id(row)] == minimum
        ]
    seed = rng.choice(eligible)
    seen_targets[_seed_target_id(seed)] += 1
    seen_compounds[_seed_compound_id(seed)] += 1
    return seed


def _grounding_facts(
    kb: ScienceKB,
    topk: int,
    cards: list[dict[str, Any]],
    seed: dict[str, Any],
) -> list[dict[str, Any]]:
    text = json.dumps(cards, ensure_ascii=False).lower()
    need_sequence = "protein_sequence" in text or '"sequence"' in text
    pairs = [seed]
    pairs.extend(
        kb.find_target_ligand_pairs(
            protein_id=str(seed.get("protein_id") or ""),
            limit=max(1, topk),
            include_sequence=False,
        )
    )
    facts: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for pair in pairs:
        record_id = str(pair.get("record_id") or "")
        if not record_id or record_id in seen_records or len(facts) >= topk:
            continue
        seen_records.add(record_id)
        facts.append({
            "record_id": record_id,
            "source": pair["source_database"],
            "type": "target_ligand_pair",
            "value": {
                key: pair.get(key)
                for key in [
                    "protein_id", "uniprot_accession", "gene_name", "protein_name",
                    "pdb_ids", "compound_id", "compound_name", "canonical_smiles",
                    "activity_type", "activity_value", "activity_unit",
                ]
                if pair.get(key) not in (None, "", [])
            },
        })
    if need_sequence and seed and len(facts) < topk:
        protein = kb.get_protein(str(seed.get("uniprot_accession") or seed["protein_id"]))
        if protein and protein.get("sequence"):
            facts.append({
                "record_id": protein["record_id"],
                "source": protein["source_database"],
                "type": "protein",
                "value": {
                    key: protein.get(key)
                    for key in ["protein_id", "uniprot_accession", "gene_name", "protein_name", "organism", "sequence", "pdb_ids"]
                    if protein.get(key) not in (None, "", [])
                },
            })
    return facts


def _prepare_workdir(config: ProjectConfig, workdir: Path, context: dict[str, Any], prompt: str) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    write_json(workdir / "simple_context.json", context)
    write_json(workdir / "output_schema.json", SIMPLE_QUESTION_OUTPUT_SCHEMA)
    (workdir / "prompt.txt").write_text(prompt, encoding="utf-8")


def _science_kb_mcp(config: ProjectConfig, trace_path: Path) -> tuple[str, dict[str, Any]]:
    name = "molclaw-science-kb"
    root = config.paths.root
    return name, {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            "-m", "molclaw_kg.science_kb_mcp",
            "--sqlite", str(root / "science_kb/processed/science_kb.sqlite"),
            "--manifest", str(root / "science_kb/manifests/science_kb_manifest.json"),
            "--trace", str(trace_path),
        ],
        "env": {"PYTHONPATH": str(root / "src")},
    }


def _call_agent(
    runtime: ClaudeCodeRuntime,
    config: ProjectConfig,
    workdir: Path,
    prompt: str,
) -> tuple[dict[str, Any] | None, str]:
    server_name, server_cfg = _science_kb_mcp(config, workdir / "kb_query_trace.jsonl")
    run = runtime.run_prompt(
        prompt,
        run_label=workdir.name,
        add_dirs=[workdir],
        # The program has already selected compact grounding facts. Keeping the
        # MCP connected preserves provenance/runtime checks without letting a
        # simple-generation call expand its own context unpredictably.
        allowed_tools="Read",
        workdir=workdir,
        mcp_servers={server_name: server_cfg},
        expected_mcp_servers=[server_name],
    )
    raw = run.result_text or run.assistant_text or run.raw_stream
    parsed = extract_json_object(raw)
    parse_source = "assistant_text"
    output_json = workdir / "output.json"
    if parsed is None and output_json.is_file():
        try:
            candidate = json.loads(output_json.read_text(encoding="utf-8"))
            parsed = candidate if isinstance(candidate, dict) else None
            parse_source = "output.json" if parsed is not None else "none"
        except (OSError, json.JSONDecodeError):
            parse_source = "none"
    write_json(workdir / "agent_trace.json", {
        "return_code": run.return_code,
        "latency_sec": round(run.latency_sec, 4),
        "parsed_ok": isinstance(parsed, dict),
        "parse_source": parse_source,
        "session_file": run.session_file,
        "attempt_session_files": run.attempt_session_files,
        "claude_attempts": run.claude_attempts,
        "selected_claude_attempt": run.selected_claude_attempt,
    })
    return parsed, raw


_TOOL_BRAND_TOKENS = {
    "admet", "bioemu", "boltz2", "chai1", "chroma", "dleps", "equiscore",
    "esmfold", "evobind", "foldx", "fpocket", "goca", "hdock", "karmadock",
    "libinvent", "linkinvent", "openawsem", "openmm", "p2rank", "pepinvent",
    "prolif", "proteinmpnn", "pulchura", "quickvina", "reinvent",
}


def _tool_leaks(text: str, nodes: list[str], cards: dict[str, dict[str, Any]] | None = None) -> list[str]:
    leaks = []
    for tool in nodes:
        variants = {tool.lower(), tool.lower().replace("_", " "), tool.lower().replace("_", "-")}
        card = (cards or {}).get(tool, {})
        variants.update(str(x).lower() for x in card.get("aliases") or [] if len(str(x).strip()) >= 4)
        for token in tool.lower().split("_"):
            if token in _TOOL_BRAND_TOKENS:
                variants.add(token)
        if any(re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", text.lower()) for v in variants):
            leaks.append(tool)
    return leaks


def _sequence_hint(text: str) -> bool:
    return bool(
        re.search(r"\b(?:first|then|next|finally|afterwards|subsequently)\b", text, re.I)
        or re.search(r"(?:先|再|然后|最后|随后)", text)
    )


_USER_FOLLOWUP_PATTERNS = [
    "ask me", "ask the user", "please provide", "provide a seed",
    "user-provided", "user provided", "to be requested", "to be provided",
    "if unavailable", "if not available", "if coordinates are not available",
    "before proceeding", "need you to provide", "needs to be provided",
    "请用户提供", "需要用户提供", "请你提供", "需要你提供",
    "如果没有", "如果不可用", "等待用户", "用户提供的",
]

_PLACEHOLDER_PATTERNS = [
    "user_provided", "user-provided", "given_file", "target_protein",
    "ligand_smiles", "to_be_specified", "to be specified", "/path/to/",
    "path/to/file", "fake base64",
]


def _sample_text(sample: dict[str, Any]) -> str:
    return "\n".join([
        str(sample.get("public_question_text") or ""),
        json.dumps(sample.get("question_payload") or {}, ensure_ascii=False),
        str(sample.get("rationale") or ""),
    ]).lower()


def contains_user_followup_request(sample: dict[str, Any]) -> bool:
    text = _sample_text(sample)
    return any(pattern.lower() in text for pattern in _USER_FOLLOWUP_PATTERNS)


def contains_placeholder_or_fake_input(sample: dict[str, Any]) -> bool:
    text = _sample_text(sample)
    return any(pattern.lower() in text for pattern in _PLACEHOLDER_PATTERNS)


def _grounding_seed_context(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": seed.get("record_id"),
        "source": seed.get("source_database"),
        "seed_type": "target_ligand_pair",
        "target_uniprot": seed.get("uniprot_accession") or seed.get("protein_id"),
        "gene_name": seed.get("gene_name"),
        "compound_id": seed.get("compound_id"),
        "compound_name": seed.get("compound_name"),
        "smiles": seed.get("canonical_smiles"),
        "pdb_ids": seed.get("pdb_ids") or [],
    }


def sample_simple_questions(
    config: ProjectConfig,
    *,
    target_successes: int,
    max_attempts: int,
    min_hops: int = 2,
    max_hops: int = 4,
    json_repair_rounds: int = 1,
    science_kb_topk: int = 3,
    grounding_selection: str = "random_seeded",
    max_repeat_target: int = 2,
    max_repeat_compound: int = 2,
    seed: int | None = None,
    sampling_profile_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        target_successes < 1 or max_attempts < 1 or min_hops < 1 or max_hops < min_hops
        or max_repeat_target < 1 or max_repeat_compound < 1
    ):
        raise ValueError("invalid simple sampling limits")
    if grounding_selection != "random_seeded":
        raise ValueError(f"unsupported grounding selection: {grounding_selection}")
    run_dir = config.paths.run_dir
    kb_path, manifest_path = (
        config.paths.root / "science_kb/processed/science_kb.sqlite",
        config.paths.root / "science_kb/manifests/science_kb_manifest.json",
    )
    kb = ScienceKB(kb_path, manifest_path)
    graph, card_rows, decision_rows = load_canonical_sampling_inputs(run_dir)
    cards = {str(row["tool_id"]): row for row in card_rows if row.get("tool_id")}
    decisions = _decision_index(decision_rows)
    runtime, rng = ClaudeCodeRuntime(config), random.Random(seed)
    grounding_records = kb.list_target_ligand_pair_seeds()
    if not grounding_records:
        raise RuntimeError("Science-KB contains no target-ligand grounding seeds")
    seen_targets: Counter[str] = Counter()
    seen_compounds: Counter[str] = Counter()
    profile_values = (sampling_profile_meta or {}).get("resolved_sampling_config") or {}
    prompt_path = Path(
        (sampling_profile_meta or {}).get("prompt_path")
        or config.paths.configs / "prompts/toolchain_question_simple_v1.md"
    )
    repair_prompt_path = config.paths.configs / str(
        profile_values.get("json_repair_prompt")
        or "prompts/toolchain_question_json_repair_v1.md"
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    repair_prompt = repair_prompt_path.read_text(encoding="utf-8")
    results = run_dir / "results"
    intermediate = run_dir / "intermediate" / "stage3"
    workdirs = intermediate / "workdir" / "simple_toolchain_question"
    results.mkdir(parents=True, exist_ok=True)
    workdirs.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []

    with tqdm(total=target_successes, desc="simple-question-successes", unit="success") as progress:
        for attempt_index in range(1, max_attempts + 1):
            if len(successes) >= target_successes:
                break
            sample_id = f"simple_{attempt_index:04d}"
            blueprint = sample_hidden_toolchain(graph, cards, min_hops, max_hops, rng)
            base = {
                "sample_id": sample_id,
                "attempt_index": attempt_index,
                "status": "sampling_failed",
                "failure_reason": "",
                "created_at_utc": _now_utc(),
            }
            if blueprint is None:
                base["failure_reason"] = "hidden_toolchain_sampling_failed"
                attempts.append(base)
                continue
            nodes, edges = blueprint["hidden_toolchain_nodes"], blueprint["hidden_toolchain_edges"]
            tool_cards = [_tool_context(cards[node]) for node in nodes]
            grounding_seed = select_grounding_seed(
                grounding_records,
                seen_targets,
                seen_compounds,
                max_repeat_target,
                max_repeat_compound,
                rng,
            )
            context = {
                "hidden_toolchain": blueprint,
                "tool_cards": tool_cards,
                "edge_evidence": [_edge_evidence(edge, decisions) for edge in edges],
                "grounding_seed": _grounding_seed_context(grounding_seed),
                "grounding_facts": _grounding_facts(kb, science_kb_topk, tool_cards, grounding_seed),
            }
            sample_root = workdirs / f"{sample_id}__{safe_name(nodes[0])}__{safe_name(nodes[-1])}"
            attempt_dir = sample_root / "attempt_00"
            _prepare_workdir(config, attempt_dir, context, prompt)
            parsed, raw = _call_agent(runtime, config, attempt_dir, prompt)
            raw_path = attempt_dir / "raw_output.txt"
            raw_path.write_text(raw, encoding="utf-8")
            errors = validate_simple_output(parsed)
            repaired = False
            for repair_index in range(1, max(0, json_repair_rounds) + 1):
                if not errors:
                    break
                repaired = True
                repair_dir = sample_root / f"json_repair_{repair_index:02d}"
                repair_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_path, repair_dir / "raw_output.txt")
                write_json(repair_dir / "output_schema.json", SIMPLE_QUESTION_OUTPUT_SCHEMA)
                (repair_dir / "prompt.txt").write_text(repair_prompt, encoding="utf-8")
                parsed, raw = _call_agent(runtime, config, repair_dir, repair_prompt)
                raw_path = repair_dir / "raw_output.txt"
                raw_path.write_text(raw, encoding="utf-8")
                errors = validate_simple_output(parsed)
            base.update({
                **blueprint,
                "raw_llm_output_path": str(raw_path),
                "workdir": str(sample_root),
                "json_repaired": repaired,
                "grounding_seed": context["grounding_seed"],
            })
            if errors:
                base.update({"status": "parse_failed" if parsed is None else "schema_failed", "failure_reason": errors[0]})
                attempts.append(base)
                continue
            assert parsed is not None
            if parsed["status"] == "reject":
                base.update({"status": "reject", "failure_reason": parsed["rationale"], **parsed})
                attempts.append(base)
                continue
            public_text = str(parsed["public_question_text"])
            soft_warnings: list[str] = []
            leaks = _tool_leaks(public_text, nodes, cards)
            if leaks:
                soft_warnings.append(f"hidden_toolchain_leak:{leaks}")
            if _sequence_hint(public_text):
                soft_warnings.append("explicit_hidden_tool_order_hint")
            if contains_user_followup_request(parsed):
                base.update({
                    **parsed,
                    "status": "reject",
                    "failure_reason": "non_rolloutable_user_followup",
                    "soft_warnings": soft_warnings,
                })
                attempts.append(base)
                continue
            if contains_placeholder_or_fake_input(parsed):
                base.update({
                    **parsed,
                    "status": "reject",
                    "failure_reason": "placeholder_or_fake_input",
                    "soft_warnings": soft_warnings,
                })
                attempts.append(base)
                continue
            row = {
                **base,
                **parsed,
                "status": "success",
                "failure_reason": "",
                "grounding_sources": sorted({str(x["source"]) for x in context["grounding_facts"]}),
                "soft_warnings": soft_warnings,
            }
            attempts.append(row)
            successes.append(row)
            progress.update(1)

    kb.close()
    attempts_path = intermediate / "sample_attempts.jsonl"
    tasks_path = results / "tasks.jsonl"
    write_jsonl(attempts_path, attempts)
    tasks = [canonical_task(row, run_dir.name) for row in successes]
    write_jsonl(tasks_path, tasks)
    meta = {
        "sampling_mode": "simple_toolchain_question",
        "target_successes": target_successes,
        "success_count": len(successes),
        "attempt_count": len(attempts),
        "max_attempts": max_attempts,
        "grounding_selection": grounding_selection,
        "science_kb_seed_count": len(grounding_records),
        "max_repeat_target": max_repeat_target,
        "max_repeat_compound": max_repeat_compound,
        "unique_targets_selected": len(seen_targets),
        "unique_compounds_selected": len(seen_compounds),
        "failure_breakdown": dict(Counter(str(x.get("status")) for x in attempts if x.get("status") != "success")),
        "failure_reason_breakdown": dict(
            Counter(str(x.get("failure_reason")) for x in attempts if x.get("status") != "success")
        ),
        "valid_edge_pool_count": len(_valid_edge_pool(graph, cards)),
        "seed": seed,
        "attempts_path": str(attempts_path),
        "tasks_path": str(tasks_path),
        "intermediate_dir": str(intermediate),
        "created_at_utc": _now_utc(),
        "sampling_profile": (sampling_profile_meta or {}).get("sampling_profile"),
        "resolved_sampling_config": (sampling_profile_meta or {}).get("resolved_sampling_config"),
        "config_sha256": (sampling_profile_meta or {}).get("config_sha256"),
        "profile_sha256": (sampling_profile_meta or {}).get("profile_sha256"),
        "cli_overrides": (sampling_profile_meta or {}).get("cli_overrides"),
        "prompt_sha256": (sampling_profile_meta or {}).get("prompt_sha256"),
        "prompt_hashes": (sampling_profile_meta or {}).get("prompt_hashes"),
    }
    write_json(intermediate / "sampling_meta.json", meta)
    update_manifest_tasks(run_dir, len(tasks), sampling_profile_meta)
    return meta
