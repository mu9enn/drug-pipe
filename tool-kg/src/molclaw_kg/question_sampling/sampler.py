from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from ..adjudicators.claude_code_runtime import ClaudeCodeRuntime, extract_json_object, safe_name
from ..constants import TRANSITION_EDGE_TYPES
from ..io_utils import write_json, write_jsonl
from ..science_kb import ScienceKB
from ..settings import ProjectConfig
from .schemas import QUESTION_SAMPLER_OUTPUT_SCHEMA
from .canonical_io import canonical_task, load_canonical_sampling_inputs, update_manifest_tasks


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sampling_config(config: ProjectConfig) -> dict[str, Any]:
    path = config.paths.configs / "question_sampling_v2.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage3 config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("legacy_policies") or {}


def _load_template(config: ProjectConfig, repair: bool = False) -> str:
    name = "toolchain_question_repair_v1.md" if repair else "toolchain_question_sampler_v1.md"
    path = config.paths.configs / "prompts" / "legacy" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def _copy_skills_bundle(skills_root: Path, workdir: Path) -> None:
    src_claude = skills_root / ".claude"
    src_md = skills_root / "CLAUDE.md"
    if not src_claude.is_dir() or not src_md.is_file():
        raise FileNotFoundError(f"invalid skills bundle: {skills_root}")
    shutil.copytree(src_claude, workdir / ".claude", dirs_exist_ok=True)
    shutil.copy2(src_md, workdir / "CLAUDE.md")


def _prepare_attempt_workdir(
    config: ProjectConfig,
    workdir: Path,
    files: dict[str, Any],
    prompt: str,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    _copy_skills_bundle(config.runtime.skills_root, workdir)
    for name, value in files.items():
        write_json(workdir / name, value)
    write_json(workdir / "output_schema.json", QUESTION_SAMPLER_OUTPUT_SCHEMA)
    (workdir / "prompt.txt").write_text(prompt, encoding="utf-8")


def _build_prompt(template: str, repair_round: int) -> str:
    required = [
        "task_context.json",
        "anchor_spec.json",
        "kg_context.json",
        "tool_catalog.json",
        "edge_decision_context.json",
        "output_schema.json",
        ".claude/skills/**/*",
    ]
    if repair_round:
        required.extend(["previous_proposal.json", "validation_feedback.json"])
    return (
        f"{template}\n\n"
        f"Repair round: {repair_round}\n"
        "Read these local files before answering:\n- "
        + "\n- ".join(required)
        + "\nUse the read-only local Science-KB MCP for real scientific entities. Return strict JSON only."
    )


def _science_kb_paths(config: ProjectConfig, sampling_cfg: dict[str, Any]) -> tuple[Path, Path, str]:
    kb_cfg = sampling_cfg.get("science_kb") or {}
    sqlite_path = (config.paths.root / str(kb_cfg.get("sqlite_path", "science_kb/processed/science_kb.sqlite"))).resolve()
    manifest_path = (config.paths.root / str(kb_cfg.get("manifest_path", "science_kb/manifests/science_kb_manifest.json"))).resolve()
    server_name = str(kb_cfg.get("mcp_server_name", "molclaw-science-kb"))
    if not sqlite_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"fixed local Science-KB missing: {sqlite_path} / {manifest_path}. "
            f"Build it with: python scripts/build_science_kb.py --replace"
        )
    return sqlite_path, manifest_path, server_name


def _science_kb_mcp_server(sqlite_path: Path, manifest_path: Path, trace_path: Path) -> dict[str, Any]:
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            "-m",
            "molclaw_kg.science_kb_mcp",
            "--sqlite",
            str(sqlite_path),
            "--manifest",
            str(manifest_path),
            "--trace",
            str(trace_path),
        ],
        "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")},
    }


def _edge_debug_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        key = (str(r.get("source_tool")), str(r.get("target_tool")), str(r.get("edge_type")))
        old = out.get(key)
        if old is None or len(r.get("satisfied_mappings") or []) > len(old.get("satisfied_mappings") or []):
            out[key] = r
    return out


def _filtered_edges(
    rows: list[dict[str, Any]],
    debug_idx: dict[tuple[str, str, str], dict[str, Any]],
    edge_profile: str,
    partial_policy: str,
    sampling_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    profiles = sampling_cfg.get("edge_profiles") or {}
    if edge_profile not in profiles:
        raise ValueError(f"unknown edge_profile: {edge_profile}")
    allowed_views = set(profiles[edge_profile].get("allowed_views") or [])
    require_partial_mapping = bool((sampling_cfg.get("partial_edge_policy") or {}).get("require_satisfied_mapping", True))
    out = []
    for r in rows:
        edge_type = str(r.get("edge_type", ""))
        is_canonical = r.get("schema_version") == "tool_kg_graph_edge_v1" and r.get("eligible_for_sampling") is True
        if not is_canonical and r.get("view") not in allowed_views:
            continue
        if r.get("relation_status") != "valid" or not bool(r.get("direct_transition", False)):
            continue
        if edge_type not in TRANSITION_EDGE_TYPES:
            continue
        if partial_policy == "exclude" and edge_type == "generates_partial_input_for":
            continue
        if edge_type == "generates_partial_input_for" and require_partial_mapping:
            dbg = debug_idx.get((str(r.get("source_tool")), str(r.get("target_tool")), edge_type), {})
            if not (dbg.get("satisfied_mappings") or []):
                continue
        out.append(r)
    return out


def _adjacency(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in edges:
        out[str(e["source_tool"])].append(e)
    return out


def _has_walk(node: str, steps: int, adj: dict[str, list[dict[str, Any]]], memo: dict[tuple[str, int], bool]) -> bool:
    key = (node, steps)
    if key in memo:
        return memo[key]
    if steps == 0:
        return True
    memo[key] = any(_has_walk(str(e["target_tool"]), steps - 1, adj, memo) for e in adj.get(node, []))
    return memo[key]


def _walk(
    starts: list[str],
    hops: int,
    adj: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    memo: dict[tuple[str, int], bool],
    usage: Counter[str],
    excluded_terminal: set[str],
) -> tuple[list[str], list[dict[str, Any]], str | None]:
    if not starts:
        return [], [], "no_feasible_start"
    weights = [1.0 / (1.0 + usage[s]) for s in starts]
    cur = rng.choices(starts, weights=weights, k=1)[0]
    nodes = [cur]
    picked_edges = []
    for idx in range(hops):
        remain = hops - idx - 1
        options = [e for e in adj.get(cur, []) if _has_walk(str(e["target_tool"]), remain, adj, memo)]
        if remain == 0:
            non_utility = [e for e in options if str(e["target_tool"]) not in excluded_terminal]
            options = non_utility or options
        if not options:
            return nodes, picked_edges, f"dead_end_at_{idx}"
        ew = [
            max(0.01, float(e.get("confidence_calibrated", 0.5))) / (1.0 + usage[str(e["target_tool"])])
            for e in options
        ]
        e = rng.choices(options, weights=ew, k=1)[0]
        picked_edges.append(e)
        cur = str(e["target_tool"])
        nodes.append(cur)
    usage.update(nodes)
    return nodes, picked_edges, None


def _edge_public(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_tool": e.get("source_tool"),
        "target_tool": e.get("target_tool"),
        "edge_type": e.get("edge_type"),
        "confidence": e.get("confidence_calibrated", e.get("confidence")),
        "pair_id": e.get("pair_id", ""),
        "view": e.get("view", "unknown"),
        "relation_status": e.get("relation_status", "valid"),
    }


def _slot_public(slot: dict[str, Any]) -> dict[str, Any]:
    """Keep only slot facts needed for workflow planning and closure."""
    return {
        key: slot.get(key)
        for key in [
            "name", "raw_type", "semantic_type", "format", "cardinality",
            "parameter_kind", "requirement_status", "required",
        ]
        if slot.get(key) not in (None, "", [])
    }


def _card_public(card: dict[str, Any]) -> dict[str, Any]:
    """Create a readable planning catalog instead of exposing full audit cards."""
    return {
        "tool_id": card.get("tool_id"),
        "title": card.get("title"),
        "description_summary": card.get("description_summary"),
        "primary_stage": card.get("primary_stage"),
        "connectable_inputs": [_slot_public(x) for x in card.get("connectable_inputs") or []],
        "connectable_outputs": [_slot_public(x) for x in card.get("connectable_outputs") or []],
        "input_requirement_sets": card.get("input_requirement_sets") or [],
    }


def _debug_public(row: dict[str, Any]) -> dict[str, Any]:
    """Expose actionable mapping evidence while keeping verbose audit text in sidecars."""
    return {
        "source_tool": row.get("source_tool"),
        "target_tool": row.get("target_tool"),
        "edge_type": row.get("edge_type"),
        "pair_id": row.get("pair_id"),
        "satisfied_mappings": [
            {
                key: mapping.get(key)
                for key in ["source_output_slot", "target_input_slot", "source_slot", "target_slot"]
                if mapping.get(key) not in (None, "", [])
            }
            for mapping in row.get("satisfied_mappings") or []
            if isinstance(mapping, dict)
        ],
        "unsatisfied_required_inputs": [
            {
                key: missing.get(key)
                for key in [
                    "target_input_slot", "input_name", "semantic_type", "format",
                    "can_be_user_provided", "can_be_satisfied_by_other_upstream_tool",
                ]
                if missing.get(key) not in (None, "", [])
            }
            for missing in row.get("unsatisfied_required_inputs") or []
            if isinstance(missing, dict)
        ],
        "evidence_refs": row.get("evidence_refs") or [],
    }


def _tokenize(text: str) -> set[str]:
    return {x for x in re.split(r"[^a-zA-Z0-9_]+", str(text).lower()) if x}


def _compat(a: dict[str, Any], b: dict[str, Any]) -> float:
    asem, bsem = str(a.get("semantic_type", "unknown")), str(b.get("semantic_type", "unknown"))
    afmt, bfmt = str(a.get("format", "unknown")).lower(), str(b.get("format", "unknown")).lower()
    if asem != "unknown" and bsem != "unknown":
        sem = 1.0 if asem == bsem else (0.72 if asem in bsem or bsem in asem else 0.1)
    else:
        sem = 0.35
    if afmt != "unknown" and bfmt != "unknown":
        fmt = 1.0 if afmt == bfmt else 0.15
    else:
        fmt = 0.35
    ta, tb = _tokenize(a.get("name", "")), _tokenize(b.get("name", ""))
    name = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return 0.62 * sem + 0.25 * fmt + 0.13 * name


def _required_sets(card: dict[str, Any]) -> list[list[dict[str, Any]]]:
    inputs = {str(x.get("name")): x for x in card.get("connectable_inputs") or [] if isinstance(x, dict)}
    sets = []
    for rs in card.get("input_requirement_sets") or []:
        if not isinstance(rs, dict):
            continue
        slots = [inputs[n] for n in rs.get("required_slots") or [] if n in inputs]
        if slots:
            sets.append(slots)
    if sets:
        return sets
    fallback = [x for x in inputs.values() if bool(x.get("required")) or str(x.get("requirement_status")) == "required"]
    return [fallback]


def _outputs(card: dict[str, Any]) -> list[dict[str, Any]]:
    vals = []
    for key in ["connectable_outputs", "outputs", "side_effects"]:
        vals.extend(x for x in card.get(key) or [] if isinstance(x, dict))
    return vals


def _topological(tools: set[str], edges: list[tuple[str, str]]) -> list[str] | None:
    adj: dict[str, list[str]] = defaultdict(list)
    indeg = {t: 0 for t in tools}
    for a, b in set(edges):
        if a not in tools or b not in tools or a == b:
            return None
        adj[a].append(b)
        indeg[b] += 1
    q = deque(sorted(t for t, d in indeg.items() if d == 0))
    out = []
    while q:
        cur = q.popleft()
        out.append(cur)
        for nxt in adj.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return out if len(out) == len(tools) else None


def _topological_node_ids(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str] | None:
    node_ids = {str(x["node_id"]) for x in nodes}
    pairs = [(str(x["source"]), str(x["target"])) for x in edges]
    return _topological(node_ids, pairs)


def _contains_value(container: Any, wanted: Any) -> bool:
    if container == wanted:
        return True
    if isinstance(container, dict):
        return any(_contains_value(value, wanted) for value in container.values())
    if isinstance(container, list):
        return any(_contains_value(value, wanted) for value in container)
    return False


def _ancestors(tool: str, edges: list[tuple[str, str]]) -> set[str]:
    preds: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        preds[b].append(a)
    seen, stack = set(), list(preds.get(tool, []))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(preds.get(cur, []))
    return seen


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flatten_values(v)]
    if isinstance(value, list):
        return [x for v in value for x in _flatten_values(v)]
    return [str(value)]


def _closure_analysis(
    tools: set[str],
    edges: list[tuple[str, str]],
    cards: dict[str, dict[str, Any]],
    grounded_inputs: list[dict[str, Any]],
    question_payload: dict[str, Any],
    edge_debug: dict[tuple[str, str, str], dict[str, Any]],
    edge_type_by_pair: dict[tuple[str, str], str],
) -> dict[str, Any]:
    payload_blob = "\n".join(_flatten_values(question_payload)).lower()
    per_tool, open_reqs = [], []
    for tool in _topological(tools, edges) or sorted(tools):
        card = cards.get(tool, {})
        candidates = _required_sets(card)
        best_set_report = None
        for req_set_idx, reqs in enumerate(candidates):
            covered, missing = [], []
            for req in reqs:
                name = str(req.get("name", ""))
                kind = str(req.get("parameter_kind", "unknown"))
                if kind in {"config", "control"} and name.lower() in payload_blob:
                    covered.append({"input": name, "source": "question_payload_config"})
                    continue
                best: dict[str, Any] | None = None
                best_score = 0.0
                for inp in grounded_inputs:
                    score = _compat(inp, req)
                    if score > best_score:
                        best_score, best = score, {"source": "grounded_initial_input", "name": inp.get("name"), "score": score}
                for anc in _ancestors(tool, edges):
                    for out in _outputs(cards.get(anc, {})):
                        score = _compat(out, req)
                        if score > best_score:
                            best_score, best = score, {"source": "ancestor_tool", "tool_id": anc, "output_slot": out.get("name"), "score": score}
                    et = edge_type_by_pair.get((anc, tool), "")
                    dbg = edge_debug.get((anc, tool, et), {})
                    for mapping in dbg.get("satisfied_mappings") or []:
                        if str(mapping.get("target_input_slot")) == name:
                            best_score, best = 1.0, {"source": "edge_debug_mapping", "tool_id": anc, "mapping": mapping}
                if best and best_score >= 0.58:
                    covered.append({"input": name, "coverage": best})
                else:
                    missing.append(
                        {
                            "requirement_id": f"req::{tool}::{name}",
                            "tool_id": tool,
                            "input_name": name,
                            "semantic_type": req.get("semantic_type", "unknown"),
                            "format": req.get("format", "unknown"),
                            "parameter_kind": kind,
                        }
                    )
            report = {"requirement_set_index": req_set_idx, "covered_inputs": covered, "missing_inputs": missing}
            if best_set_report is None or len(missing) < len(best_set_report["missing_inputs"]):
                best_set_report = report
        best_set_report = best_set_report or {"requirement_set_index": 0, "covered_inputs": [], "missing_inputs": []}
        open_reqs.extend(best_set_report["missing_inputs"])
        per_tool.append({"tool_id": tool, "status": "closed" if not best_set_report["missing_inputs"] else "open", **best_set_report})
    return {"closure_status": "closed" if not open_reqs else "open", "per_tool_requirements": per_tool, "open_requirements": open_reqs}


def _validate_grounding(kb: ScienceKB, inputs: list[dict[str, Any]], refs: list[str], public_text: str, payload: dict[str, Any]) -> list[str]:
    errors = []
    validation = kb.validate_records(refs)
    if not validation["valid"]:
        errors.append(f"unknown_grounding_refs:{validation['missing']}")
    public_blob = public_text.lower()
    for inp in inputs:
        raw_value = inp.get("value", "")
        value = str(raw_value).strip()
        fmt = str(inp.get("format", "")).lower()
        rid = inp.get("grounding_record_id")
        source = str(inp.get("source", "science_kb"))
        if not value:
            errors.append(f"empty_grounded_value:{inp.get('name')}")
        if value and value.lower() not in public_blob and not _contains_value(payload, raw_value):
            errors.append(f"grounded_value_not_in_public_question:{inp.get('name')}")
        if any(x in fmt for x in ["path", "file"]) or value.startswith("/"):
            errors.append(f"file_path_user_input_forbidden:{inp.get('name')}")
        if source != "config" and not rid:
            errors.append(f"scientific_input_missing_grounding_ref:{inp.get('name')}")
        if rid and rid not in refs:
            errors.append(f"grounding_ref_not_declared:{rid}")
        if rid and rid in validation["found"]:
            record_blob = json.dumps(validation["found"][rid]["record"], ensure_ascii=False).lower()
            if value and value.lower() not in record_blob:
                errors.append(f"grounded_value_not_in_record:{inp.get('name')}:{rid}")
    return errors


def _validate_edge_claims(
    tools: set[str],
    proposed_edges: list[dict[str, Any]],
    graph_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[str], list[tuple[str, str]], dict[tuple[str, str], str]]:
    errors, edges, edge_type_by_pair = [], [], {}
    for e in proposed_edges:
        a, b = str(e.get("source_tool", "")), str(e.get("target_tool", ""))
        if a not in tools or b not in tools:
            errors.append(f"edge_tool_missing:{a}->{b}")
            continue
        support = str(e.get("support_source", ""))
        if support == "toolkg":
            matches = [
                x for x in graph_index.get((a, b), [])
                if x.get("view") == "core" and x.get("relation_status") == "valid" and bool(x.get("direct_transition"))
            ]
            if not matches:
                errors.append(f"unsupported_toolkg_edge:{a}->{b}")
                continue
            best = max(matches, key=lambda x: float(x.get("confidence_calibrated", 0)))
            support_ref = str(e.get("support_ref", ""))
            if support_ref not in {str(best.get("pair_id", "")), str(best.get("edge_id", ""))}:
                errors.append(f"toolkg_support_ref_mismatch:{a}->{b}:{support_ref}")
                continue
            edge_type_by_pair[(a, b)] = str(best.get("edge_type", ""))
        else:
            errors.append(f"noncanonical_edge_support:{a}->{b}:{support}")
            continue
        edges.append((a, b))
    if _topological(tools, edges) is None:
        errors.append("workflow_cycle_or_disconnected_edge")
    return errors, edges, edge_type_by_pair


def _problem_text_errors(text: str, payload: dict[str, Any], leak_patterns: list[re.Pattern[str]], sampling_cfg: dict[str, Any]) -> list[str]:
    rules = sampling_cfg.get("executability") or {}
    blob = text + "\n" + "\n".join(_flatten_values(payload))
    errors = []
    for pattern in rules.get("forbidden_placeholder_patterns") or []:
        if re.search(pattern, blob):
            errors.append(f"forbidden_placeholder:{pattern}")
    for pattern in rules.get("forbidden_workflow_design_patterns") or []:
        if re.search(pattern, text):
            errors.append(f"workflow_design_question:{pattern}")
    if not any(re.search(p, text) for p in rules.get("execution_intent_patterns") or []):
        errors.append("missing_execution_intent")
    for p in leak_patterns:
        m = p.search(text)
        if m:
            errors.append(f"tool_name_leak:{m.group(0)}")
            break
    if re.search(r"\bfirst\b[\s\S]{0,140}\b(?:then|next)\b", text, re.I) or re.search(r"先[\s\S]{0,80}再", text):
        errors.append("explicit_sequence_hint")
    return errors


def _compile_leak_patterns(cards: dict[str, dict[str, Any]]) -> list[re.Pattern[str]]:
    names = set()
    for tid, card in cards.items():
        names.update([tid, tid.replace("_", " "), tid.replace("_", "-")])
        names.update(str(x) for x in card.get("aliases") or [])
    return [re.compile(rf"(?<![a-zA-Z0-9]){re.escape(x)}(?![a-zA-Z0-9])", re.I) for x in names if len(x) >= 3]


def _trajectory(
    tools: set[str],
    edges: list[tuple[str, str]],
    inputs: list[dict[str, Any]],
    final_deliverable: str,
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    topo_tools = _topological(tools, edges) or sorted(tools)
    intent_map = {(str(x.get("role")), str(x.get("tool_id") or "")): str(x.get("message_intent")) for x in intents}
    nodes, wf_edges = [], []
    plan = "llm::plan::0"
    nodes.append({"node_id": plan, "type": "llm", "label": "Plan", "tool_id": None, "llm_role": "plan", "message_intent": intent_map.get(("plan", ""), "Plan an executable scientific workflow."), "payload": None})
    for idx, inp in enumerate(inputs, 1):
        nid = f"input::{idx}::{safe_name(str(inp.get('name')))}"
        nodes.append({"node_id": nid, "type": "input", "label": str(inp.get("name")), "tool_id": None, "llm_role": None, "message_intent": None, "payload": inp})
        wf_edges.append({"edge_id": f"edge::{nid}__plan", "source": nid, "target": plan, "relation": "provides_context"})
    for tool in topo_tools:
        p, t, i = f"llm::parameterize::{tool}", f"tool::{tool}", f"llm::interpret::{tool}"
        nodes.extend([
            {"node_id": p, "type": "llm", "label": f"Parameterize {tool}", "tool_id": None, "llm_role": "parameterize", "message_intent": intent_map.get(("parameterize", tool), f"Prepare validated inputs for {tool}."), "payload": None},
            {"node_id": t, "type": "tool", "label": tool, "tool_id": tool, "llm_role": None, "message_intent": None, "payload": None},
            {"node_id": i, "type": "llm", "label": f"Interpret {tool}", "tool_id": None, "llm_role": "interpret", "message_intent": intent_map.get(("interpret", tool), f"Interpret outputs from {tool}."), "payload": None},
        ])
        wf_edges.extend([
            {"edge_id": f"edge::{p}__{t}", "source": p, "target": t, "relation": "parameterizes_tool"},
            {"edge_id": f"edge::{t}__{i}", "source": t, "target": i, "relation": "tool_observation"},
        ])
    roots = set(tools) - {b for _, b in edges}
    for root in roots:
        wf_edges.append({"edge_id": f"edge::plan__{root}", "source": plan, "target": f"llm::parameterize::{root}", "relation": "routes_to_next"})
    for a, b in edges:
        wf_edges.append({"edge_id": f"edge::{a}__{b}", "source": f"llm::interpret::{a}", "target": f"llm::parameterize::{b}", "relation": "routes_to_next"})
    route, summary, output = "llm::route::final", "llm::summarize::final", "output::final"
    nodes.extend([
        {"node_id": route, "type": "llm", "label": "Route final outputs", "tool_id": None, "llm_role": "route", "message_intent": intent_map.get(("route", ""), "Route validated outputs to synthesis."), "payload": None},
        {"node_id": summary, "type": "llm", "label": "Summarize", "tool_id": None, "llm_role": "summarize", "message_intent": intent_map.get(("summarize", ""), "Summarize the actual scientific results."), "payload": None},
        {"node_id": output, "type": "output", "label": "Final deliverable", "tool_id": None, "llm_role": None, "message_intent": None, "payload": {"description": final_deliverable}},
    ])
    leaves = set(tools) - {a for a, _ in edges}
    for leaf in leaves:
        wf_edges.append({"edge_id": f"edge::{leaf}__route", "source": f"llm::interpret::{leaf}", "target": route, "relation": "routes_to_next"})
    wf_edges.extend([
        {"edge_id": "edge::route__summary", "source": route, "target": summary, "relation": "routes_to_next"},
        {"edge_id": "edge::summary__output", "source": summary, "target": output, "relation": "summarizes_result"},
    ])
    topo_nodes = _topological_node_ids(nodes, wf_edges)
    if topo_nodes is None:
        raise ValueError("generated trajectory graph is cyclic or references an unknown node")
    return {
        "schema_version": "trajectory_v2_graph",
        "workflow_graph": {"nodes": nodes, "edges": wf_edges},
        "execution_plan": {"topological_order": topo_nodes, "tool_order": topo_tools},
        "final_deliverable": final_deliverable,
    }


def _materialize_edges(
    proposed: list[dict[str, Any]],
    graph_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    for e in proposed:
        a, b = str(e.get("source_tool")), str(e.get("target_tool"))
        rows = graph_index.get((a, b), [])
        matching = [
            row
            for row in rows
            if str(e.get("support_ref")) in {
                str(row.get("pair_id", "")),
                str(row.get("edge_id", "")),
            }
        ]
        if not matching:
            raise ValueError(f"validated canonical edge missing during materialization: {a}->{b}")
        best = max(matching, key=lambda x: float(x.get("confidence_calibrated", 0)))
        out.append(_edge_public(best))
    return out


def _run_agent_attempt(
    runtime: ClaudeCodeRuntime,
    workdir: Path,
    prompt: str,
    server_name: str,
    server_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    run = runtime.run_prompt(
        prompt,
        run_label=workdir.name,
        add_dirs=[workdir],
        allowed_tools=f"Read,Glob,mcp__{server_name}",
        workdir=workdir,
        mcp_servers={server_name: server_cfg},
        expected_mcp_servers=[server_name],
    )
    parsed = extract_json_object(run.result_text) or extract_json_object(run.assistant_text) or extract_json_object(run.raw_stream)
    trace = {
        "return_code": run.return_code,
        "timed_out": run.timed_out,
        "latency_sec": round(run.latency_sec, 4),
        "command": run.command,
        "session_file": run.session_file,
        "parsed_ok": isinstance(parsed, dict),
    }
    write_json(workdir / "agent_trace.json", trace)
    return parsed if isinstance(parsed, dict) else None, trace


def sample_questions(
    config: ProjectConfig,
    *,
    sample_size: int,
    min_hops: int = 2,
    max_hops: int = 4,
    seed: int | None = None,
    sampling_mode: str = "dag_closure",
    partial_policy: str = "closure_required",
    edge_profile: str = "core_strict",
    max_repair_rounds: int = 2,
    sampling_profile_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if sample_size < 1 or min_hops < 1 or max_hops < min_hops:
        raise ValueError("invalid sample size or hop range")
    if partial_policy not in {"closure_required", "exclude"}:
        raise ValueError("partial_policy must be closure_required or exclude")
    if sampling_mode == "linear_debug":
        partial_policy = "exclude"

    run_dir = config.paths.run_dir
    sampling_cfg = _load_sampling_config(config)
    sqlite_path, manifest_path, kb_server_name = _science_kb_paths(config, sampling_cfg)
    kb = ScienceKB(sqlite_path, manifest_path)
    graph_rows, cards_rows, debug_rows = load_canonical_sampling_inputs(run_dir)
    cards = {str(x["tool_id"]): x for x in cards_rows if x.get("tool_id")}
    debug_idx = _edge_debug_index(debug_rows)
    edges = _filtered_edges(graph_rows, debug_idx, edge_profile, partial_policy, sampling_cfg)
    graph_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in edges:
        graph_index[(str(e["source_tool"]), str(e["target_tool"]))].append(e)
    adj = _adjacency(edges)

    anchor_policy = sampling_cfg.get("anchor_policy") or {}
    excluded_start = set(anchor_policy.get("excluded_start_tools") or [])
    excluded_terminal = set(anchor_policy.get("excluded_terminal_tools") or [])
    memo: dict[tuple[str, int], bool] = {}
    starts_by_hops = {
        k: sorted(n for n in adj if n not in excluded_start and _has_walk(n, k, adj, memo))
        for k in range(min_hops, max_hops + 1)
    }
    if not any(starts_by_hops.values()):
        raise RuntimeError("no feasible anchor starts")

    intermediate = run_dir / "intermediate" / "stage3"
    sample_workdir, results = intermediate / "workdir", run_dir / "results"
    sample_workdir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    runtime, rng, usage = ClaudeCodeRuntime(config), random.Random(seed), Counter()
    leak_patterns = _compile_leak_patterns(cards)
    profile_values = (sampling_profile_meta or {}).get("resolved_sampling_config") or {}
    proposal_path = Path(
        (sampling_profile_meta or {}).get("prompt_path")
        or config.paths.configs / "prompts/legacy/toolchain_question_sampler_v1.md"
    )
    repair_path = config.paths.configs / str(
        profile_values.get("repair_prompt")
        or "prompts/legacy/toolchain_question_repair_v1.md"
    )
    proposal_template = proposal_path.read_text(encoding="utf-8").strip()
    repair_template = repair_path.read_text(encoding="utf-8").strip()

    attempts, successes, closure_rows, grounding_rows = [], [], [], []
    for idx in tqdm(range(1, sample_size + 1), desc="sample-questions-v2", unit="sample"):
        sample_id = f"sample_{idx:04d}"
        hops = rng.randint(min_hops, max_hops)
        nodes, walk_edges_raw, walk_error = _walk(starts_by_hops[hops], hops, adj, rng, memo, usage, excluded_terminal)
        base = {
            "index": idx, "sample_id": sample_id, "task_type": "toolchain_derived_v2",
            "anchor_toolchain_nodes": nodes, "anchor_toolchain_edges": [_edge_public(x) for x in walk_edges_raw],
            "walk_hops": hops, "start_tool": nodes[0] if nodes else "", "end_tool": nodes[-1] if nodes else "",
            "status": "failed", "failure_reason": walk_error or "", "created_at_utc": _now_utc(),
        }
        if walk_error:
            attempts.append(base)
            continue

        neighbor_tools = set(nodes)
        # Give the agent a bounded one-hop neighborhood. The full graph remains
        # available to Python validators, but an agent should not wander across
        # hundreds of unrelated edges while repairing one workflow.
        context_edges_raw = [
            e for e in edges
            if e.get("source_tool") in nodes or e.get("target_tool") in nodes
        ]
        context_tools = set(neighbor_tools)
        for e in context_edges_raw:
            context_tools.update([str(e.get("source_tool")), str(e.get("target_tool"))])
        kg_context_edges = [_edge_public(e) for e in context_edges_raw]
        context_pairs = {
            (str(e.get("source_tool")), str(e.get("target_tool")), str(e.get("edge_type")))
            for e in context_edges_raw
        }
        relevant_debug = [
            _debug_public(d) for d in debug_rows
            if (str(d.get("source_tool")), str(d.get("target_tool")), str(d.get("edge_type"))) in context_pairs
        ]
        catalog = {tid: _card_public(cards[tid]) for tid in sorted(context_tools) if tid in cards}
        sample_root = sample_workdir / f"{sample_id}__{safe_name(nodes[0])}__{safe_name(nodes[-1])}"

        previous: dict[str, Any] | None = None
        final_feedback: list[str] = []
        accepted: dict[str, Any] | None = None
        accepted_closure: dict[str, Any] | None = None
        accepted_edges: list[tuple[str, str]] = []
        accepted_edge_types: dict[tuple[str, str], str] = {}
        for repair_round in range(max(0, max_repair_rounds) + 1):
            attempt_dir = sample_root / f"attempt_{repair_round:02d}"
            kb_trace = attempt_dir / "kb_query_trace.jsonl"
            files: dict[str, Any] = {
                "task_context.json": {
                    "task_type": "grounded_agent_workflow_proposal",
                    "anchor_is_inspiration_only": True,
                    "no_scientific_tool_execution": True,
                    "science_kb_mcp_server": kb_server_name,
                    "repair_round": repair_round,
                },
                "anchor_spec.json": {"nodes": nodes, "edges": [_edge_public(x) for x in walk_edges_raw]},
                "kg_context.json": {"edges": kg_context_edges},
                "tool_catalog.json": catalog,
                "edge_decision_context.json": {"edges": relevant_debug},
            }
            if previous is not None:
                files["previous_proposal.json"] = previous
                files["validation_feedback.json"] = {"errors": final_feedback}
            prompt = _build_prompt(repair_template if repair_round else proposal_template, repair_round)
            _prepare_attempt_workdir(config, attempt_dir, files, prompt)
            parsed, trace = _run_agent_attempt(
                runtime,
                attempt_dir,
                prompt,
                kb_server_name,
                _science_kb_mcp_server(sqlite_path, manifest_path, kb_trace),
            )
            if parsed is None:
                final_feedback = ["agent_output_parse_failed"]
                continue
            write_json(attempt_dir / "proposal.json", parsed)
            previous = parsed
            try:
                jsonschema.validate(parsed, QUESTION_SAMPLER_OUTPUT_SCHEMA)
            except jsonschema.ValidationError as exc:
                final_feedback = [f"response_schema_invalid:{exc.message}"]
                continue
            if parsed.get("status") == "reject":
                final_feedback = [f"agent_reject:{parsed.get('reject_reason') or ''}"]
                break

            tools = set(str(x) for x in parsed["workflow_proposal"]["tools"])
            proposed_edges = parsed["workflow_proposal"]["edges"]
            errors = [f"unknown_tool:{x}" for x in tools if x not in cards]
            if len(tools) < 2 or not proposed_edges:
                errors.append("workflow_requires_at_least_two_tools_and_one_edge")
            edge_errors, wf_edges, edge_types = _validate_edge_claims(tools, proposed_edges, graph_index)
            errors.extend(edge_errors)
            public_text, payload = str(parsed.get("public_question_text", "")), parsed.get("question_payload") or {}
            inputs, refs = parsed.get("grounded_initial_inputs") or [], parsed.get("grounding_refs") or []
            errors.extend(_validate_grounding(kb, inputs, refs, public_text, payload))
            errors.extend(_problem_text_errors(public_text, payload, leak_patterns, sampling_cfg))
            necessity = {str(x.get("tool_id")): bool(x.get("necessary")) for x in parsed.get("tool_necessity") or []}
            for tool in tools:
                if necessity.get(tool) is not True:
                    errors.append(f"tool_not_justified:{tool}")
            closure = _closure_analysis(tools, wf_edges, cards, inputs, payload, debug_idx, edge_types)
            if closure["closure_status"] != "closed":
                errors.append(f"closure_open:{[x['requirement_id'] for x in closure['open_requirements']]}")
            write_json(attempt_dir / "validation_feedback.json", {"errors": errors, "closure_report": closure})
            if errors:
                final_feedback = errors
                continue
            accepted, accepted_closure, accepted_edges, accepted_edge_types = parsed, closure, wf_edges, edge_types
            base["repair_rounds"] = repair_round
            break

        if accepted is None:
            base.update({
                "failure_reason": final_feedback[0] if final_feedback else "proposal_not_accepted",
                "validation_errors": final_feedback,
                "workdir": str(sample_root),
            })
            attempts.append(base)
            continue

        tools = set(str(x) for x in accepted["workflow_proposal"]["tools"])
        proposal_edges = accepted["workflow_proposal"]["edges"]
        materialized = _materialize_edges(proposal_edges, graph_index)
        trajectory = _trajectory(
            tools, accepted_edges, accepted["grounded_initial_inputs"],
            accepted["workflow_proposal"]["final_deliverable"], accepted["llm_message_intents"],
        )
        anchor_set = set(nodes)
        grounding_rows.append({"sample_id": sample_id, "grounding_refs": accepted["grounding_refs"], "grounded_initial_inputs": accepted["grounded_initial_inputs"]})
        closure_rows.append({"sample_id": sample_id, "closure_report": accepted_closure})
        row = {
            **base,
            "status": "success", "failure_reason": "",
            "public_question_text": accepted["public_question_text"],
            "question_payload": accepted["question_payload"],
            "toolchain_nodes": sorted(tools),
            "toolchain_edges": materialized,
            "expected_trajectory": trajectory,
            "grounded_initial_inputs": accepted["grounded_initial_inputs"],
            "grounding_refs": accepted["grounding_refs"],
            "workflow_support": proposal_edges,
            "scientific_task_rationale": accepted["scientific_task_rationale"],
            "closure_status": "closed",
            "grounding_count": len(accepted["grounding_refs"]),
            "grounding_sources": sorted({x.split("::", 1)[0] for x in accepted["grounding_refs"]}),
            "workflow_tool_count": len(tools),
            "anchor_tool_count": len(anchor_set),
            "added_tool_count": len(tools - anchor_set),
            "removed_anchor_tool_count": len(anchor_set - tools),
            "partial_edge_count": sum(x.get("edge_type") == "generates_partial_input_for" for x in materialized),
            "executability_status": "strict_pass",
            "workdir": str(sample_root),
        }
        attempts.append(row)
        successes.append(row)

    kb.close()
    write_jsonl(intermediate / "sample_attempts.jsonl", attempts)
    write_jsonl(intermediate / "input_closure_report.jsonl", closure_rows)
    write_jsonl(intermediate / "grounding_records.jsonl", grounding_rows)
    tasks = [canonical_task(row, run_dir.name) for row in successes]
    write_jsonl(results / "tasks.jsonl", tasks)

    failures = Counter(str(x.get("failure_reason") or "") for x in attempts if x.get("status") != "success")
    grounding_source_counts = Counter(
        source
        for row in successes
        for source in row.get("grounding_sources") or []
    )
    llm_role_counts = Counter(
        str(node.get("llm_role"))
        for row in successes
        for node in ((row.get("expected_trajectory") or {}).get("workflow_graph") or {}).get("nodes") or []
        if node.get("type") == "llm" and node.get("llm_role")
    )
    tool_usage_counts = Counter(tool for row in successes for tool in row.get("toolchain_nodes") or [])
    quality = {
        "run_id": run_dir.name, "attempt_count": len(attempts), "success_count": len(successes),
        "failure_count": len(attempts) - len(successes), "failure_breakdown": dict(failures),
        "strict_executability_pass_rate": len(successes) / len(attempts) if attempts else 0.0,
        "edge_profile": edge_profile, "partial_policy": partial_policy, "edge_pool_count": len(edges),
        "closure_status_distribution": {"closed": len(successes), "open_or_rejected": len(attempts) - len(successes)},
        "grounding_source_distribution": dict(grounding_source_counts),
        "llm_role_distribution": dict(llm_role_counts),
        "tool_usage_distribution": dict(tool_usage_counts),
        "partial_edge_count_in_successes": sum(int(x.get("partial_edge_count") or 0) for x in successes),
        "created_at_utc": _now_utc(),
    }
    write_json(results / "workflow_quality_report.json", quality)
    meta = {
        **quality, "sample_size_requested": sample_size, "seed": seed,
        "hop_range": {"min_hops": min_hops, "max_hops": max_hops},
        "science_kb_sqlite": str(sqlite_path), "science_kb_manifest": str(manifest_path),
        "tasks_path": str(results / "tasks.jsonl"), "intermediate_dir": str(intermediate),
        "sampling_profile": (sampling_profile_meta or {}).get("sampling_profile"),
        "resolved_sampling_config": (sampling_profile_meta or {}).get("resolved_sampling_config"),
        "config_sha256": (sampling_profile_meta or {}).get("config_sha256"),
        "prompt_sha256": (sampling_profile_meta or {}).get("prompt_sha256"),
    }
    write_json(intermediate / "sampling_meta.json", meta)
    update_manifest_tasks(run_dir, len(tasks), sampling_profile_meta)
    return meta
