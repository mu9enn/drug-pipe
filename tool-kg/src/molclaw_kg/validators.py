from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import TRANSITION_EDGE_TYPES
from .io_utils import read_jsonl, write_json, write_jsonl
from .models import ValidationResult
from .relation_utils import context_from_legacy_fields, normalize_edge_types, normalize_relation_status
from .settings import ProjectConfig
from .stage_taxonomy import load_stage_taxonomy


POSITIVE_RELATIONS = {"valid"}


def _collect_evidence_refs(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for x in (row.get("evidence_refs") or []):
        if isinstance(x, str) and x.strip():
            refs.append(x.strip())
    for et in row.get("edge_types", []):
        if not isinstance(et, dict):
            continue
        for x in (et.get("evidence_ids") or []):
            if isinstance(x, str) and x.strip():
                refs.append(x.strip())
    out = []
    for r in refs:
        if r not in out:
            out.append(r)
    return out


def _is_canonical_ref(ref: str) -> bool:
    if ref.startswith("snapshot::"):
        return True
    if ref.startswith("alternative_cluster::"):
        return True
    if ref.startswith("mcp::"):
        return True
    if "/.claude/skills/" in ref or ref.startswith(".claude/skills/"):
        return True
    if "/skills_full/" in ref or ref.startswith("skills_full/"):
        return True
    return False


def _is_derived_only_ref(ref: str) -> bool:
    bad_tokens = [
        "pair_spec.json",
        "task_context.json",
        "pair_payload.json",
        "doc_context.jsonl",
        "tool_A_card.json",
        "tool_B_card.json",
        "deterministic_base_tool_card.json",
    ]
    return any(t in ref for t in bad_tokens)


def validate_pairs(config: ProjectConfig) -> dict[str, Any]:
    pairs = read_jsonl(config.paths.run_dir / "candidate_pairs.jsonl")
    adjud = read_jsonl(config.paths.run_dir / "pair_adjudications.jsonl")

    pair_map = {p["pair_id"]: p for p in pairs}

    taxonomy = load_stage_taxonomy(config.stage_taxonomy_path)
    reporting_stages = {
        s
        for s in taxonomy.allowed_stages()
        if any(tok in s for tok in ["report", "visualization", "file_export"])
    }
    require_direct = bool(config.rules.get("validator", {}).get("require_directness_for_transition_edge", True))

    results: list[ValidationResult] = []
    adjud_validated = []

    for row in adjud:
        pid = row["pair_id"]
        pair = pair_map.get(pid, {})
        rel = normalize_relation_status(row.get("relation_status", row.get("decision")))
        edge_types = normalize_edge_types(
            row.get("edge_types", []),
            rel,
            legacy_decision=row.get("decision"),
            legacy_coverage_level=row.get("coverage_level"),
        )
        src = pair.get("source_tool", row.get("source_tool"))
        tgt = pair.get("target_tool", row.get("target_tool"))
        src_stage = pair.get("source_stage", "") or row.get("source_stage", "")
        tgt_stage = pair.get("target_stage", "") or row.get("target_stage", "")

        # ShapeEdgeValidator
        shape_status = "pass"
        shape_msg = "ok"
        if src == tgt:
            shape_status = "fail"
            shape_msg = "self loop is not allowed"
        if rel in POSITIVE_RELATIONS | {"alternative"} and not edge_types:
            shape_status = "fail"
            shape_msg = "missing edge_types for valid/alternative relation"
        results.append(
            ValidationResult(
                edge_id=f"{pid}::shape",
                pair_id=pid,
                validator_name="ShapeEdgeValidator",
                status=shape_status,  # type: ignore[arg-type]
                message=shape_msg,
            )
        )

        # TypedIOValidator
        schema_score = float(pair.get("schema_score", 0.0))
        typed_status = "pass"
        typed_msg = f"schema_score={schema_score:.3f}"
        if rel == "valid":
            transition_scopes = [
                str(e.get("support_scope", "n/a"))
                for e in edge_types
                if isinstance(e, dict) and e.get("type") in TRANSITION_EDGE_TYPES
            ]
            has_full = any(s == "full" for s in transition_scopes)
            has_partial = any(s == "partial" for s in transition_scopes)
            if has_full and schema_score < 0.20:
                typed_status = "fail"
                typed_msg = "valid(full) with too low schema_score"
            elif has_full and schema_score < 0.35:
                typed_status = "weak"
                typed_msg = "valid(full) with low schema_score"
            elif has_partial and schema_score < 0.15:
                typed_status = "weak"
                typed_msg = "valid(partial) with low schema_score"
        results.append(
            ValidationResult(
                edge_id=f"{pid}::typed_io",
                pair_id=pid,
                validator_name="TypedIOValidator",
                status=typed_status,  # type: ignore[arg-type]
                message=typed_msg,
            )
        )

        # DirectnessValidator
        direct_status = "pass"
        direct_msg = "ok"
        if require_direct and rel in POSITIVE_RELATIONS:
            if not row.get("direct_transition", False):
                direct_status = "fail"
                direct_msg = "valid edge must be direct_transition=true"
            else:
                all_rel = {e.get("type") for e in edge_types if isinstance(e, dict)}
                if all_rel and all(x not in TRANSITION_EDGE_TYPES for x in all_rel):
                    direct_status = "fail"
                    direct_msg = "valid transition lacks transition edge type"
        results.append(
            ValidationResult(
                edge_id=f"{pid}::directness",
                pair_id=pid,
                validator_name="DirectnessValidator",
                status=direct_status,  # type: ignore[arg-type]
                message=direct_msg,
            )
        )

        # StageValidator
        stage_status = "pass"
        stage_msg = "ok"
        if rel in POSITIVE_RELATIONS:
            if src_stage == tgt_stage:
                stage_status = "fail"
                stage_msg = f"same-stage transition not allowed: {src_stage}"
            elif not taxonomy.is_transition_allowed(str(src_stage), str(tgt_stage)):
                stage_status = "fail"
                stage_msg = f"stage transition not allowed: {src_stage}->{tgt_stage}"
            elif src_stage in reporting_stages and tgt_stage not in reporting_stages:
                stage_status = "fail"
                stage_msg = "reporting/export stage should not produce new transition"
        results.append(
            ValidationResult(
                edge_id=f"{pid}::stage",
                pair_id=pid,
                validator_name="StageValidator",
                status=stage_status,  # type: ignore[arg-type]
                message=stage_msg,
            )
        )

        # EvidenceSourceValidator
        ev_status = "pass"
        ev_msg = "ok"
        refs = _collect_evidence_refs(row)
        has_canonical = any(_is_canonical_ref(r) for r in refs)
        only_derived = bool(refs) and all(_is_derived_only_ref(r) for r in refs)

        if rel in POSITIVE_RELATIONS:
            if not refs:
                ev_status = "fail"
                ev_msg = "missing evidence_refs for valid relation"
            elif only_derived or not has_canonical:
                ev_status = "fail"
                ev_msg = "valid relation cites only non-canonical derived evidence"
        elif rel == "alternative":
            if not refs:
                ev_status = "weak"
                ev_msg = "alternative relation without explicit evidence refs"
        results.append(
            ValidationResult(
                edge_id=f"{pid}::evidence_source",
                pair_id=pid,
                validator_name="EvidenceSourceValidator",
                status=ev_status,  # type: ignore[arg-type]
                message=ev_msg,
                details={"evidence_refs": refs},
            )
        )

        # ConflictResolver
        conflict_status = "pass"
        conflict_msg = "ok"
        types = {e.get("type") for e in edge_types if isinstance(e, dict)}
        if "alternative_to" in types and any(t in TRANSITION_EDGE_TYPES for t in types):
            conflict_status = "weak"
            conflict_msg = "alternative and transition types coexist"
        if rel == "negative" and any(t in TRANSITION_EDGE_TYPES for t in types):
            conflict_status = "fail"
            conflict_msg = "negative relation cannot carry transition edge type"
        if rel == "alternative" and row.get("direct_transition", False):
            conflict_status = "fail"
            conflict_msg = "alternative relation must have direct_transition=false"
        results.append(
            ValidationResult(
                edge_id=f"{pid}::conflict",
                pair_id=pid,
                validator_name="ConflictResolver",
                status=conflict_status,  # type: ignore[arg-type]
                message=conflict_msg,
            )
        )
        normalized = dict(row)
        normalized["relation_status"] = rel
        normalized["edge_types"] = edge_types
        normalized["context"] = str(normalized.get("context") or context_from_legacy_fields(normalized))
        normalized.pop("coverage_level", None)
        normalized.pop("requires_additional_context", None)
        normalized.pop("single_source_executable", None)
        adjud_validated.append(normalized)

    rows = [r.model_dump() for r in results]
    out_results = config.paths.run_dir / "validation_results.jsonl"
    out_validated = config.paths.run_dir / "pair_adjudications_validated.jsonl"
    write_jsonl(out_results, rows)
    write_jsonl(out_validated, adjud_validated)

    by_validator = defaultdict(lambda: defaultdict(int))
    fail_pairs = set()
    for r in rows:
        by_validator[r["validator_name"]][r["status"]] += 1
        if r["status"] == "fail":
            fail_pairs.add(r["pair_id"])

    summary = {
        "result_count": len(rows),
        "pair_count": len(adjud),
        "pair_fail_count": len(fail_pairs),
        "validator_status_count": {v: dict(sorted(s.items())) for v, s in sorted(by_validator.items())},
        "output_results": str(out_results),
    }
    write_json(config.paths.run_dir / "validation_meta.json", summary)
    return summary
