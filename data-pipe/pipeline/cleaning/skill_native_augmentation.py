from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline.cleaning.invariants import validate_semantic_record
from pipeline.cleaning.io import base_manifest, read_jsonl, write_json, write_jsonl, write_pretty_json


LOCAL_TOOLS = {"Read", "Write", "Edit", "Bash", "Grep", "Glob"}

# Canonical raw MolClaw tool name -> the L1 skill that documents it. This is a
# data-pipeline mapping only; it is deliberately not copied into model workdirs.
TOOL_TO_SKILL = {
    "pred_protein_structure_esmfold": "molclaw-esmfold",
    "pred_mol_admet": "molclaw-admet",
    "calculate_dleps_score": "molclaw-dleps",
    "molecule_docking_quickvina_fullprocess": "molclaw-quickvina-docking",
    "pred_pocket_prank": "molclaw-p2rank",
    "pred_binding_affinity_boltz2": "molclaw-boltz2-affinity",
    "is_valid_protein_sequence": "molclaw-sequence-valid-check",
    "is_valid_smiles": "molclaw-smiles-valid-check",
    "convert_smiles_to_format": "molclaw-equiscore-docking",
    "convert_pdb_to_pdbqt_dock": "molclaw-equiscore-docking",
    "convert_complex_cif_to_pdb": "molclaw-equiscore-docking",
    "retrieve_smiles_by_compoundname": "molclaw-compound-retrieve",
    "calculate_mol_basic_info": "molclaw-mol-basic-metrics",
    "calculate_mol_hydrophobicity": "molclaw-mol-hydrophobicity-metrics",
    "calculate_mol_hbond": "molclaw-mol-hbond-metrics",
    "calculate_mol_structure_complexity": "molclaw-mol-structure-metrics",
    "calculate_mol_topology": "molclaw-mol-topology-metrics",
    "calculate_mol_drug_chemistry": "molclaw-drug-likeness",
    "calculate_mol_charge": "molclaw-mol-charge-metrics",
    "calculate_mol_complexity": "molclaw-mol-complexity-metrics",
    "calculate_protein_sequence_properties": "molclaw-protein-metrics",
    "calculate_pdb_basic_info": "molclaw-protein-metrics",
    "calculate_pdb_structural_geometry": "molclaw-protein-metrics",
    "calculate_pdb_quality_metrics": "molclaw-protein-metrics",
    "calculate_pdb_composition_info": "molclaw-protein-metrics",
    "calculate_morgan_fingerprint_similarity": "molclaw-mol-similarity",
    "calculate_common_fragments": "molclaw-mol-similarity",
    "extract_and_save_chains": "molclaw-extract-chains",
    "extract_pdb_chains": "molclaw-extract-chains",
    "server_file_to_base64": "molclaw-file-transfer",
    "base64_to_server_file": "molclaw-file-transfer",
    "reinvent_mol2mol_sampling": "molclaw-mol2mol-sampling",
    "reinvent_denovo_sampling": "molclaw-denovo-sampling",
    "linkinvent_linker_sampling_by_warheads": "molclaw-linker-sampling",
    "linkinvent_linker_sampling_by_warhead_pair_name": "molclaw-linker-sampling",
    "libinvent_rgroup_sampling_by_scaffold": "molclaw-rgroup-sampling",
    "libinvent_rgroup_sampling_by_scaffold_name": "molclaw-rgroup-sampling",
    "get_pepinvent_info": "molclaw-peptide-sampling",
    "pepinvent_peptide_sampling_by_template": "molclaw-peptide-sampling",
    "pepinvent_peptide_sampling_by_peptide": "molclaw-peptide-sampling",
    "fix_pdb": "molclaw-fix-pdb",
    "fpocket_toolkit": "molclaw-fpocket",
    "equiscore_pocket": "molclaw-equiscore-tool",
    "equiscore_screen": "molclaw-equiscore-tool",
    "equiscore_pipeline": "molclaw-equiscore-tool",
    "prolif_md": "molclaw-prolif-tool",
    "prolif_docking": "molclaw-prolif-tool",
    "prolif_pdb": "molclaw-prolif-tool",
    "prolif_protein_protein": "molclaw-prolif-tool",
    "protein_openmm_md": "molclaw-protein-openmm",
    "run_bioemu": "molclaw-run-bioemu",
    "openawsem_sim": "molclaw-openawsem-tool",
    "run_mmpbsa": "molclaw-mmpbsa-toolkit",
    "gmx_mmpbsa_propro": "molclaw-mmpbsa-toolkit",
    "analyze_mmpbsa": "molclaw-mmpbsa-toolkit",
    "prepare_complex": "molclaw-mmpbsa-toolkit",
    "prepare_protein_md": "molclaw-mmpbsa-toolkit",
    "proteinmpnn_tool": "molclaw-proteinmpnn-tool",
    "evobind_tool": "molclaw-evobind-tool",
    "hdock_tool": "molclaw-hdock-tool",
    "chai1_predict": "molclaw-chai1-predict",
    "karmadock_tool": "molclaw-karmadock-tool",
    "chroma_monomer": "molclaw-chroma-toolkit",
    "chroma_complex": "molclaw-chroma-toolkit",
    "chroma_symmetry": "molclaw-chroma-toolkit",
    "goca_pipeline": "molclaw-goca-tool",
    "pulchura_rebuild": "molclaw-pulchura-rebuild",
    "openmm_extract_frames": "molclaw-protein-openmm",
    "extract_bioemu_structures": "molclaw-run-bioemu",
    "openawsem_traj_extract": "molclaw-openawsem-tool",
    "pack_sidechains": "molclaw-pack-sidechains",
    "retrieve_protein_structure_by_pdb_id": "molclaw-protein-structure-retrieve",
    "retrieve_protein_structure_by_uniprot_id": "molclaw-protein-structure-retrieve",
    "retrieve_protein_structure_by_gene_name": "molclaw-protein-structure-retrieve",
    "retrieve_protein_sequence": "molclaw-protein-sequence-retrieve",
    "visualize_protein": "molclaw-visualize-protein",
    "visualize_molecule": "molclaw-visualize-molecule",
    "analyze_protein_ligand_interactions": "molclaw-protein-ligand-interactions",
    "residue_mapper": "molclaw-residue-mapper",
    "interaction_visualizer": "molclaw-interaction-visualizer",
    "foldx_tool": "molclaw-foldx-tool",
}

OBSOLETE_SKILL_ALIASES = {
    "molclaw-pdbfixer": "molclaw-fix-pdb",
    "molclaw-fpocket-toolkit-base": "molclaw-fpocket",
    "molclaw-prolif-docking": "molclaw-prolif-tool",
    "molclaw-prolif-md": "molclaw-prolif-tool",
    "molclaw-prolif-pdb": "molclaw-prolif-tool",
    "molclaw-prolif-protein-protein": "molclaw-prolif-tool",
    "molclaw-protein-ligand-mmpbsa": "molclaw-mmpbsa-toolkit",
    "molclaw-protein-protein-mmpbsa": "molclaw-mmpbsa-toolkit",
    "molclaw-reinvent-mol2mol-sampling": "molclaw-mol2mol-sampling",
    "molclaw-mol-structure-complexity": "molclaw-mol-structure-metrics",
    "molclaw-mol-drug-chemistry": "molclaw-drug-likeness",
    "molclaw-mol-basic-info": "molclaw-mol-basic-metrics",
    "molclaw-mol-hydrophobicity": "molclaw-mol-hydrophobicity-metrics",
    "molclaw-mol-hbond": "molclaw-mol-hbond-metrics",
    "molclaw-mol-topology": "molclaw-mol-topology-metrics",
    "molclaw-pdb-basic-info": "molclaw-protein-metrics",
    "molclaw-structure-complexity-metrics": "molclaw-mol-structure-metrics",
    "molclaw-retrieve-protein-structure": "molclaw-protein-structure-retrieve",
    "molclaw-sequence-retrieve": "molclaw-protein-sequence-retrieve",
}

CANONICAL_SKILLS = frozenset(TOOL_TO_SKILL.values())
SKILL_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])molclaw-[a-z0-9][a-z0-9_-]*",
    re.IGNORECASE,
)


def _tool_aliases() -> dict[str, str]:
    """Return unambiguous historical names mechanically derived from tool names."""
    candidates: dict[str, set[str]] = {}
    for tool, skill in TOOL_TO_SKILL.items():
        forms = {tool, tool.replace("_", "-")}
        parts = tool.split("_")
        if parts[0] in {
            "analyze", "calculate", "get", "pred", "retrieve", "run",
            "reinvent", "linkinvent", "libinvent", "pepinvent",
        } and len(parts) > 1:
            stripped = "_".join(parts[1:])
            forms.update({stripped, stripped.replace("_", "-")})
        for form in forms:
            candidates.setdefault(f"molclaw-{form}", set()).add(skill)
    return {
        alias: next(iter(skills))
        for alias, skills in candidates.items()
        if len(skills) == 1 and alias not in CANONICAL_SKILLS
    }


SKILL_ALIASES = {**_tool_aliases(), **OBSOLETE_SKILL_ALIASES}
STALE_STANDALONE_SKILLS = frozenset({"molclaw-scp-server"})
NON_SKILL_IDENTIFIERS = frozenset({"molclaw-kg", "molclaw-tests", "molclaw-vs"})


def normalize_skill_references(
    value: Any,
    *,
    canonicalized: Counter[str] | None = None,
    neutralized: Counter[str] | None = None,
) -> Any:
    """Remove stale callable-looking skill identifiers from model-visible text."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            token_key = token.lower()
            if (
                token_key in CANONICAL_SKILLS
                or token_key == "molclaw-scp"
                or token_key in NON_SKILL_IDENTIFIERS
                or token_key.startswith("molclaw-scp-server_")
            ):
                return token
            replacement = SKILL_ALIASES.get(token_key)
            if replacement:
                if canonicalized is not None:
                    canonicalized[token_key] += 1
                return replacement
            before = value[max(0, match.start() - 120):match.start()].lower()
            after = value[match.end():match.end() + 8].lower()
            if not (
                token_key in STALE_STANDALONE_SKILLS
                or "skill" in before
                or after.startswith(".md")
            ):
                return token
            if neutralized is not None:
                neutralized[token_key] += 1
            label = token_key.removeprefix("molclaw-").replace("_", " ").replace("-", " ")
            return label

        return SKILL_IDENTIFIER_RE.sub(replace, value)
    if isinstance(value, list):
        return [
            normalize_skill_references(item, canonicalized=canonicalized, neutralized=neutralized)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: normalize_skill_references(item, canonicalized=canonicalized, neutralized=neutralized)
            for key, item in value.items()
        }
    return value

SKILL_PATH_RE = re.compile(
    r"(?:^|/)(?:L1_tools/)?(?P<skill>molclaw-[a-z0-9-]+)/SKILL\.md$"
)


def normalize_l1_paths(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(".claude/skills/L1_tools", ".agents/skills").replace(
            "skills/L1_tools", ".agents/skills"
        )
    if isinstance(value, list):
        return [normalize_l1_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_l1_paths(item) for key, item in value.items()}
    return value


def canonical_name(name: str) -> str:
    value = str(name or "").strip()
    return value.rsplit("__", 1)[-1] if value.startswith("mcp__") else value


def skill_from_read(call: dict[str, Any]) -> str | None:
    if canonical_name(str(call.get("name") or "")) != "Read" and str(call.get("name")) != "read":
        return None
    path = str((call.get("arguments") or {}).get("file_path") or "").replace("\\", "/")
    match = SKILL_PATH_RE.search(path)
    return match.group("skill") if match else None


def _skill_description(text: str) -> str:
    import yaml
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not match:
        raise ValueError("skill frontmatter is missing")
    return str(yaml.safe_load(match[1])["description"]).strip()


def _catalog_description(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= 500 else value[:497] + "..."


@lru_cache(maxsize=4)
def skill_catalog(skills_root: Path) -> tuple[dict[str, str], ...]:
    result = []
    for skill in sorted(set(TOOL_TO_SKILL.values())):
        path = skills_root / skill / "SKILL.md"
        if not path.is_file():
            raise ValueError(f"missing mapped L1 skill: {path}")
        result.append({"name": skill, "description": _catalog_description(_skill_description(path.read_text(encoding="utf-8")))})
    return tuple(result)


def render_catalog(catalog: tuple[dict[str, str], ...]) -> str:
    lines = [
        "<system-reminder>",
        "A skill is a reusable set of task-specific instructions. The following skills are available in this session:",
        "",
        "<available_skills>",
    ]
    for item in catalog:
        lines.append(f"- `{html.escape(item['name'], quote=False)}`: {html.escape(item['description'], quote=False)}")
    lines.extend([
        "</available_skills>",
        "",
        "If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.",
        "A user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.",
        "</system-reminder>",
    ])
    return "\n".join(lines)


def render_skill_result(skill: str, skills_root: Path) -> str:
    instructions = (skills_root / skill / "SKILL.md").read_text(encoding="utf-8")
    instructions = re.sub(r"^---\s*\n.*?\n---\s*\n", "", instructions, count=1, flags=re.S).strip()
    return (
        f'<skill_content name="{html.escape(skill)}">\n'
        "<skill_resources>\n"
        f"Base directory for this skill: .agents/skills/{skill}\n"
        "Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.\n"
        "</skill_resources>\n\n"
        "<skill_instructions>\n"
        f"{instructions.rstrip()}\n"
        "</skill_instructions>\n"
        "</skill_content>"
    )


def _is_catalog_discovery(call: dict[str, Any]) -> bool:
    tool = canonical_name(str(call.get("name") or "")).lower()
    if tool not in {"bash", "glob"}:
        return False
    command = str((call.get('arguments') or {}).get('command') or '').strip()
    if command in {'ls -la .claude', 'ls -la .claude/', 'find .claude -type d | head -20'}:
        return True
    arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False).lower()
    return ("skills/l1_tools" in arguments or ".agents/skills" in arguments) and "skill.md" not in arguments


def augment_record(record: dict[str, Any], skills_root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    output = copy.deepcopy(record)
    failed_ids = {e['source_tool_use_id'] for e in output.get('events', [])
                  if e['type'] == 'tool_observation' and (e.get('is_error') or e.get('status') == 'error')}
    loaded: set[str] = set()
    pending_skills: dict[str, str] = {}
    removed_call_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    inserted_skills = 0
    converted_l1_reads = 0
    removed_catalog_calls = 0
    removed_obsolete_skill_reads = 0
    removed_duplicate_skill_calls = 0
    canonicalized_skill_references: Counter[str] = Counter()
    neutralized_skill_references: Counter[str] = Counter()

    for event in output.get("events") or []:
        if event.get("type") == "tool_observation":
            call_id = str(event.get("source_tool_use_id") or "")
            if call_id in removed_call_ids:
                continue
            if call_id in failed_ids:
                events.append(event)
                continue
            event["content"] = normalize_skill_references(
                normalize_l1_paths(event.get("content")),
                canonicalized=canonicalized_skill_references,
                neutralized=neutralized_skill_references,
            )
            skill = pending_skills.pop(call_id, None)
            if skill:
                event["name"] = "skill"
                event["status"] = "success"
                event["is_error"] = False
                event["content"] = render_skill_result(skill, skills_root)
                loaded.add(skill)
            events.append(event)
            continue

        if event.get("type") != "assistant_decision":
            events.append(event)
            continue

        calls = event.get("tool_calls") or []
        event["reasoning"] = normalize_skill_references(
            normalize_l1_paths(event.get("reasoning", "")),
            canonicalized=canonicalized_skill_references,
            neutralized=neutralized_skill_references,
        )
        if event.get("final_answer") is not None:
            event["final_answer"] = normalize_skill_references(
                event["final_answer"],
                canonicalized=canonicalized_skill_references,
                neutralized=neutralized_skill_references,
            )
        kept_calls = []
        removed_here = 0
        for call in calls:
            if str(call.get('source_tool_use_id')) in failed_ids:
                kept_calls.append(call)
                continue
            if _is_catalog_discovery(call):
                removed_call_ids.add(str(call.get("source_tool_use_id") or ""))
                removed_catalog_calls += 1
                removed_here += 1
                continue
            if canonical_name(str(call.get("name") or "")) in LOCAL_TOOLS or str(call.get("name")) in {
                name.lower() for name in LOCAL_TOOLS
            }:
                call["arguments"] = normalize_l1_paths(call.get("arguments") or {})
            skill = skill_from_read(call)
            if skill:
                skill = OBSOLETE_SKILL_ALIASES.get(skill, skill)
            if skill and (skills_root / skill / "SKILL.md").is_file():
                if skill in loaded or skill in pending_skills.values():
                    removed_call_ids.add(str(call.get("source_tool_use_id") or ""))
                    removed_duplicate_skill_calls += 1
                    removed_here += 1
                    continue
                call["name"] = "skill"
                call["arguments"] = {"name": skill}
                pending_skills[str(call.get("source_tool_use_id") or "")] = skill
                converted_l1_reads += 1
            elif skill:
                removed_call_ids.add(str(call.get("source_tool_use_id") or ""))
                removed_obsolete_skill_reads += 1
                removed_here += 1
                continue
            elif str(call.get("name") or "") == "skill":
                skill = str((call.get("arguments") or {}).get("name") or "")
                skill = SKILL_ALIASES.get(skill, skill)
                if skill not in CANONICAL_SKILLS:
                    raise ValueError(f"{record.get('id')}: unavailable structured skill call: {skill!r}")
                call["arguments"] = {**(call.get("arguments") or {}), "name": skill}
                pending_skills[str(call.get("source_tool_use_id") or "")] = skill
            call["arguments"] = normalize_skill_references(
                call.get("arguments") or {},
                canonicalized=canonicalized_skill_references,
                neutralized=neutralized_skill_references,
            )
            kept_calls.append(call)
        calls = kept_calls
        event["tool_calls"] = calls
        if not calls and not event.get("final_answer") and removed_here:
            continue

        missing: list[str] = []
        first_tools: list[str] = []
        for call in calls:
            tool = canonical_name(str(call.get("name") or ""))
            skill = TOOL_TO_SKILL.get(tool)
            if skill and skill not in loaded and skill not in missing:
                skill_path = skills_root / skill / "SKILL.md"
                if not skill_path.is_file():
                    raise ValueError(f"{record.get('id')}: missing L1 skill for {tool}: {skill_path}")
                missing.append(skill)
                first_tools.append(tool)

        if missing:
            original_id = str(event.get("source_message_id") or "decision")
            digest = hashlib.sha256((str(record.get("id")) + "\0" + original_id).encode()).hexdigest()[:12]
            skill_calls = []
            for index, skill in enumerate(missing, 1):
                call_id = f"l1skill_{digest}_{index}"
                skill_calls.append({
                    "name": "skill",
                    "arguments": {"name": skill},
                    "source_tool_use_id": call_id,
                })
            skill_reasoning = (
                f"This step first uses {', '.join(first_tools)}. I will load the matching "
                f"L1 skill guidance ({', '.join(missing)}) before calling those tools."
            )
            original_reasoning = event.get("reasoning", "").strip()
            if not events and original_reasoning.startswith("High-level plan:\n"):
                skill_reasoning = original_reasoning + "\n\n" + skill_reasoning
                event["reasoning"] = ""
            events.append({
                "type": "assistant_decision",
                "source_message_id": f"{original_id}:l1",
                "reasoning": skill_reasoning,
                "tool_calls": skill_calls,
                "final_answer": None,
            })
            for call, skill in zip(skill_calls, missing):
                events.append({
                    "type": "tool_observation",
                    "name": "skill",
                    "source_tool_use_id": call["source_tool_use_id"],
                    "status": "success",
                    "is_error": False,
                    "content": render_skill_result(skill, skills_root),
                })
                loaded.add(skill)
                inserted_skills += 1
            prefix = f"I have loaded the required L1 skill guidance for {', '.join(first_tools)}."
            event["reasoning"] = prefix + (f"\n\n{event.get('reasoning', '').strip()}" if event.get("reasoning", "").strip() else "")

            retained_calls = []
            for call in event["tool_calls"]:
                if (call.get("source_tool_use_id") not in failed_ids and call.get("name") == "skill"
                        and str((call.get("arguments") or {}).get("name")) in loaded):
                    removed_call_ids.add(str(call.get("source_tool_use_id") or ""))
                    pending_skills.pop(str(call.get("source_tool_use_id") or ""), None)
                    removed_duplicate_skill_calls += 1
                else:
                    retained_calls.append(call)
            event["tool_calls"] = retained_calls

        events.append(event)

    output["events"] = events
    catalog = skill_catalog(skills_root)
    output.setdefault("metadata", {})["skill_native_augmentation"] = {
        "schema_version": "molclaw_skill_native_augmentation_v1",
        "inserted_skills": inserted_skills,
        "converted_l1_reads": converted_l1_reads,
        "removed_catalog_calls": removed_catalog_calls,
        "removed_obsolete_skill_reads": removed_obsolete_skill_reads,
        "removed_duplicate_skill_calls": removed_duplicate_skill_calls,
        "canonicalized_skill_references": dict(sorted(canonicalized_skill_references.items())),
        "neutralized_unavailable_skill_references": dict(sorted(neutralized_skill_references.items())),
        "catalog": list(catalog),
        "catalog_message": render_catalog(catalog),
    }
    validation = validate_semantic_record(output)
    if not validation["ok"]:
        raise ValueError(f"{record.get('id')}: augmented semantic validation failed: {validation['errors']}")
    return output, {
        "inserted_skills": inserted_skills,
        "converted_l1_reads": converted_l1_reads,
        "removed_catalog_calls": removed_catalog_calls,
        "removed_obsolete_skill_reads": removed_obsolete_skill_reads,
        "removed_duplicate_skill_calls": removed_duplicate_skill_calls,
        "canonicalized_skill_references": sum(canonicalized_skill_references.values()),
        "neutralized_unavailable_skill_references": sum(neutralized_skill_references.values()),
    }


def augment_file(input_path: Path, output_root: Path, skills_root: Path) -> dict[str, Any]:
    records, parse_errors = read_jsonl(input_path)
    if parse_errors:
        raise ValueError(f"invalid semantic JSONL: {parse_errors}")
    augmented = []
    audits = []
    totals = {
        "inserted_skills": 0,
        "converted_l1_reads": 0,
        "removed_catalog_calls": 0,
        "removed_obsolete_skill_reads": 0,
        "removed_duplicate_skill_calls": 0,
        "canonicalized_skill_references": 0,
        "neutralized_unavailable_skill_references": 0,
    }
    for record in records:
        result, stats = augment_record(record, skills_root)
        augmented.append(result)
        audits.append({"id": record.get("id"), **stats})
        for key in totals:
            totals[key] += stats[key]
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "semantic_trajectories.jsonl"
    write_jsonl(output_path, augmented)
    write_pretty_json(output_root / "semantic_trajectories.pretty.json", augmented)
    audit_path = output_root / "augmentation_audit.jsonl"
    write_jsonl(audit_path, audits)
    manifest = {
        **base_manifest(step="molclaw_skill_native_augmentation", source=input_path, repo_root=Path(__file__).resolve().parents[3]),
        "input_count": len(records),
        "output_count": len(augmented),
        "skills_root": str(skills_root.resolve()),
        "tool_to_skill_count": len(TOOL_TO_SKILL),
        **totals,
        "output": str(output_path.resolve()),
        "audit": str(audit_path.resolve()),
    }
    write_json(output_root / "augmentation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert first-use MolClaw L1 guidance to native skill calls.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--skills-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(augment_file(args.input.resolve(), args.output_root.resolve(), args.skills_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
