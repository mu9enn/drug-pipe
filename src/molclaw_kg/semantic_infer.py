from __future__ import annotations

from typing import Any


def infer_format(slot_name: str, raw_type: str, description: str) -> str:
    text = f"{slot_name} {raw_type} {description}".lower()
    if "pdbqt" in text:
        return "pdbqt"
    if "pdb" in text:
        return "pdb"
    if "smi" in text or "smiles" in text:
        return "smiles"
    if "sdf" in text:
        return "sdf"
    if "csv" in text:
        return "csv"
    if "json" in text:
        return "json"
    if "png" in text or "image" in text:
        return "png"
    return "unknown"


def infer_semantic_type(slot_name: str, description: str, tool_name: str = "") -> str:
    text = f"{slot_name} {description} {tool_name}".lower()

    if "sequence" in text and "protein" in text:
        return "protein_sequence"
    if any(k in text for k in ["pdb_file", "pdb path", "protein_pdb", "structure"]):
        if "fix" in text or "repair" in text:
            return "repaired_pdb"
        return "protein_structure_pdb"
    if "chain" in text and "pdb" in text:
        return "pdb_chain_subset"
    if "smiles" in text:
        if any(k in text for k in ["list", "file", "inputs", "candidate"]):
            return "ligand_smiles_list"
        return "ligand_smiles"
    if any(k in text for k in ["pose", "docking_file", "pdbqt"]):
        return "docking_pose"
    if any(k in text for k in ["score", "affinity", "probability", "rank"]):
        if "admet" in text:
            return "admet_profile"
        if "affinity" in text:
            return "affinity_score"
        return "docking_score"
    if any(k in text for k in ["pocket_center", "pocket_size", "box"]):
        return "pocket_box"
    if any(k in text for k in ["admet", "tox", "cyp"]):
        return "admet_profile"
    if any(k in text for k in ["work_dir", "workspace", "md directory"]):
        return "mmpbsa_workdir"
    if any(k in text for k in ["result", "summary", "report"]):
        return "report_markdown"
    if any(k in text for k in ["image", "plot", "png"]):
        return "image_plot"
    if any(k in text for k in ["threshold", "cutoff", "parameter", "config", "size_x", "size_y", "size_z"]):
        return "config_param"

    return "unknown"


def infer_stage(tool_name: str, description: str) -> str:
    t = f"{tool_name} {description}".lower()
    if any(k in t for k in ["retrieve", "download", "lookup"]):
        return "acquisition_lookup"
    if any(k in t for k in ["is_valid", "mapper", "map residue", "resolve"]):
        return "entity_resolution"
    if any(k in t for k in ["fix_pdb", "protein_prep", "extract_and_save_chains", "pack_sidechains"]):
        return "protein_prep"
    if any(k in t for k in ["convert_smiles", "ligand", "smiles", "conformer"]):
        return "ligand_prep"
    if any(k in t for k in ["sampling", "reinvent", "libinvent", "linkinvent", "pepinvent", "chroma", "proteinmpnn"]):
        return "generation_editing"
    if any(k in t for k in ["docking", "predict", "openmm", "goca", "openawsem", "run_mmpbsa", "gmx_mmpbsa", "boltz", "equiscore", "hdock"]):
        return "simulation_prediction"
    if any(k in t for k in ["filter", "valid", "drug_chemistry", "common_fragments"]):
        return "filtering_selection"
    if any(k in t for k in ["rank", "score", "affinity", "similarity"]):
        return "ranking_scoring"
    if any(k in t for k in ["prolif", "analyze", "validate", "quality_metrics", "structural_geometry"]):
        return "validation_crosscheck"
    if any(k in t for k in ["visualize", "report", "summary", "base64"]):
        return "reporting_visualization"
    return "simulation_prediction"


def cardinality_from_schema(schema_prop: dict[str, Any]) -> str:
    t = schema_prop.get("type")
    if t == "array":
        return "list"
    if t == "object":
        return "map"
    if t in {"string", "number", "integer", "boolean"}:
        return "single"
    return "unknown"
