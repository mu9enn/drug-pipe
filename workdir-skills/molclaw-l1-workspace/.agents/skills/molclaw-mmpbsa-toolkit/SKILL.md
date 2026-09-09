---
name: molclaw-mmpbsa-toolkit
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__analyze_mmpbsa`, `mcp__molclaw-scp__gmx_mmpbsa_propro`, `mcp__molclaw-scp__prepare_complex`, `mcp__molclaw-scp__prepare_protein_md`, `mcp__molclaw-scp__run_mmpbsa`. Prepare, run, and analyze protein-ligand or protein-protein MM/GB(PB)SA calculations."
license: MIT license
metadata:
    skill-author: PJLab
---

# MM/GB(PB)SA Toolkit

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. All input paths must be server-visible paths returned by earlier MolClaw tools.

## Protein-ligand path

1. `prepare_complex` consumes `protein` and `ligand` and returns an MD workspace in `output_dir`.
2. `run_mmpbsa` consumes that directory as `work_dir`; choose `method` from `gb`, `pb`, or `both`.
3. `analyze_mmpbsa` consumes the calculation output as `work_dir` and generates reports.

## Protein-protein path

1. `prepare_protein_md` consumes a protein-only complex through `protein_pdb` and returns `run_dir`.
2. `gmx_mmpbsa_propro` consumes that directory as `work_dir`; choose `method` from `gb`, `pb`, or `both`.
3. `analyze_mmpbsa` consumes the calculation output as `work_dir`.

Check each call's `status`, preserve returned directories exactly, and do not treat dry-run outputs as scientific results. Detailed protein-ligand and protein-protein handoff and recovery notes are retained under this skill's `references/` directory; read only the relevant files when needed.
