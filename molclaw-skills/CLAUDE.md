# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MolClaw** is an AI-guided computational drug discovery knowledge system. It is a hierarchical skill framework designed to guide Claude agents through drug discovery workflows — not a compiled software project. There are no standardized build, lint, or test commands. The repository primarily consists of structured Markdown (`SKILL.md`) files, with a small number of supporting Python, shell, and JSON files.

## Architecture

The system uses a three-level hierarchy that agents read top-down:

```
L3_methodology/   ← Read first: strategic principles (2 files)
L2_workflows/     ← Read second: step-by-step protocol for the task domain (14 files)
L1_tools/         ← Read on-demand: individual tool specifications (60 directories)
LR_research/      ← Read for research workflows (1 file)
auto-generated-.claude/skills/  ← Crystallized skills from prior runs; check skill-index.md
                         Currently contains only skill-index.md
```

The skill directories shown above live directly under this repository root.

### L3 — Methodology
`L3_methodology/`:
- `molclaw-drug-discovery-methodology.md` — core principles: task planning, tiered screening funnels, iterative optimization, multi-method consensus, QC checkpoints, and reporting standards
- `molclaw-drug-discovery-methodology-supplement-ch8.md` — advanced guidance on novel problem discovery and draft workflow authoring

### L2 — Workflows
Step-by-step protocols in `L2_workflows/`:
- `00-problem-discovery-and-feasibility-triage.md`
- `01-target-protein-preparation.md`
- `02-molecular-docking-screening.md`
- `03-molecular-property-analysis-filtering.md`
- `04-generative-molecular-design.md`
- `05-iterative-molecular-optimization.md`
- `06-binding-free-energy-calculation.md`
- `07-conformational-sampling-analysis.md`
- `08-post-docking-evaluation.md`
- `09-peptide-protein-binder-design.md`
- `10-protein-sequence-design-validation.md`
- `11-multi-target-selectivity-assessment.md`
- `12-skill-crystallization.md`
- `13-draft-workflow-authoring.md`

### LR — Research
Research workflows in `LR_research/workflows/`:
- `deep-research.md`

### L1 — Tools
Individual tool specs in `L1_tools/` (60 directories, each containing a `SKILL.md`). Some directories include supplementary files — see notes below. Full list:

**Molecular descriptors & properties:**
- `molclaw-mol-basic-metrics` — MW, formula, atom counts
- `molclaw-mol-hydrophobicity-metrics` — LogP, molar refractivity
- `molclaw-mol-hbond-metrics` — H-bond donors/acceptors
- `molclaw-mol-charge-metrics` — Gasteiger charges
- `molclaw-mol-complexity-metrics` — complexity descriptors
- `molclaw-mol-structure-metrics` — rotatable bonds, rings
- `molclaw-mol-topology-metrics` — TPSA, topological features
- `molclaw-mol-similarity` — molecular similarity metrics
- `molclaw-mol-opt-physchem` — physicochemical optimization
- `molclaw-drug-likeness` — QED, Lipinski rules
- `molclaw-admet` — 90+ ADMET endpoints (CYP inhibition, hERG, solubility, etc.)

**Docking & scoring:**
- `molclaw-quickvina-docking` — QuickVina2-GPU molecular docking
- `molclaw-diffdock-auto` — DiffDock blind docking *(skill documentation exists, but DiffDock is not enabled in the current 79-tool MCP deployment)*
- `molclaw-karmadock-tool` — KarmaDock large-scale docking
- `molclaw-docking-screening` — unified docking workflow
- `molclaw-equiscore-docking` — EquiScore rescoring (docking context)
- `molclaw-equiscore-tool` — EquiScore rescoring (standalone)
- `molclaw-boltz2-affinity` — Boltz-2 binding affinity prediction
- `molclaw-hdock-tool` — HDOCK protein-protein docking

**Protein structure prep & analysis:**
- `molclaw-protein-structure-retrieve` — retrieve structures from PDB
- `molclaw-protein-sequence-retrieve` — retrieve protein sequences
- `molclaw-pdbfixer` — PDB structure repair
- `molclaw-fix-pdb` — advanced PDB fixing
- `molclaw-foldx-tool` — FoldX-based protein structure analysis and mutation energy evaluation
- `molclaw-extract-chains` — chain extraction
- `molclaw-pulchura-rebuild` — structure rebuilding
- `molclaw-pack-sidechains` — sidechain packing
- `molclaw-fpocket` — fpocket binding-pocket detection
- `molclaw-fpocket-toolkit-base` — fpocket base toolkit
- `molclaw-p2rank` — P2Rank pocket prediction

**Interaction analysis (ProLIF & visualization):**
- `molclaw-prolif-docking` — interaction fingerprints for docking poses
- `molclaw-prolif-pdb` — interaction analysis on PDB structures
- `molclaw-prolif-md` — MD trajectory interaction fingerprints
- `molclaw-prolif-protein-protein` — protein-protein interaction analysis
- `molclaw-prolif-tool` — general ProLIF tool
- `molclaw-interaction-visualizer` — render interaction diagrams (PNG/SVG) from ProLIF results *(also contains `molclaw_interaction_visualizer.py` Python implementation)*

**MD simulation & free energy:**
- `molclaw-protein-openmm` — OpenMM MD simulations
- `molclaw-protein-ligand-mmpbsa` — MM-PBSA binding free energy (protein-ligand) *(also has `reference_fix_pdb.md`, `reference_prepare_complex.md`, `reference_run_mmpbsa.md`, `reference_analyze_mmpbsa.md`)*
- `molclaw-protein-protein-mmpbsa` — MM-PBSA binding free energy (protein-protein) *(also has `reference_fix_pdb.md`, `reference_prepare_protein_md.md`, `reference_gmx_mmpbsa_propro.md`, `reference_analyze_mmpbsa.md`)*

**Structure & sequence prediction / design:**
- `molclaw-chai1-predict` — Chai-1 structure prediction
- `molclaw-esmfold` — ESMFold structure prediction
- `molclaw-proteinmpnn-tool` — ProteinMPNN sequence design
- `molclaw-evobind-tool` — EvoBind de novo peptide design
- `molclaw-chroma-toolkit` — Chroma protein scaffold generation
- `molclaw-openawsem-tool` — OpenAWSEM protein folding
- `molclaw-goca-tool` — protein optimization
- `molclaw-run-bioemu` — biostructure emulation

**Generative molecular design:**
- `molclaw-denovo-sampling` — de novo molecule generation
- `molclaw-mol2mol-sampling` — Mol2Mol generative model
- `molclaw-peptide-sampling` — peptide generation/sampling
- `molclaw-rgroup-sampling` — R-group replacement
- `molclaw-linker-sampling` — linker design
- `molclaw-smiles-fg-editor` — functional group editing
- `molclaw-dleps` — DLEPS generative model

**Utilities:**
- `molclaw-compound-retrieve` — compound lookup
- `molclaw-smiles-valid-check` — SMILES validation
- `molclaw-sequence-valid-check` — sequence validation
- `molclaw-file-transfer` — file management
- `molclaw-scp-server` — SCP server integration
- `molclaw-skill-template-writer` — scaffold new SKILL.md files for tool documentation

## Root-Level Files

- **`CLAUDE.md`** — this file; project guidance for Claude Code
- **`system_prompt_FULL.md`** — full agent system prompt (approximately 28 KB); defines the 5-phase execution framework, data integrity rules, file naming conventions, and experiential learning configuration ("Full-EL" mode)

## Execution Framework

The agent execution framework is defined in `system_prompt_FULL.md`. It specifies:

1. **5-phase execution:** read skills → plan → self-check → execute → synthesize
2. **File naming conventions:** sequential (`step01_`, `step02_`), iterative (`round01_`, `round02_`), retry (`_retry1`)
3. **Required outputs:** `result.md` (final summary) and `run_log.md` (step-by-step log, written incrementally)

### Key enforcement rules from `system_prompt_FULL.md`

- **Data integrity:** Every number in `result.md` must be programmatically verified from source files before writing. Three mandatory checkpoints: after each tool call (A), before each round summary (B), and before final report (C).
- **Docking constraints:** Vina/QuickVina scores MUST be negative. Minimum box size is 25 Å per dimension. For the current QuickVina2-GPU deployment, progressive enlargement on failure is 25 → 30 → 40 → 47.625 Å before switching to the currently enabled KarmaDock service. DiffDock must not be selected unless it is re-enabled in the MCP deployment.
- **File collection:** ALL structure files (PDB, PDBQT, SDF, CIF) and visualization images (PNG, SVG) generated by tools MUST be downloaded to local workspace using `molclaw-file-transfer`. Verify with `ls -la` after each download.
- **Residue numbering:** When referencing specific residues, build an explicit mapping table between UniProt canonical, PDB author, and tool-internal numbering schemes. Check `DBREF` records in PDB files for the offset.
- **Literature values:** Must be labeled `⚠️ LITERATURE VALUE` and only used after exhausting all computational alternatives.

## Core Architectural Principles

- **Tiered screening funnel:** Physicochemical filtering → docking → rescoring → MD validation (increasing cost/accuracy)
- **Multi-method consensus:** Critical steps (pocket detection, docking scores) require ≥2 independent tools
- **Iterative optimization loop:** Each round defines what to improve, how, and how to measure; convergence criteria must be set upfront
- **QC checkpoints:** Positive docking affinity scores are failures; MW/formula consistency must be verified; uncertainty must be explicitly annotated
- **Tools are loaded on-demand** — read L1 tool SKILL.md files relevant to the current workflow step
