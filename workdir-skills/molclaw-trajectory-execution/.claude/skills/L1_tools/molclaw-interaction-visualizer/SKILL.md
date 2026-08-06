---
name: molclaw-interaction-visualizer
description: >
  **PRIMARY tool for all single-structure interaction analysis.** MCP-exposed protein–ligand /
  peptide / protein–protein interaction analysis and Schrödinger-style multi-dimensional
  visualization. Pure Python/NumPy engine covering 9 interaction types with 2D diagram,
  3D PyMOL rendering, residue bar, interface heatmap, interface network, and decision-ready
  CSV/JSON export. Always use this tool first; fall back to ProLIF MCP only for batch
  docking fingerprint comparison or MD trajectory analysis.
license: MIT license
metadata:
    skill-author: PJLab
    skill-level: L1-Tool
    version: 1.0
    methodology-ref: >
      L3 Principle 15 (Mandatory image collection — all PNGs must be downloaded),
      L3 Principle 17 (Residue numbering reconciliation — use --resid_offset),
      L3 Principle 13 (Computation-first — interaction conclusions must come from tool output)
---

# Interaction Visualizer — MCP and Local Analysis

Note: 
- Local files are not directly accessible by the server. Please upload them to the server using `molclaw-file-transfer` before execution. 
- For PDB file inputs, it is recommended to preprocess them using `molclaw-pdbfixer` before execution.
- Please refer to skill `molclaw-scp-server` to complete tool invocation.

> [!NOTE]
> The current MolClaw server exposes this capability as the MCP tool
> `interaction_visualizer`. The bundled `molclaw_interaction_visualizer.py` remains
> available as an optional local CLI; it does not need to be uploaded to the server.

## When To Use This Skill

> **This is the PRIMARY tool for all single-structure interaction analysis.**
> ProLIF is only needed for batch docking fingerprint comparison or MD trajectory dynamics.

| Scenario | Use this skill | Use ProLIF instead |
|----------|:--------------:|:------------------:|
| **Single complex structure interaction analysis** | ✅ **PRIMARY** | Only if this tool unavailable |
| **Peptide or protein-protein interface analysis (single structure)** | ✅ **PRIMARY** | Only if this tool unavailable |
| Need Schrödinger-style 2D interaction diagram | ✅ | ❌ |
| Need PyMOL 3D multi-angle renderings | ✅ | ❌ |
| Need residue role annotations (Hinge/Gatekeeper/DFG) | ✅ | ❌ |
| Need decision-ready JSON with top residues + hot sites | ✅ | ❌ |
| Need `partner_site.csv` for ligand atom modification diagnosis | ✅ | ❌ |
| Need interaction fingerprint across MD trajectory frames | ❌ | ✅ `prolif_md` |
| Need batch docking pose fingerprint comparison (≥ 2 poses) | ❌ | ✅ `prolif_docking` |
| Need protein-protein trajectory interface profiling | ❌ | ✅ `prolif_protein_protein` |

## Interaction Types Detected

| # | Type | Key Geometry | Default Cutoff |
|---|------|-------------|----------------|
| 1 | Hydrogen bond | D-H···A angle ≥ 130° | 3.5 Å |
| 2 | Hydrophobic contact | C···C distance | 4.5 Å |
| 3 | π-π stacking (Face-to-Face) | Ring plane ∠ ≤ 35° | 5.5 Å |
| 4 | π-π stacking (Edge-to-Face) | Ring plane ∠ 50°–90° | 6.5 Å |
| 5 | Salt bridge | Charge center distance | 5.5 Å |
| 6 | Cation-π | Cation-centroid + normal ∠ | 6.0 Å |
| 7 | Halogen bond | X···A with A-X-D geometry | 3.5 Å |
| 8 | Metal coordination | Metal···coordinating atom | 3.0 Å |
| 9 | van der Waals | Sum of VdW radii + tolerance | +0.5 Å |

## Three Analysis Modes

- **`ligand`** (default): Small-molecule ligand vs protein. Partner identified by HETATM
  records or `--ligand_resname`. Produces 2D diagram + residue bar + partner site CSV.
- **`peptide`**: One protein chain as partner peptide. Requires `--partner_chain`.
  Produces interface heatmap + interface network + residue bar.
- **`protein`**: Two full protein chains. Requires `--chain_a` and `--chain_b`.
  Produces interface heatmap + interface network + residue bar.

## Input Source Mapping

| Parameter | Source Guidance |
|-----------|----------------|
| `--complex` | Pre-merged complex PDB from docking output, Boltz-2/Chai-1 predicted structure, MD extracted frame, or crystal structure |
| `--receptor` | Protein PDB from `molclaw-protein-structure-retrieve` → `molclaw-pdbfixer`, or predicted structure |
| `--ligand` | Docking pose file from `molclaw-quickvina-docking` (.pdbqt), `molclaw-karmadock-tool` (.sdf), an externally supplied DiffDock SDF, or any .mol/.mol2/.xyz |
| `--partner_pdb` | Partner protein PDB for peptide/protein modes (from `molclaw-extract-chains` or separate structure) |
| `--resid_offset` | PDB→UniProt offset computed from sequence alignment (L3 Principle 17) |
| `--residue_roles_json` | User-provided or literature-derived residue functional annotations |
| `--score` | Docking affinity from upstream docking tool (kcal/mol) |
| `--smiles` | Ligand SMILES from user input or molecule retrieval |

## MCP Invocation

```python
response = await client.session.call_tool(
    "interaction_visualizer",
    arguments={
        "mode": "ligand",
        "receptor_path": "/server/path/receptor_fixed.pdb",
        "ligand_path": "/server/path/docking_pose.pdbqt",
        "resid_offset": 0,
        "score": -8.3,
        "skip_pymol3d": True
    }
)
result = client.parse_result(response)
output_dir = result["output_dir"]
key_files = result["key_files"]
```

`mode` is required and must be one of `ligand`, `peptide`, or `protein`. Input paths
must be server-side paths; use `molclaw-file-transfer` for local input files.

## Bundled Local CLI Usage Patterns

### Pattern 1: Single Complex File (Ligand Mode)

For a pre-merged complex PDB containing both protein and ligand:

```bash
python molclaw_interaction_visualizer.py \
    --complex complex.pdb \
    --mode ligand \
    --ligand_resname LIG \
    --out_dir viz_out \
    --resid_offset 0 \
    --title "EGFR–Erlotinib" \
    --score -8.3 \
    --smiles "C=Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
```

### Pattern 2: Separate Receptor + Ligand Files (Auto-Merge)

When receptor and ligand come from different upstream tools (e.g., PDBFixer output + docking pose PDBQT/SDF). The script automatically converts the ligand format and merges into a single complex.

```bash
python molclaw_interaction_visualizer.py \
    --receptor receptor_fixed.pdb \
    --ligand docking_pose_best.sdf \
    --mode ligand \
    --out_dir viz_out \
    --resid_offset 574 \
    --residue_roles_json roles.json \
    --title "CDK2–Compound_7" \
    --score -9.1 \
    --delta_score -1.2
```

Supported ligand formats for auto-merge: `.sdf`, `.mol`, `.mol2`, `.pdb`, `.pdbqt`, `.xyz`.

### Pattern 3: Peptide–Protein Interface

```bash
python molclaw_interaction_visualizer.py \
    --complex complex.pdb \
    --mode peptide \
    --partner_chain B \
    --out_dir viz_out \
    --title "PD1–PeptideBinder"
```

### Pattern 4: Protein–Protein Interface (Separate Files)

```bash
python molclaw_interaction_visualizer.py \
    --receptor chain_A.pdb \
    --partner_pdb chain_B.pdb \
    --mode protein \
    --chain_a A \
    --chain_b B \
    --out_dir viz_out \
    --title "IL6–IL6R interface"
```

### Pattern 5: Quick CSV-Only (Skip All Plots)

For agent-only consumption when visualizations are not needed:

```bash
python molclaw_interaction_visualizer.py \
    --complex complex.pdb \
    --mode ligand \
    --ligand_resname LIG \
    --out_dir viz_out \
    --skip_diagram2d --skip_bar --skip_heatmap --skip_network --skip_pymol3d
```

## Bundled Local CLI Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--complex` | str | — | Single PDB/PDBQT file containing receptor + partner |
| `--receptor` | str | — | Receptor-only file (used with `--ligand` or `--partner_pdb`; alternative to `--complex`) |
| `--ligand` | str | — | Ligand-only file (.sdf/.mol/.mol2/.pdb/.pdbqt/.xyz); used with `--receptor` in ligand mode |
| `--partner_pdb` | str | — | Partner protein PDB for peptide/protein modes when receptor and partner are separate files |
| `--merged_out` | str | `OUT_DIR/auto_merged.pdb` | Path to save the auto-merged complex (when using --receptor + --ligand/--partner_pdb) |
| `--mode` | str | `ligand` | Analysis mode: `ligand`, `peptide`, or `protein` |
| `--ligand_resname` | str | — | (ligand mode) Restrict partner to a specific HETATM residue name |
| `--partner_chain` | str | — | (peptide mode) Chain ID of the peptide partner |
| `--chain_a` | str | — | (protein mode) Receptor chain ID |
| `--chain_b` | str | — | (protein mode) Partner chain ID |
| `--out_dir` | str | `viz_out` | Output directory (created if missing) |
| `--resid_offset` | int | `0` | Added to receptor resid in CSV output for PDB→UniProt mapping (L3 Principle 17) |
| `--title` | str | `""` | Common title prefix for all figures |
| `--score` | float | — | Docking score (kcal/mol) displayed in 2D diagram header |
| `--delta_score` | float | — | ΔScore vs baseline displayed in 2D diagram header |
| `--smiles` | str | — | Ligand SMILES displayed in 2D diagram footer |
| `--residue_roles_json` | str | — | JSON file mapping residues to functional roles: `{"MET769":"Hinge", "THR766":"Gatekeeper", ...}` |
| `--ligand_scale` | float | `1.0` | Scale factor for ligand drawing in 2D diagram (0.8=smaller, 1.3=larger) |
| `--top_n_bar` | int | `20` | Number of residues shown in stacked bar chart |
| `--heatmap_cutoff` | float | `4.5` | Atom-pair distance cutoff (Å) for interface heatmap |
| `--skip_diagram2d` | flag | — | Skip 2D interaction diagram generation |
| `--skip_bar` | flag | — | Skip residue stacked bar chart |
| `--skip_heatmap` | flag | — | Skip interface heatmap (peptide/protein modes) |
| `--skip_network` | flag | — | Skip interface network diagram (peptide/protein modes) |
| `--skip_csv` | flag | — | Skip CSV export |
| `--skip_pymol3d` | flag | — | Skip PyMOL 3D rendering |
| `--pymol_width` | int | `1400` | Width of PyMOL ray-traced images |
| `--pymol_height` | int | `1200` | Height of PyMOL ray-traced images |

## Output Files and Download Policy

### Ligand Mode Outputs

| Output File | Format | Download Policy | Consumed By |
|-------------|--------|-----------------|-------------|
| `interactions_{label}.csv` | CSV | **A — MUST download** | L2-08 consensus ranking, L2-05 optimization diagnosis |
| `interactions_{label}_residue_summary.csv` | CSV | **A — MUST download** | Residue-level SAR analysis |
| `interactions_{label}_partner_site.csv` | CSV | **A — MUST download** | Ligand modification hotspot identification |
| `diagram2d_{label}.png` | PNG | **A — MUST download** (L3 P15) | Report, user presentation |
| `residue_bar_{label}.png` | PNG | **A — MUST download** (L3 P15) | Report |
| `scene_{label}.pml` | PML script | B — record path in log | PyMOL manual inspection |
| `pymol_{label}_{front,side,top}.png` | PNG | **A — MUST download** (if generated) | Report 3D views |
| `summary_{label}.json` | JSON | **A — MUST download** | Agent decision loop integration |

### Peptide / Protein Mode Additional Outputs

| Output File | Format | Download Policy | Consumed By |
|-------------|--------|-----------------|-------------|
| `interface_heatmap_{label}.png` | PNG | **A — MUST download** (L3 P15) | Interface quality evaluation |
| `interface_network_{label}.png` | PNG | **A — MUST download** (L3 P15) | Interface topology analysis |

## CSV Column Specification

### `interactions_{label}.csv` — Per-Interaction Detail

| Column | Description |
|--------|-------------|
| `type` | Normalized interaction type (HBond, SaltBridge, PiStacking, CationPi, Halogen, Metal, Hydrophobic, VdW) |
| `subtype` | Direction/variant (LigDonor, ProtDonor, FaceToFace, EdgeToFace, etc.) |
| `rec_res` | Receptor residue label (e.g., MET769A) |
| `rec_resid_pdb` | Receptor residue number in PDB numbering |
| `rec_resid_mapped` | Receptor residue number after applying `--resid_offset` (for UniProt mapping) |
| `rec_res_class` | Residue classification (hydrophobic, aromatic, polar, positive, negative) |
| `rec_atom` | Receptor atom name(s) involved |
| `partner_atom` | Partner atom name(s) involved |
| `distance_A` | Interaction distance in Ångströms |
| `angle_deg` | Relevant geometry angle in degrees (if applicable) |
| `strength_hint` | Qualitative label: strong / moderate / weak / packing / contact |

### `*_residue_summary.csv` — Per-Residue Rollup

One row per contacting residue, columns for each interaction type count, total count, and minimum distance.

### `*_partner_site.csv` — Ligand Atom Hotspot (Ligand Mode Only)

One row per ligand atom involved in interactions, columns for each interaction type count. Use this to identify:
- **Atoms to preserve:** high total count → critical pharmacophore points
- **Atoms to modify:** low or zero count → optimization candidates

## Interpreting `summary_{label}.json` (Agent Decision Integration)

```json
{
  "mode": "ligand",
  "label": "LIG",
  "n_interactions": 42,
  "n_contact_residues": 15,
  "interaction_type_counts": {
    "HBond": 3, "SaltBridge": 1, "PiStacking": 2,
    "CationPi": 0, "Halogen": 0, "Metal": 0,
    "Hydrophobic": 8, "VdW": 15
  },
  "top_residues": [
    {"res": "MET769A", "class": "hydrophobic", "total": 5,
     "by_type": {"HBond": 2, "Hydrophobic": 3}},
    {"res": "LEU694A", "class": "hydrophobic", "total": 4,
     "by_type": {"Hydrophobic": 4}}
  ],
  "hot_partner_sites": [
    {"partner_atom": "N3", "total": 4,
     "by_type": {"HBond": 2, "PiStacking": 2}},
    {"partner_atom": "C15", "total": 1,
     "by_type": {"Hydrophobic": 1}}
  ],
  "outputs": { ... }
}
```

**Agent decision rules:**
- `top_residues` → Verify expected key contacts match the task requirements (e.g., "confirm hinge interaction with Met793").
- `hot_partner_sites` → Atoms with high `total` and strong interaction types (HBond, SaltBridge) should be **preserved** during optimization. Atoms with low `total` are safe **modification candidates**.
- `interaction_type_counts` → A molecule with 0 HBonds to the hinge region is likely a poor kinase inhibitor candidate; flag for re-evaluation.

## Residue Numbering Reconciliation (L3 Principle 17 — MANDATORY)

Use `--resid_offset N` where `N = UniProt_number − PDB_number` for the protein.

**Example:** If PDB 1M17 has Met at position 769, but UniProt numbering is Met793, then `--resid_offset 24`.

All CSV outputs include a `rec_resid_mapped` column with the offset applied. When reporting results:

- **CORRECT:** "Interaction visualizer detected HBond at MET769 (PDB) = Met793 (UniProt, offset +24). This confirms the expected hinge interaction."
- **WRONG:** "Interaction visualizer did not find Met793." (False negative from numbering mismatch.)

For Boltz-2/Chai-1 predicted structures where the offset is non-trivial, compute the offset from sequence alignment BEFORE running this tool.

## Strength Hints (Qualitative — NOT Quantitative Energies)

| Type | "strong" | "moderate" | "weak" / "packing" / "contact" |
|------|----------|------------|--------------------------------|
| HBond | d ≤ 2.9 Å and ∠ ≥ 150° | d ≤ 3.2 Å and ∠ ≥ 130° | otherwise |
| SaltBridge | d ≤ 4.0 Å | d > 4.0 Å | — |
| PiStacking | d ≤ 4.5 Å | d > 4.5 Å | — |
| Hydrophobic | — | — | "packing" (always) |
| VdW | — | — | "contact" (always) |

These are qualitative drug-chemist-style labels. They are NOT quantitative binding energy estimates. Do not use them as substitutes for MM-PBSA or FEP calculations.

## Common Failures & Recovery

| Failure | Likely Cause | Recovery |
|---------|-------------|----------|
| `No atoms parsed from {path}` | Wrong file format, empty file, or binary format given as text | Verify file is valid PDB/PDBQT; check encoding |
| `Partner atom list empty` | Ligand resname not found in PDB, or wrong `--partner_chain` | Check `--ligand_resname` matches actual HETATM resname in PDB; verify chain IDs |
| `RDKit failed to build ligand mol` | Bond perception or sanitization issue with ligand | 2D diagram skipped; CSV and all other outputs still produced. Install/upgrade rdkit if needed |
| 2D diagram PNG not generated | RDKit not installed | `pip install rdkit`; or accept CSV-only output and use `--skip_diagram2d` |
| PyMOL images not generated | PyMOL not on system PATH | Install PyMOL (`conda install -c conda-forge pymol-open-source`); or use `--skip_pymol3d` |
| 0 interactions detected | Ligand not in binding pocket (bad docking pose) or atoms too far apart | Verify docking pose quality upstream; re-dock with larger box |
| Auto-merge PDBQT fails | PDBQT has non-standard formatting | Convert PDBQT → PDB first using OpenBabel, then use `--ligand pose.pdb` |
| `AssertionError: --partner_chain required` | Peptide mode invoked without chain specification | Inspect PDB chain IDs and provide `--partner_chain` |

## Integration with Upstream Skills

| Upstream Skill | Provides | This Skill Uses As |
|----------------|----------|--------------------|
| `molclaw-quickvina-docking` | PDBQT docking pose + affinity score | `--ligand` (PDBQT) + `--score` |
| Externally supplied DiffDock result | SDF docking pose + confidence | `--ligand` (SDF) + `--score`; `diffdock_auto` is not deployed on the current MCP server |
| `molclaw-karmadock-tool` | SDF docking pose + score | `--ligand` (SDF) + `--score` |
| `molclaw-pdbfixer` | Cleaned receptor PDB | `--receptor` |
| `molclaw-fix-pdb` | Fixed PDB structure | `--receptor` or `--complex` |
| `molclaw-protein-structure-retrieve` | PDB file from RCSB/AlphaFold | `--receptor` or `--complex` |
| `molclaw-chai1-predict` | Predicted complex structure | `--complex` |
| `molclaw-boltz2-affinity` | Complex CIF + binding probability | `--complex` (convert CIF→PDB if needed) |
| `molclaw-esmfold` | Predicted protein structure | `--receptor` |
| `molclaw-proteinmpnn-tool` | Designed sequence + structure | `--complex` (protein mode) |
| `molclaw-evobind-tool` | Peptide binder + complex | `--complex` (peptide mode) |
| `molclaw-hdock-tool` | Protein-protein or protein-peptide docked complex | `--complex` with protein mode for protein-protein, or peptide mode with `partner_chain` from `hdock_tool.partner_chains` for protein-peptide |
| `molclaw-extract-chains` | Individual chain PDB files | `--receptor` / `--partner_pdb` |
| `molclaw-compound-retrieve` | Ligand SMILES | `--smiles` |

## Integration with Downstream Consumers

| This Skill Produces | Downstream Consumer | How It Is Used |
|--------------------|---------------------|----------------|
| `interactions_*.csv` | L2-08 Post-Docking Evaluation (Module 2B) | Interaction data for consensus ranking |
| `*_residue_summary.csv` | L2-08 Module 4 (SAR analysis) | Per-residue interaction profile for SAR reasoning |
| `*_partner_site.csv` | L2-05 Iterative Optimization (Step 2 Diagnose) | Identify ligand atoms to preserve vs modify |
| `summary_*.json` | Agent decision loop | Programmatic access to top residues and interaction counts |
| `diagram2d_*.png` | L2 Report assembly | Schrödinger-style visual for publication/presentation |
| `residue_bar_*.png` | L2 Report assembly | Contact composition overview |
| `interface_heatmap_*.png` | L2-09 Peptide Design validation | Interface quality assessment |
| `interface_network_*.png` | L2-09, L2-11 | Interface topology and selectivity comparison |
| `pymol_*.png` | L2 Report assembly | 3D structural context |

## Residue Role Annotation JSON Format

The `--residue_roles_json` file maps residue labels to functional roles. This enables
the 2D diagram to annotate each residue bubble with its pharmacological significance.

```json
{
    "MET769": "Hinge",
    "LEU768": "Hinge",
    "THR766": "Gatekeeper",
    "LYS721": "Catalytic",
    "GLU738": "αC-helix",
    "ASP831": "DFG",
    "PHE832": "DFG",
    "VAL702": "P-loop",
    "LEU820": "Hydrophobic",
    "ALA719": "Hydrophobic"
}
```

Keys are matched by prefix (3-letter resname + resid), so `"MET769"` matches `MET769A`, `MET769B`, etc.

Common role vocabularies by target class:
- **Kinases:** Hinge, Gatekeeper, DFG, P-loop, αC-helix, Catalytic, Hydrophobic
- **GPCRs:** TM1-TM7, ECL1-ECL3, ICL1-ICL3, Orthosteric, Allosteric
- **Proteases:** Catalytic triad, Oxyanion hole, S1-S4 pockets

## Dependencies & Installation

**Required (typically in base environment):**
- `numpy` — core geometry computation
- `matplotlib` — all 2D plot generation (diagram, bar, heatmap, network)

**Optional (enables additional output types):**

| Package | Enables | Install | Without It |
|---------|---------|---------|------------|
| `rdkit` | 2D ligand diagram + SDF/MOL2/XYZ input parsing | `pip install rdkit` or `conda install -c conda-forge rdkit` | 2D diagram skipped; CSV and all other outputs still produced |
| `pymol` (open-source) | 3D multi-angle ray-traced rendering | `conda install -c conda-forge pymol-open-source` | 3D images skipped; use `--skip_pymol3d` |
| `Pillow` | Image post-processing and compositing | `pip install Pillow` | Minor formatting differences only |

## Choosing Between This Skill and ProLIF

> **Rule: This skill is the default for all single-structure interaction analysis.**
> ProLIF is only used when its unique capabilities (batch, trajectory) are required.

| Need | This Skill (**PRIMARY**) | ProLIF (MCP, supplement) |
|------|:------------------------:|:------------------------:|
| Single complex structure interaction analysis | ✅ **default** | Only if this tool unavailable |
| Peptide/protein interface (single structure) | ✅ **default** | Only if this tool unavailable |
| Schrödinger-style 2D interaction diagram | ✅ | ❌ |
| PyMOL 3D multi-angle renderings | ✅ | ❌ |
| Residue role annotations (Hinge/Gatekeeper/DFG) | ✅ | ❌ |
| Decision-ready JSON with top residues + hot sites | ✅ | ❌ |
| Native `--resid_offset` for PDB→UniProt mapping | ✅ | ❌ |
| ProLIF MCP server unreachable or down | ✅ (local) | ❌ |
| Interaction fingerprint across MD trajectory frames | ❌ | ✅ (`prolif_md`) |
| Batch docking pose fingerprint comparison (≥ 2 poses) | ❌ | ✅ (`prolif_docking`) |
| Protein-protein trajectory interface profiling | ❌ | ✅ (`prolif_protein_protein`) |
