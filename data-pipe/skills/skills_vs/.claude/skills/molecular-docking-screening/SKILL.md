---
name: molclaw-molecular-docking-screening
description: >
  Molecular docking virtual screening workflow: from molecule acquisition and validation,
  through pocket detection, docking execution, rescoring, and interaction analysis.
  Supports automated tiered strategy selection based on library size.
license: MIT license
metadata:
    skill-author: PJLab
    skill-level: L2-Workflow
    version: 3.2-vs-complete
    methodology-ref: >
      Tiered screening with count verification at each boundary,
      multi-method pocket detection with consensus,
      docking parameter safeguards, mandatory file download, interaction analysis
---

# Molecular Docking Virtual Screening Workflow (MolBench-VS)

## Applicability

**Use this skill when:** The user provides a target protein and small molecules (SMILES/names), and requests molecular docking, virtual screening, binding assessment, or ranking molecules by predicted binding affinity.

**Do NOT use this skill when:** The user only needs property filtering without docking; the task is peptide-protein docking; or you already have docking results and only need post-hoc evaluation.

## Prerequisites

| Input | Source | Required? |
|-------|--------|-----------|
| Prepared protein PDB | Workflow 01 | Yes |
| Molecule SMILES list | User input | Yes |
| Binding pocket info | Optional (computed if missing) | No |

## Phase 1: Scale Assessment and Strategy Selection

**Before any computation, determine the screening strategy based on library size:**

| Library size | Strategy | Tiers | Pocket detection |
|-------------|----------|-------|-----------------|
| **≤ 10** | Full evaluation | Direct docking + EquiScore + interaction analysis | Single method (fpocket) |
| **11–100** | Two-tier | Pre-filter (QED/Lipinski) → Dock all | Dual method consensus |
| **101–500** | Three-tier | Pre-filter → Dock → EquiScore top 20 | Dual method consensus |
| **501–2000** | Four-tier | Strict pre-filter → Dock top 100 → EquiScore → Interaction analysis top 15 | Dual method consensus |
| **> 2000** | Recommend KarmaDock or stricter pre-filter | - | Specialized high-throughput tool |

**Record the chosen strategy and rationale in a log file.**

## Phase 2: Molecule Acquisition and Validation

### Step 1: Obtain SMILES
```
Input: compound_names OR direct_smiles_list OR mixed

if compound_names:
    call: retrieve_smiles_by_compoundname(compound_names)
    → Returns: smiles_for_each_compound

Combine all into single smiles_list
```

### Step 2: Validate SMILES
```
call: is_valid_smiles(smiles_list)
→ Returns: valid_smiles, invalid_smiles, error_reasons

⚠️ COUNT GATE (MANDATORY):
  Input count: X molecules
  Valid: Y molecules
  Invalid: Z molecules (record reasons for each)
  Record this programmatically; do NOT estimate
```

### Step 3: Drug-Likeness Pre-Filtering (if strategy requires)
```
call: calculate_mol_drug_chemistry(valid_smiles)
→ Returns: qed_score, lipinski_violations for each

Apply thresholds:
  - QED >= 0.2 (keep drug-like molecules)
  - Lipinski violations <= 2 (allow some flexibility)
  - Optional: MW in range [150, 900] Da

⚠️ COUNT GATE (MANDATORY):
  Pre-filter input: Y molecules
  Passed filters: P molecules
  Eliminated: Y-P molecules
  Record breakdown: Lipinski>2 (A), QED<0.2 (B), MW out of range (C)
```

### Step 4: Format Conversion
```
call: convert_smiles_to_format(inputs=screened_smiles, target_format="pdbqt")
→ Returns: pdbqt_files (for QuickVina docking)

⚠️ COUNT GATE (MANDATORY):
  Conversion input: P molecules
  Successful: Q molecules
  Failed: P-Q molecules (record errors)
```

## Phase 3: Binding Pocket Detection

### Decision: Is detection needed?
- User provided pocket center coordinates → Use directly
- User specified active site residues → Still run detection; validate against reference
- No pocket information → Detection is mandatory

### Dual-Tool Detection Strategy (recommended)

**Method 1: fpocket**
```
call: fpocket_toolkit(
    pdb_file=prepared_pdb,
    top_n=1,
    min_druggability=None
)
→ Returns: pocket_center=(x_fp, y_fp, z_fp), druggability_score
```

**Method 2: P2Rank**
```
call: pred_pocket_prank(pdb_file_path=prepared_pdb)
→ Returns: top_pocket with center=(x_pr, y_pr, z_pr), confidence_score
```

### Consensus Logic
```
Compute distance between fpocket_center and prank_center:
  
  if distance < 5 Å:
    → High confidence
    → Use fpocket center
    → Record both methods agree
    
  if 5 Å ≤ distance < 10 Å:
    → Moderate confidence
    → Use midpoint of two centers
    → Note slight disagreement in report
    
  if distance ≥ 10 Å:
    → Divergence detected
    → Report both pockets as alternatives
    → Use the one with higher druggability score by default
    → Allow user to select
```

## Phase 4: Docking Execution

### Docking Box Parameters
```
Docking parameters (LOCKED AFTER BASELINE):
  center_x, center_y, center_z (from consensus pocket detection)
  size_x=25.0, size_y=25.0, size_z=25.0 (default; enlarge if needed)

⚠️ For iterative tasks: Lock these parameters AFTER baseline docking.
   Do NOT re-run pocket detection in subsequent rounds.
   Use exact same coordinates for all follow-up rounds.
```

### QuickVina Docking
```
for each molecule (in batch):
    call: molecule_docking_quickvina_fullprocess(
        pdb_file_path=prepared_pdb,
        smiles=molecule_smiles,
        pocket_center_x=x,
        pocket_center_y=y,
        pocket_center_z=z,
        pocket_size_x=25.0,
        pocket_size_y=25.0,
        pocket_size_z=25.0
    )
    → Returns: affinity_score, pose_pdbqt

Collect all docking results:
  molecules_docked = [affinity scores with corresponding SMILES]
```

### EquiScore Rescoring (for top N molecules)

**Step 1: Prepare EquiScore pocket**
```
If user has co-crystal ligand pose (e.g., crystallized inhibitor):
    call: equiscore_pocket(
        receptor_file=prepared_pdb,
        ligand_file=cocrystal_ligand_pdb,
        pocket_radius=10.0
    )
    → Returns: pocket_definition_file
    
If no co-crystal:
    Use QuickVina top-1 pose from previous step as reference
```

**Step 2: Rescore top N candidates**
```
Select N molecules (e.g., top 15-20 by QuickVina affinity):
  
for molecule in top_n_molecules:
    call: equiscore_screen(
        pocket_file=pocket_definition_file,
        ligand_file=molecule_pose_pdb,
        batch_size=32
    )
    → Returns: equiscore_affinity (more reliable than QuickVina alone)

Rerank using combined score:
  final_score = 0.5 * quickvina_affinity + 0.5 * equiscore_affinity
```

## Phase 5: Interaction Analysis (for top poses)

### Top Pose Interaction Analysis

For each molecule in top 3-5:
```
call: interaction_visualizer(
    receptor=prepared_pdb,
    ligand=docked_pose_pdb,
    mode="ligand",
    score=combined_affinity_score,
    title="SMILES_code_vs_target"
)
→ Returns: 
    - interaction_image (PNG)
    - interaction_summary (residue contacts, H-bonds, hydrophobic)
    - residue_roles.json (annotated interactions)
```

### Optional: ProLIF Interaction Fingerprinting
```
For detailed interaction comparison across molecules:

call: prolif_docking(
    receptor_file=prepared_pdb,
    ligand_file=docked_pose_pdb,
    ligand_id="molecule_smiles"
)
→ Returns: interaction_fingerprint, contact_summary
```

## Phase 6: Results Aggregation and Ranking

### Final Output Structure
```
results:
  - molecule: SMILES string
  - quickvina_affinity: kcal/mol
  - equiscore_affinity: kcal/mol
  - combined_score: (weighted average)
  - rank: final rank (1 = best binder)
  - interactions: [residue names, contact types]
  - interaction_image_path: PNG file
  - confidence: "high/moderate/low"
```

### Mandatory File Download
```
For top 5 poses:
  - interaction_image: download PNG locally
  - docked_pose_pdb: download PDBQT → convert to PDB locally
  - interaction_summary.json: download locally

Use: server_file_to_base64() → decode and save locally
Verify: ls -la [each file] (size > 0)
```

## Output Contract

- `ranked_molecules`: List of [SMILES, combined_affinity, interactions_summary, rank]
- `top_pose_images`: Local paths to interaction visualization PNGs (top 3-5)
- `docking_parameters`: Locked box center and size used
- `screening_log`: Funnel counts at each filtering stage
- `interaction_details`: Residue-level contact analysis for top molecules
    pocket_center = fpocket_center  # prefer fpocket for this workflow
else:
    pocket_center = fpocket_center or prank_center
```

### Step 4: QuickVina Docking (Initial Scoring)
**Tool:** `molclaw-quickvina-docking` → `molecule_docking_quickvina_fullprocess`

Generate docking poses for all validated molecules:

```python
response = await client.session.call_tool(
    "molecule_docking_quickvina_fullprocess",
    arguments={
        "pdb_file": prepared_pdb,
        "pocket_center": pocket_center,
        "pocket_size_x": 30,
        "pocket_size_y": 30,
        "pocket_size_z": 30,
        "smiles_list": validated_smiles,
        "num_modes": 5,
        "seed": 42,
        "timeout_per_ligand_sec": 120,
        "exhaustiveness": 8
    }
)
result = client.parse_result(response)
docking_output_sdf = result["output_sdf"]
initial_scores = result.get("scores")  # {smiles: best_score}
```

**Retry Strategy (if needed):**
If many docking failures occur, progressively increase box size:
- Box 30×30×30 (default)
- Box 40×40×40 (if >30% failures)
- Box 50×50×50 (if still failing)

### Step 5: EquiScore Rescoring & Ranking (ML-Based Refinement)
**Tool:** `molclaw-equiscore-tool` + `molclaw-equiscore-docking`

Apply learned scoring function for improved ranking:

```python
# Option A: One-click pipeline (simplest)
response = await client.session.call_tool(
    "equiscore_pipeline",
    arguments={
        "docking_result": docking_output_sdf,
        "receptor_pdb": prepared_pdb,
        "ngpu": 1,
        "multi_pose": False,
        "pose_num": 1,
        "dry_run": False
    }
)
equiscore_result = client.parse_result(response)
predictions_path = equiscore_result["predictions_path"]
```

**OR Option B: Step-by-step (for advanced control):**

```python
# Step B1: Pocket extraction
response = await client.session.call_tool(
    "equiscore_pocket",
    arguments={
        "docking_result": docking_output_sdf,
        "receptor_pdb": prepared_pdb,
        "pocket_cutoff": None,
        "dry_run": False
    }
)
pocket_result = client.parse_result(response)
pocket_dir = pocket_result["pocket_dir"]

# Step B2: Pocket scoring
response = await client.session.call_tool(
    "equiscore_screen",
    arguments={
        "pocket_dir": pocket_dir,
        "ngpu": 1,
        "batch_size": 128,
        "num_workers": 8,
        "multi_pose": False,
        "dry_run": False
    }
)
screen_result = client.parse_result(response)
predictions_path = screen_result["predictions_path"]
```

**EquiScore Interpretation:**
- EquiScore outputs a "score" column (typically `test_pred` in range 0.0–1.0)
- Higher scores indicate higher predicted activity
- **Ranking rule:** Sort predictions CSV by score column in **descending order** (best predictions first)
- Use for **relative ranking**, not absolute classification

### Step 6: Parse EquiScore Predictions
**Tool:** File read + CSV parsing

Retrieve and rank the EquiScore predictions:

```python
import csv
import pandas as pd

# Read predictions CSV
df_pred = pd.read_csv(predictions_path)

# Rank by score (descending order)
df_ranked = df_pred.sort_values("test_pred", ascending=False)

# Extract final SMILES list
final_ranking_smiles = df_ranked[["smiles"]].values.flatten().tolist()

# Verify length consistency
assert len(final_ranking_smiles) > 0, "No predictions generated"
print(f"EquiScore ranking produced {len(final_ranking_smiles)} ranked predictions")
```

### Step 7: Top-N Interaction Analysis (Validation)
**Tool:** `molclaw-interaction-visualizer` (optional, for top 3-5 poses)

Validate top-ranked poses with detailed interaction analysis:

```python
top_n = min(3, len(final_ranking_smiles))

for idx, smiles in enumerate(final_ranking_smiles[:top_n]):
    # Extract corresponding docking pose from output SDF (ligand-specific)
    # This requires indexing the docking output SDF or converting back to complex PDB
    
    response = await client.session.call_tool(
        "analyze_interactions_local",
        arguments={
            "receptor": prepared_pdb,
            "ligand": pose_sdf_for_smiles,  # specific docking pose
            "mode": "ligand",
            "out_dir": f"viz_top_{idx}",
            "score": initial_scores.get(smiles)
        }
    )
    analysis = client.parse_result(response)
    # Log key interactions (hydrogen bonds, salt bridges)
```

**Optional:** Only for top 3-5 predictions; skip for full ranking to save computation.

## Output Contract

### Required Output
- **Final ranking:** Ordered SMILES list (highest EquiScore predictions first)
  - Length = number of molecules successfully passing all filters + docking + scoring
  - Order: descending by EquiScore prediction
  
- **`result.md`** summary including:
  - Drug-likeness filtering: `X → Y molecules` (X input, Y retained)
  - SMILES validation: `Y → Z molecules` (Z valid)
  - Docking completion: `Z molecules docked, W successful`
  - EquiScore rescoring: `W → final predictions` (EquiScore ranked)
  - Method description: "QuickVina docking + EquiScore ML rescoring"
  - Sanity checks:
    - Expected score ranges (QuickVina: typically -12 to 0 kcal/mol; EquiScore: 0.0–1.0)
    - Outlier detection (any scores outside expected ranges?)
  - Limitations:
    - "EquiScore trained on actives/inactives; not a binding affinity estimator"
    - "Top predictions should be validated experimentally"

### Optional Output (for reports)
- Top-3 pose visualization PNGs (interaction diagrams)
- Docking score distribution histogram
- EquiScore prediction distribution

## Key Tool References

| Tool | Purpose | Import Module |
|------|---------|---------------|
| `molclaw-drug-likeness` | Pre-filter by QED + Lipinski | `calculate_mol_drug_chemistry` |
| `molclaw-smiles-valid-check` | Validate SMILES structures | `is_valid_smiles` |
| `molclaw-fpocket` | Detect binding pocket (consensus) | `fpocket_toolkit` |
| `molclaw-p2rank` | Detect binding pocket (ML-based) | `pred_pocket_prank` |
| `molclaw-quickvina-docking` | Initial docking poses + scores | `molecule_docking_quickvina_fullprocess` |
| `molclaw-equiscore-tool` | ML-based pocket-level rescoring | `equiscore_pocket`, `equiscore_screen`, `equiscore_pipeline` |
| `molclaw-interaction-visualizer` | Validate top poses with interaction analysis | `analyze_interactions_local` |
| `molclaw-file-transfer` | Transfer large outputs if needed | `server_file_to_base64`, `base64_to_server_file` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 0 molecules pass drug-likeness filter | Lower QED/Lipinski thresholds or use unfiltered candidates |
| Docking completion << 100% | Increase pocket box size (30→40→50) or extend timeout_per_ligand_sec |
| EquiScore pocket extraction fails (pocket_item_count=0) | Verify docking output SDF is valid; check box size covers binding site |
| Final ranking is empty | Check EquiScore predictions_path file exists and is readable |
| Ranking length << original candidates | Expected due to filtering (drug-likeness) + docking (failures) + rescoring (valid pockets) |
