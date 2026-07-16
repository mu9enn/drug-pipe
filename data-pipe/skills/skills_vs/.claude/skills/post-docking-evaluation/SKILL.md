---
name: molclaw-post-docking-evaluation
description: >
  Post-docking analysis and validation workflow. Takes docked poses and docking results
  from upstream workflows, performs consensus rescoring, interaction validation,
  and generates ranked predictions with interaction annotations.
license: MIT license
metadata:
    skill-author: PJLab
    skill-level: L2-Workflow
    version: 3.1-vs-complete
    methodology-ref: >
      Consensus scoring across multiple docking methods,
      mandatory interaction visualization for top poses,
      residue-specific interaction profiling,
      confidence assessment for predictions
---

# Post-Docking Evaluation and Ranking Workflow (MolBench-VS)

## Applicability

**Use this skill when:** You have docking results (poses + affinity scores) from QuickVina, and need:
- Consensus rescoring using EquiScore
- Detailed interaction analysis for top poses
- Confidence assessment of predictions
- Generation of final ranked molecular list with interaction annotations

**Do NOT use this skill when:** You have not yet run docking (use Workflow 2 first); or you only need raw docking scores without post-analysis.

**Boundary with Workflow 2:** Workflow 2 covers the full pipeline from raw inputs to ranked results. This Workflow 5 is for detailed post-hoc analysis when you already have docking outputs.

## Prerequisites

| Input | Source | Required? |
|-------|--------|-----------|
| Docking results (poses + affinities) | Workflow 2 | Yes |
| Prepared receptor PDB | Workflow 1 | Yes |
| Top N candidate molecules | Ranked by QuickVina | Yes (recommend top 10–20) |
| Co-crystal ligand pose (optional) | Original PDB or reference | No |

## Phase 1: EquiScore Consensus Rescoring

### Preparation: Define Reference Pocket

**If co-crystal ligand is available:**
```
use co-crystal ligand centroid as pocket reference

call: equiscore_pocket(
    receptor_file=prepared_pdb,
    ligand_file=cocrystal_ligand_pdb,
    pocket_radius=10.0
)
→ Returns: pocket_definition_file
```

**If no co-crystal, use QuickVina top-1 pose:**
```
Use the highest-affinity QuickVina pose from upstream workflow
Assumption: best QuickVina pose ≈ native-like binding mode
Define pocket around this pose
```

### Rescoring Top Candidates

```
For each molecule in top_N (e.g., top 15–20 by QuickVina):
    
    call: equiscore_screen(
        pocket_file=pocket_definition_file,
        ligand_file=docked_pose_pdb,
        batch_size=32
    )
    → Returns: equiscore_affinity_score
    
Collect EquiScore results for all top molecules
```

### Consensus Scoring

```
Combine QuickVina and EquiScore into a unified score:

for each molecule:
    w_vina = 0.4  # QuickVina weight
    w_eq = 0.6    # EquiScore weight (more reliable ML-based)
    
    consensus_score = w_vina * quickvina_affinity + w_eq * equiscore_affinity
    
Rerank all molecules by consensus_score (best = most negative)

⚠️ QUALITY GATE:
  If consensus_score > -5.0 kcal/mol for top molecule:
    → Issue warning: "All predicted affinities are weak. 
                      Binding may not occur. Consider re-evaluating pocket or receptor."
    → Still report results but flag confidence as "LOW"
```

## Phase 2: Detailed Interaction Analysis

### Top Poses Interaction Visualization

**For molecules in top 3–5 by consensus score:**

```
call: interaction_visualizer(
    receptor=prepared_pdb,
    ligand=docked_pose_pdb,
    mode="ligand",
    score=consensus_score,
    title="SMILES_identifier_vs_target_receptor"
)
→ Returns:
    - interaction_image (PNG visualization)
    - contact_summary (residues involved, contact types)
    - interaction_fingerprint (for comparison across molecules)
```

### Interaction Quality Assessment

```
For each interaction image, assess:

1. Ligand fit in pocket:
   - Is ligand completely buried in pocket? YES/NO
   - Does ligand make >3 interactions? YES/NO
   - Are interactions distributed (not all on one side)? YES/NO

2. Key interactions:
   - H-bonds to main chain carbonyl/amide? YES (strong indicator)
   - Hydrophobic contacts with aromatic residues? YES (favorable)
   - Unfavorable steric overlaps? NO (bad sign)

3. Confidence scoring:
   if all three checks are positive:
       confidence = "HIGH"
   elif two checks are positive:
       confidence = "MODERATE"
   elif one check is positive:
       confidence = "LOW"
   else:
       confidence = "VERY_LOW"
```

### ProLIF Fingerprinting (Optional, for multi-molecule comparison)

```
For comparing interaction patterns across multiple molecules:

call: prolif_docking(
    receptor_file=prepared_pdb,
    ligand_file=docked_pose_pdb,
    ligand_id="molecule_smiles"
)
→ Returns: interaction_fingerprint (binary vector)

Compare fingerprints:
    - Molecules with similar fingerprints likely use similar binding modes
    - Divergent fingerprints suggest different binding modes
    - Useful for structure-activity relationship (SAR) analysis
```

## Phase 3: Final Ranking and Reporting

### Unified Ranking Table

```
Generate final ranking by consensus_score:

| Rank | SMILES | Consensus_Score | QuickVina | EquiScore | Key_Interactions | Confidence |
|------|--------|-----------------|-----------|-----------|------------------|------------|
| 1 | CC(=O)Nc1ccccc1 | -8.2 | -7.5 | -9.0 | H-bond to ASP123, hydrophobic ARG45 | HIGH |
| 2 | ... | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... | ... |

Record for each molecule:
  - consensus_affinity_score
  - residues involved in binding (from interaction analysis)
  - interaction types (H-bond, hydrophobic, ionic, etc.)
  - confidence level (HIGH/MODERATE/LOW/VERY_LOW)
```

### Mandatory File Download

```
For all top 3–5 ranked molecules:

1. Download interaction visualization images:
   call: server_file_to_base64(file_path=interaction_image_png)
   → Decode and save locally as: rank1_molecule_interactions.png
   
2. Download docked pose PDB:
   call: server_file_to_base64(file_path=docked_pose_pdb)
   → Decode and save locally as: rank1_docked_pose.pdb
   
3. Download interaction summary JSON:
   call: server_file_to_base64(file_path=interaction_summary_json)
   → Decode and save locally as: rank1_interactions.json

Verify: ls -la rank*_*.png rank*_*.pdb (size > 0 for each)
```

## Phase 4: Confidence Assessment and Reporting

### Confidence Scoring Rubric

For each molecule, assign confidence based on:

| Factor | Score contribution | Assessment |
|--------|-------------------|------------|
| Consensus affinity < -7.0 kcal/mol | +25 points | Strong binder |
| Multiple interaction types (H-bond + hydrophobic + ionic) | +20 points | Versatile binding |
| Affinity agreement between QuickVina and EquiScore (< 1.0 kcal/mol difference) | +20 points | Consensus |
| Buried ligand in pocket (no solvent-exposed heavy atoms) | +15 points | Good fit |
| Interaction with known active site residues (if provided) | +20 points | Mechanism validation |

```
Confidence levels:
  Score ≥ 80: HIGH confidence
  Score 60–79: MODERATE confidence
  Score 40–59: LOW confidence
  Score < 40: VERY_LOW confidence
```

### Final Report Structure

```
Post-Docking Analysis Report
============================

1. Executive Summary
   - Top molecule SMILES and consensus score
   - Predicted binding mode (brief description)
   - Recommended experiments (e.g., biochemical assay)

2. Docking Statistics
   - Total molecules docked: N
   - Molecules with affinity < -5.0 kcal/mol: M (binding probability)
   - Average consensus score across top 10: [score]

3. Top 5 Ranked Molecules
   [See unified ranking table above]

4. Interaction Analysis
   [For each top 3 molecule, include:]
   - Interaction visualization image
   - Residue contact list
   - Interaction fingerprint (if ProLIF analysis done)
   - Confidence assessment

5. SAR (Structure-Activity Relationship) Insights
   [If multiple similar molecules present:]
   - Common binding features
   - Substructures driving binding
   - Suggestions for optimization

6. Limitations and Caveats
   - Pocket detection method and consensus confidence
   - RMSD or quality metrics (if available)
   - Any unusual binding modes that warrant manual review
```

## Output Contract

- `ranked_molecules`: Final list sorted by consensus_score [SMILES, affinity, interactions, rank]
- `confidence_scores`: Confidence assessment for each molecule [molecule_id, confidence_level, score]
- `interaction_images`: Local file paths to top 3–5 PNG visualizations
- `interaction_summary`: Residue-level contact data for all analyzed molecules
- `final_report_md`: Markdown report with all above information
- `warnings_and_flags`: Any anomalies detected (e.g., all affinities weak, divergent rescoring, etc.)