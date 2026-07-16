---
name: molclaw-drug-discovery-methodology
description: >
  Comprehensive L3 methodology for MolBench-VS virtual screening runs.
  Covers protein preparation, molecular docking, rescoring, interaction analysis,
  and confidence assessment.
license: MIT license
metadata:
  skill-author: PJLab
  skill-level: L3-Methodology
  version: 3.2-vs-complete
---

## Workflow Skill Routing (New Architecture)

Use these four workflow skills under `./.claude/skills`:

1. `target-protein-preparation/SKILL.md`
2. `molecular-docking-screening/SKILL.md`
3. `molecular-property-analysis-filtering/SKILL.md`
4. `post-docking-evaluation/SKILL.md`

Workflow selection guidance:
- If receptor structure is missing or not prepared, start with `target-protein-preparation`.
- For full virtual screening and ranking, use `molecular-docking-screening` as the main pipeline.
- If the task is property-centric filtering or pre-filtering, use `molecular-property-analysis-filtering`.
- If docking has already been completed and only post-hoc scoring/interaction validation is required, use `post-docking-evaluation`.

This document defines the global methodology and quality bar for MolBench-VS tasks.

# MolBench-VS Virtual Screening Methodology

## Scope

**Task:** Rank small molecule candidates by predicted binding affinity to a target protein using computational docking and rescoring.

**Inputs:** 
- Target protein (PDB ID, gene name, UniProt ID, sequence, or PDB file)
- Candidate molecules (SMILES strings or compound names)

**Output:** Complete ranked list of candidates sorted by predicted binding affinity (best binder first)

**Out of Scope:** De novo design, peptide/protein design, molecular dynamics, free-energy calculations

## Core Principles

### Principle 1: Understand Before Acting
- Parse the target information: identify protein source and expected format
- Parse the candidate list: count molecules, validate SMILES
- Confirm expected output: rank ALL candidates or select top N?
- Identify any special handling (e.g., macrocycles, covalent inhibitors, reference compounds)

**Implementation:**
```
In your initial analysis:
  1. Extract target protein identifier (PDB ID / gene name / sequence / file)
  2. Count candidates and validate SMILES format
  3. Check for special molecule types (record any found)
  4. Note any task-specific requirements
```

### Principle 2: Structure-First Pipeline
All downstream work depends on a high-quality prepared receptor. This is non-negotiable.

**Workflow 1 (Target Protein Preparation):**
- Acquire structure via appropriate tool
- Assess quality (resolution, pLDDT, B-factors)
- Repair using `fix_pdb` (remove water/heterogens, standardize residues)
- Mandatory file download: verify PDB exists locally with size > 0

### Principle 3: Tiered Screening with Count Verification
Always verify counts at filtering boundaries to detect data loss.

**Count Gates:**

| Gate | Verification | Action |
|------|-------------|--------|
| **Input** | X total candidates | Record input count explicitly |
| **After SMILES validation** | Y valid SMILES | Record invalid count + reasons |
| **After pre-filter (if used)** | P candidates passed QED/Lipinski | Record eliminated count + breakdown |
| **After format conversion** | Q successful PDBQT files | Record conversion failures |
| **After docking** | Q docked successfully | Record docking failures with reasons |
| **After rescoring** | R EquiScore available | Record rescoring gaps |
| **Final ranking** | R molecules ranked | Verify R = expected deliverable count |

**Implementation:**
```
After each filter stage, execute:
  print(f"Gate: Input {X} → Processed {Y} → Eliminated {X-Y}")
  print(f"Breakdown: [reason1=A, reason2=B, other={X-Y-A-B}]")
  
These counts MUST come from actual tool output, not estimates.
```

### Principle 4: Multi-Method Pocket Detection
Never rely on a single pocket prediction. Use consensus.

**Dual-Method Consensus:**
```
Run both: fpocket_toolkit + pred_pocket_prank

If results agree (distance < 5 Å):
  → HIGH confidence, proceed
  
If results differ by 5–10 Å:
  → MODERATE confidence, use midpoint
  
If results diverge (> 10 Å):
  → RED FLAG, investigate or use user-specified pocket
```

### Principle 5: Docking Parameter Safeguards
Inadequate box size is a common cause of false negatives.

**Box Size Requirements:**
```
Minimum size for each dimension: 25.0 Å

If docking fails or produces suspiciously weak affinities:
  Progressive enlargement strategy:
    1. First attempt: 25 × 25 × 25 Å
    2. If fails/weak: 30 × 30 × 30 Å
    3. If still fails: 40 × 40 × 40 Å
    4. If still fails: 50 × 50 × 50 Å
    
  Document box size used in final report.
  Flag if enlarged box was necessary.
```

### Principle 6: Affinity Consistency Checking
Compare across multiple scoring methods to detect outliers.

**Consistency Criteria:**
```
For each molecule, compare:
  - QuickVina affinity (initial docking score)
  - EquiScore affinity (ML-based rescoring)
  
Disagreement check:
  if |quickvina - equiscore| > 2.0 kcal/mol:
    → Flag molecule as "inconsistent scoring"
    → Manual review of pose quality recommended
    → Use consensus score instead of either alone
```

### Principle 7: Interaction Validation for Top Poses
Predicted scores are fallible. Validate top predictions with interaction analysis.

**Interaction Validation Checklist:**

For top 3–5 ranked molecules:
```
call: interaction_visualizer(...)
→ Examine the generated PNG:

Quality check:
  ☐ Ligand is completely buried in pocket (not partially solvent-exposed)
  ☐ Ligand makes ≥ 3 interactions (H-bonds, hydrophobic, ionic)
  ☐ Interactions are distributed across multiple residues (not clustered on one side)
  ☐ No obvious steric clashes or atom overlaps
  
If all checks pass:
  → Confidence = HIGH
  
If 2 checks pass:
  → Confidence = MODERATE
  
If 1 check passes:
  → Confidence = LOW
  
If 0 checks pass:
  → Confidence = VERY_LOW (question prediction quality)
```

### Principle 8: Mandatory File Download and Verification
All structural outputs must be verified locally.

**Files to Download:**

For top 5 ranked molecules:
```
1. Interaction visualization PNG
   call: server_file_to_base64(file_path=interaction_image)
   → Decode, save locally as: rank1_interactions.png
   → Verify: ls -la rank1_interactions.png (size > 0)

2. Docked pose PDB
   call: server_file_to_base64(file_path=docked_pose_pdb)
   → Decode, save locally as: rank1_pose.pdb
   → Verify: ls -la rank1_pose.pdb (size > 0)

3. Interaction summary JSON
   call: server_file_to_base64(file_path=interaction_summary)
   → Decode, save locally as: rank1_interactions.json
   → Verify: file is valid JSON
```

**Do NOT report results until all files are downloaded and verified.**

### Principle 9: No Fabrication
Always report tool output accurately. Never invent scores or interactions.

**Prohibited Actions:**
- Do NOT interpolate missing docking scores
- Do NOT invent interaction details not shown in interaction_visualizer output
- Do NOT extrapolate affinity from related molecules (if a molecule failed to dock, report failure)
- Do NOT hide docking failures in the final list

**Required Disclosures:**
```
If molecules failed any processing stage:
  → Include in final report with failure reason
  → Do NOT omit from deliverable
  
  Example: "SMILES_xyz: docking timeout after 15 min (unknown reason)"
```

### Principle 10: Confidence Tiers in Output

Assign confidence scores to all predictions.

**Confidence Assignment Rubric:**

| Factor | Score | Assessment |
|--------|-------|------------|
| Affinity < -7.0 kcal/mol | +25 | Strong binder |
| Multiple interaction types | +20 | Versatile binding |
| Agreement between methods (< 1 kcal/mol) | +20 | Consensus |
| Buried ligand in pocket | +15 | Good fit |
| Recognized interaction with known active site | +20 | Mechanism validation |

Total score: 0–100
- 80+: HIGH confidence
- 60–79: MODERATE confidence
- 40–59: LOW confidence
- <40: VERY_LOW confidence

**Implementation:**
```
For each ranked molecule, include:
  {
    "rank": 1,
    "smiles": "...",
    "affinity_kcal_mol": -8.2,
    "confidence": "HIGH",
    "confidence_score": 85,
    "key_interactions": ["ASP123 H-bond", "ALA45 hydrophobic"]
  }
```

## Workflow Integration

### Standard Workflow (for all VS tasks)

```
1. Input Parsing (Principle 1)
   ↓
2. Target Protein Preparation (Workflow 1, Principle 2)
   ↓
3. Property Pre-Filtering (optional, Workflow 3)
   ↓
4. SMILES Validation (Principle 3)
   ↓
5. Pocket Detection (Principle 4)
   ↓
6. QuickVina Docking (Principle 5–6)
   ↓
7. EquiScore Rescoring (Principle 6)
   ↓
8. Ranking and Consensus Scoring (Principle 6)
   ↓
9. Interaction Validation (Workflow 4, Principle 7)
   ↓
10. File Download and Verification (Principle 8)
   ↓
11. Confidence Assignment (Principle 10)
   ↓
12. Final Report and Deliverable (Principles 1, 9)
```

## Mandatory Deliverable Contract

**Final answer must include:**

```json
{
  "ranking": [
    {
      "rank": 1,
      "smiles": "CC(=O)Nc1ccccc1",
      "consensus_affinity": -8.2,
      "confidence": "HIGH"
    },
    ...
  ],
  "summary": {
    "total_candidates": 100,
    "successfully_docked": 98,
    "top_binder_affinity": -8.2,
    "method": "QuickVina2 + EquiScore consensus"
  },
  "caveats": [
    "2 molecules failed docking (timeout after 15 min)",
    "Pocket detected by consensus of fpocket and P2Rank",
    "All affinities >-5 kcal/mol should be treated as uncertain"
  ]
}
```

## Quality Checklist

Before submitting final answer, verify:

- [ ] All input candidates counted and recorded
- [ ] SMILES validation count gate documented
- [ ] Pocket detection method and consensus documented
- [ ] Docking box size locked and reported
- [ ] Affinity consistency across methods checked
- [ ] Top 3–5 poses have interaction visualizations downloaded
- [ ] Confidence scores assigned based on rubric
- [ ] Failures and workarounds documented
- [ ] Final ranking length matches expected output count
- [ ] Answer is formatted as JSON array inside `<answer>...</answer>` tags
