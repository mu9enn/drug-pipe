---
name: molclaw-molecular-property-analysis-filtering
description: >
  Comprehensive small molecule property analysis, drug-likeness assessment,
  and compound library filtering. Can be used standalone or as a sub-module
  within the docking workflow.
license: MIT license
metadata:
    skill-author: PJLab
    skill-level: L2-Workflow
    version: 3.0-vs-complete
    methodology-ref: >
      Tiered property analysis with count verification at each gate,
      drug-likeness assessment using QED and Lipinski rules,
      mandatory consistency checking between MW and molecular formula
---

# Molecular Property Analysis and Filtering Workflow (MolBench-VS)

## Applicability

**Use this skill when:** The user provides molecules (SMILES or compound names) and needs property computation, drug-likeness assessment, or library filtering.

**Do NOT use this skill when:** The task requires only docking without property filtering (use Workflow 2 directly); or the molecules are peptides/proteins (Lipinski rules do not apply).

**Special molecule types:** Macrocycles, PROTACs, natural products, and covalent inhibitors violate Lipinski rules by design. When handling these, relax or skip Lipinski-based filters and document the exception.

## Prerequisites

No upstream dependency. SMILES input only.

## Core Workflow

### Step 1: Input Acquisition and Standardization

Convert all inputs to a SMILES list:

```
if compound_names:
    call: retrieve_smiles_by_compoundname(compound_names)
    → Returns: smiles_for_each_compound

if mixed_input:
    Classify each entry: valid SMILES or compound name
    Process separately, then combine

Final result: unified_smiles_list
```

### Step 2: Validate and Deduplicate

```
call: is_valid_smiles(unified_smiles_list)
→ Returns: valid_smiles, invalid_smiles, error_reasons

Remove duplicates: unique_smiles_list = set(valid_smiles)

⚠️ COUNT GATE (MANDATORY):
  Total input: X molecules
  Valid SMILES: Y molecules
  Invalid SMILES: Z molecules (record specific errors for each)
  Unique after dedup: W molecules
  
  Record these counts programmatically from validation output.
  These counts MUST NOT be estimates.
```

## Step 3: Three-Tier Property Analysis

### Tier 1: Basic Properties (ALWAYS execute)

This is the foundation for all downstream decisions.

```
call: calculate_mol_basic_info(unique_smiles_list)
→ Returns:
    - molecular_weight (MW)
    - molecular_formula
    - heavy_atom_count
    - bond_count
    - formal_charge
```

**⚠️ Mandatory consistency check:**
For each molecule, verify that MW matches the molecular formula:
```
Example: C₁₄H₁₄O₃ 
  Expected MW = (14×12.01) + (14×1.008) + (3×16.00) = 242.26 Da
  If returned MW = 244.29, MISMATCH detected
  → Re-validate SMILES, recompute
  → Flag this molecule for manual review
```

Do NOT silently accept MW-formula inconsistencies.

```
call: calculate_mol_drug_chemistry(unique_smiles_list)
→ Returns:
    - QED score [0.0 to 1.0]
    - Lipinski rule-of-5 violations count [0-4]
```

**Interpretation:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| QED | ≥ 0.2 | Drug-like character (0.0=worst, 1.0=best) |
| Lipinski violations | ≤ 2 | Orally active (0 violations = best) |
| MW (from basic_info) | 150–900 Da | Typical drug MW range |
| HBD | ≤ 5 | Hydrogen bond donors |
| HBA | ≤ 10 | Hydrogen bond acceptors |

### Tier 2: Extended Properties (execute when detailed analysis needed)

```
call: calculate_mol_basic_info (if not already called)
→ Returns: HBD, HBA (if tool supports; otherwise estimate from SMILES)

Additional computations (if available):
- TPSA (Topological Polar Surface Area)
  * TPSA < 60 Å² → likely BBB-penetrant (CNS passive permeability)
  * TPSA > 140 Å² → poor oral bioavailability
  
- Rotatable bonds count
  * ≤ 10 → good oral bioavailability (Veber rules)
  * > 12 → likely poor bioavailability
```

### Tier 3: Binding Affinity Pre-Prediction (optional, if available)

If available, use rapid binding affinity tools before full docking:

```
call: pred_binding_affinity_boltz2(protein_structure, smiles)
→ Returns: predicted_affinity_score

Use this to:
- Rank molecules before expensive docking step
- Identify promising leads early
- Filter out molecules with predicted very poor affinity
```

## Step 4: Filtering Decision

Select filtering stringency based on task requirements:

| Mode | Criteria applied | Expected elimination | When to use |
|------|-----------------|---------------------|-------------|
| **None** | No filtering | 0% | Full characterization only; user accepts all |
| **Lenient** | Remove only: invalid SMILES, MW < 100 or > 1000 | 5–15% | Early exploration, natural products, lead discovery |
| **Standard** (default) | Lipinski violations ≤ 2; QED ≥ 0.2; MW in [150,900] | 15–40% | Routine virtual screening |
| **Strict** | Lipinski violations = 0; QED ≥ 0.4; TPSA ≤ 120; RotBonds ≤ 10 | 30–60% | Lead optimization, oral drug development |
| **Custom** | User-specified thresholds mapped to specific fields | Variable | User provides explicit criteria |

### Filtering Implementation

```
Based on chosen mode, apply thresholds:

filtered_molecules = []
for mol in analyzed_molecules:
    if apply_filtering_criteria(mol):
        filtered_molecules.append(mol)

⚠️ COUNT GATE (MANDATORY):
  Post-filter input: W molecules (from previous gate)
  Passed filters: F molecules
  Eliminated: W-F molecules
  
  Record breakdown:
    - Lipinski violations > threshold: A molecules
    - QED < threshold: B molecules
    - MW out of range: C molecules
    - Other criteria: D molecules
  
  Verify: A + B + C + D = W - F
```

## Step 5: Summary Report

Generate a property analysis report for each molecule:

```
For filtered_molecules:
  
  molecule_report = {
    "smiles": smiles_string,
    "mw": molecular_weight,
    "formula": molecular_formula,
    "qed": qed_score,
    "lipinski_violations": violation_count,
    "hbd": hydrogen_bond_donors,
    "hba": hydrogen_bond_acceptors,
    "predicted_affinity": (if available),
    "passed_filters": true/false,
    "filter_reason": (if failed)
  }
```

## Output Contract

- `filtered_smiles_list`: List of SMILES that passed filtering
- `property_table`: DataFrame or JSON with columns: [SMILES, MW, QED, Lipinski_violations, HBD, HBA, ...]
- `screening_funnel`: Count at each gate (input → valid → deduplicated → filtered)
- `eliminated_molecules`: List with reasons for elimination
- `recommendations`: Suggestions for adjusting filter thresholds if too many molecules are eliminated