---
name: molclaw-target-protein-preparation
description: >
  Obtain and prepare a clean protein structure from any input form (PDB ID, gene name,
  UniProt ID, amino acid sequence, or PDB file) for downstream virtual screening.
  Almost every protein-related workflow begins with this skill.
license: MIT license
metadata:
    skill-author: PJLab
    skill-level: L2-Workflow
    version: 3.1-vs-complete
    methodology-ref: >
      Structure quality assessment, chain processing, residue numbering reconciliation,
      mandatory structure file download and verification
---

# Target Protein Preparation Workflow (MolBench-VS)

## Applicability

**Use this skill when:** The task involves any docking, screening, or interaction analysis and the protein structure is not yet in a clean, computation-ready state.

**Do NOT use this skill when:** The user has already provided a fully prepared PDB file and explicitly states it requires no further processing.

## Input Classification and Acquisition Strategy

### Protein Input Forms

| Input type | Detection | Tool | Expected output |
|------------|-----------|------|-----------------|
| **PDB ID** | 4 chars, alphanumeric (e.g., "2L3R") | `retrieve_protein_structure_by_pdb_id` | PDB file + metadata |
| **Gene name** | Standard HGNC symbol (e.g., "TP53", "EGFR") | `retrieve_protein_structure_by_gene_name` | AlphaFold PDB |
| **UniProt ID** | "P" + 5 alphanumeric (e.g., "P38398") | `retrieve_protein_structure_by_uniprot_id` | AlphaFold PDB |
| **Sequence (raw)** | >20 chars, AAs only | `pred_protein_structure_esmfold` | Predicted PDB (if len ≤ 800) |
| **PDB file path** | Filesystem path with `.pdb` extension | Skip acquisition | Proceed to quality assessment |

### Acquisition Steps

1. **PDB ID acquisition:**
   ```
   retrieve_protein_structure_by_pdb_id(pdb_id="2L3R")
   → Returns: pdb_file_path, fasta_sequence, resolution, method
   Record: resolution (affects downstream reliability), X-ray/cryo-EM/NMR method
   ```

2. **Gene name acquisition:**
   ```
   retrieve_protein_structure_by_gene_name(gene_name="TP53", organism="9606")
   → Returns: pdb_file_path (AlphaFold model)
   Record: Model source is AlphaFold, pLDDT confidence available
   ```

3. **UniProt acquisition:**
   ```
   retrieve_protein_structure_by_uniprot_id(uniprot_id="P38398")
   → Returns: pdb_file_path (AlphaFold model)
   Record: Same as gene name path
   ```

4. **Sequence prediction (if length ≤ 800 residues):**
   ```
   First validate: is_valid_protein_sequence(sequence)
   If valid: pred_protein_structure_esmfold(sequence)
   → Returns: predicted_pdb_path
   Record: ESMFold prediction, quality likely < AlphaFold
   ```

## Structure Quality Assessment

Before repair, perform a health check.

### Step 1: Basic statistics
```
call: calculate_pdb_basic_info(pdb_file_path)
record:
  - chain count (multi-chain?)
  - heteroatom count (ligands/waters present?)
  - residue count (compare to sequence length)
  - presence of co-crystal ligand (needed for docking reference)
```

### Step 2: Structural geometry
```
call: calculate_pdb_structural_geometry(pdb_file_path)
record:
  - center_of_mass (reference for pocket detection)
  - radius_of_gyration (protein size indicator)
```

### Step 3: Quality metrics
```
call: calculate_pdb_quality_metrics(pdb_file_path)
record:
  - avg_bfactor
  - interpretation:
      if predicted structure (AlphaFold/ESMFold): bfactor = pLDDT
        - pLDDT < 50: LOW confidence (issue warning but proceed)
        - pLDDT 50-70: MODERATE (note in report)
        - pLDDT > 70: HIGH (proceed normally)
      if experimental: bfactor > 80 indicates flexible regions
```

## Chain Processing Strategy

| Scenario | Action |
|----------|--------|
| **Single chain** | Skip extraction; proceed to repair |
| **Multi-chain, specific chain specified** | Call `extract_and_save_chains(chain_ids=["A"])` |
| **Multi-chain, no specification** | Call `extract_pdb_chains` to list all; ask user for selection or use functional chain identification |

## Structure Repair

Call `fix_pdb` with appropriate parameters:

```
fix_pdb(
    input_path=raw_pdb,
    output_path=repaired_pdb,
    add_hydrogens=true,        # for docking
    remove_heterogens=true,    # remove water/ligands
    remove_water=true,
    replace_nonstandard=true,
    add_missing_residues=false # for screening, not critical
)
```

**Post-repair verification:**
- Confirm `status == "success"`
- Verify output file exists and size > 0

## Mandatory File Download and Verification

**After acquisition and after repair, download the PDB file locally:**

1. **For acquired PDB (before repair):**
   ```
   server_file_to_base64(file_path=pdb_file_path)
   → Decode and save locally as: step01_raw_protein.pdb
   Verify: ls -la step01_raw_protein.pdb (size > 0)
   ```

2. **For repaired PDB (after fix_pdb):**
   ```
   server_file_to_base64(file_path=repaired_pdb)
   → Decode and save locally as: step02_prepared_protein.pdb
   Verify: ls -la step02_prepared_protein.pdb (size > 0)
   ```

**⚠️ A preparation step is NOT complete until the PDB file has been downloaded and verified locally.**

## Output Contract

- `prepared_pdb`: Path to cleaned, repair receptor PDB
- `structure_source`: One of [experimental_xray, experimental_cryoem, alphafold_prediction, esmfold_prediction, user_provided]
- `chain_info`: Chain ID(s) used, residue count
- `quality_notes`: pLDDT/B-factor assessment, any warnings
- `pocket_reference`: If co-crystal ligand was extracted, its centroid (x, y, z)
