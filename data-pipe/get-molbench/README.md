# get-molbench

Python-first reorganized workspace for generating three MolBench-MS style datasets.

## Directory Layout
- `pipelines/`: unified Python entrypoints
  - `generate_molbench_ac.py`
  - `generate_molbench_vs.py`
  - `generate_molbench_pf.py`
- `data/`:
  - `CARA/` (kept as a full package)
  - `ACNet/` (kept minimal: required ACNet input data)
- `scripts/`: canonical Python generators and utilities
  - `generate_dataset_ACNet_v0.2.py`
  - `generate_molbench_vs.py`
  - `make_rdkit_benchmark_v0.py`
  - `make_rdkit_benchmark_v1.py`
  - `molecular_similiar.py`
  - `check_case31.py`, `check_case39.py`
- `examples/`: reference CSV examples (not used as pipeline input)
- `legacy/`: archived R scripts
- `docs/`: technical notes and provenance docs

## Quick Start

### 1) MolBench-AC
```bash
python pipelines/generate_molbench_ac.py --n-cases 25 --seed 100 --out-dir outputs/ac
```
Output: `outputs/ac/molbench-ac-25.csv`

### 2) MolBench-VS
```bash
python pipelines/generate_molbench_vs.py --n-cases 25 --seed 42 --out-dir outputs/vs --no-remote-target-name
```
Output: `outputs/vs/molbench-vs-25.csv`

### 3) MolBench-PF
```bash
python pipelines/generate_molbench_pf.py --variant v1 --n-cases 50 --seed 42 --out-dir outputs/pf
```
Output: `outputs/pf/molbench-pf-50.csv`

## Generate 900 / 900 / 900

### One-command batch generation
```bash
python scripts/generate_molbench_900.py
```

Outputs:
- `outputs/ac/molbench-ac-900.csv`
- `outputs/vs/molbench-vs-900.csv`
- `outputs/pf/v0/molbench-pf-300.csv`
- `outputs/pf/v1/molbench-pf-300.csv`
- `outputs/pf/similarity/molbench-pf-300.csv`
- `outputs/pf/molbench-pf-900.csv` (auto-merged from v0/v1/similarity)

Note:
- Similarity generation may produce fewer raw rows under strict constraints; `generate_molbench_900.py` expands to 300 rows via resampling to keep a fixed `300/300/300` split.

### Manual PF merge (if needed)
```bash
python scripts/merge_molbench_pf.py \
  --v0-csv outputs/pf/v0/molbench-pf-300.csv \
  --v1-csv outputs/pf/v1/molbench-pf-300.csv \
  --similarity-csv outputs/pf/similarity/molbench-pf-300.csv \
  --out outputs/pf/molbench-pf-900.csv
```

For script differences and PF provenance, see:
- `docs/rdkit_scripts_and_pf_provenance.md`

## Notes
- Output naming convention is unified: `molbench-<ac|vs|pf>-<N>.csv`.
- `examples/` is read-only reference data.
- R scripts are archived under `legacy/` and are not part of the primary workflow.
