#!/usr/bin/env bash
# One-time downloader for the fixed Stage3 Science-KB source snapshot.
# This is intentionally separate from run_sample_questions.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$PROJECT_ROOT/science_kb/raw"
mkdir -p "$RAW"

curl -L --fail --retry 3 \
  'https://rest.uniprot.org/uniprotkb/stream?query=%28reviewed%3Atrue%29+AND+%28organism_id%3A9606%29&format=tsv&fields=accession,gene_names,protein_name,sequence,xref_pdb' \
  -o "$RAW/uniprot_reviewed_human.tsv"

for name in interactions ligands GtP_to_UniProt_mapping; do
  output="$RAW/gtopdb_${name}.tsv"
  [[ "$name" == "GtP_to_UniProt_mapping" ]] && output="$RAW/gtopdb_uniprot_mapping.tsv"
  curl -L --fail --retry 3 "https://www.guidetopharmacology.org/DATA/${name}.tsv" -o "$output"
done

echo "Science-KB source snapshot downloaded to: $RAW"
echo "Build the SQLite KB with: python scripts/build_science_kb.py --replace"
