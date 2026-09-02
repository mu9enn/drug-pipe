#!/usr/bin/env python3
"""Create an immutable trajectory view with deterministic PF labels repaired."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


PROPERTY_FUNCTIONS: dict[str, Callable[[Chem.Mol], float]] = {
    "MolWt": Descriptors.MolWt,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "MolLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "HeteroAtoms": rdMolDescriptors.CalcNumHeteroatoms,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtoms": lambda mol: mol.GetNumHeavyAtoms(),
}

FINAL_PATTERN = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
CONSTRAINT_PATTERN = re.compile(
    r"^\s*-\s*([A-Za-z0-9_]+)\s*(<=|>=|<|>)\s*(-?[0-9.]+)", re.MULTILINE
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pf_prompt(prompt: str) -> tuple[list[str], list[tuple[str, str, float]]]:
    before_constraints = re.split(
        r"\n\s*Constraints\s*:", prompt, maxsplit=1, flags=re.IGNORECASE
    )[0]
    blocks = re.findall(
        r"(?:^|\n)\s*SMILES\s*:\s*\n(.*)",
        before_constraints,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not blocks:
        raise ValueError("PF prompt has no SMILES block")
    smiles = [line.strip() for line in blocks[-1].splitlines() if line.strip()]
    constraints = [
        (match.group(1), match.group(2), float(match.group(3)))
        for match in CONSTRAINT_PATTERN.finditer(prompt)
    ]
    if not smiles or not constraints:
        raise ValueError("PF prompt has no candidates or constraints")
    unsupported = sorted({name for name, _, _ in constraints} - PROPERTY_FUNCTIONS.keys())
    if unsupported:
        raise ValueError(f"unsupported PF properties: {unsupported}")
    return smiles, constraints


def solve_pf_prompt(prompt: str) -> list[str]:
    smiles, constraints = parse_pf_prompt(prompt)
    selected: list[str] = []
    for candidate in smiles:
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            raise ValueError(f"invalid candidate SMILES: {candidate}")
        values = {name: function(mol) for name, function in PROPERTY_FUNCTIONS.items()}
        checks = {
            "<=": lambda actual, limit: actual <= limit,
            ">=": lambda actual, limit: actual >= limit,
            "<": lambda actual, limit: actual < limit,
            ">": lambda actual, limit: actual > limit,
        }
        if all(checks[operator](values[name], limit) for name, operator, limit in constraints):
            selected.append(candidate)
    return selected


def _final_payload(content: str) -> dict[str, Any]:
    match = FINAL_PATTERN.search(content)
    if match is None:
        raise ValueError("terminal PF assistant message has no final_answer envelope")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_smiles"), list):
        raise ValueError("terminal PF final_answer has no selected_smiles list")
    return payload


def _corrected_terminal_content(payload: dict[str, Any], selected: list[str]) -> str:
    corrected = dict(payload)
    corrected["selected_smiles"] = selected
    corrected["summary"] = (
        f"Deterministic RDKit evaluation of every input candidate against every stated "
        f"constraint selected {len(selected)} molecule(s)."
    )
    thought = (
        "<thought>I rechecked every candidate against every stated constraint with "
        "the benchmark-validated RDKit descriptor functions. I will return the exact "
        "matching input strings in their original order.</thought>"
    )
    final = "<final_answer>" + json.dumps(
        corrected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "</final_answer>"
    return thought + "\n" + final


def audit_ms1_reference(path: Path) -> dict[str, Any]:
    checked = mismatches = 0
    mismatch_rows: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            prompt_key = next((key for key in row if key.lstrip("\ufeff") == "prompt"), None)
            if prompt_key is None:
                raise ValueError("MS-1 reference has no prompt column")
            try:
                parse_pf_prompt(row[prompt_key])
            except ValueError:
                continue
            checked += 1
            expected = [line.strip() for line in row["answer"].splitlines() if line.strip()]
            if solve_pf_prompt(row[prompt_key]) != expected:
                mismatches += 1
                mismatch_rows.append(row_number)
    if checked == 0 or mismatches:
        raise ValueError(
            f"RDKit verifier failed MS-1 reference gate: checked={checked}, "
            f"mismatches={mismatches}, rows={mismatch_rows}"
        )
    return {"property_filter_rows": checked, "exact_matches": checked, "mismatches": 0}


def repair_dataset(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    ms1_reference: Path,
) -> dict[str, Any]:
    if output_path.exists() or audit_path.exists():
        raise ValueError("refusing to overwrite an existing v8-pre output or audit")
    reference_audit = audit_ms1_reference(ms1_reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    records = pf_records = corrected_records = 0
    corrections: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as source, output_path.open("x", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            if str(record.get("id") or "").startswith("react_pf_"):
                pf_records += 1
                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) < 3:
                    raise ValueError(f"invalid PF trajectory at line {line_number}")
                expected = solve_pf_prompt(str(messages[1].get("content") or ""))
                payload = _final_payload(str(messages[-1].get("content") or ""))
                actual = payload["selected_smiles"]
                if actual != expected:
                    messages[-1]["content"] = _corrected_terminal_content(payload, expected)
                    corrected_records += 1
                    corrections.append(
                        {
                            "line": line_number,
                            "id": record.get("id"),
                            "before": actual,
                            "after": expected,
                            "missing": [item for item in expected if item not in actual],
                            "extra": [item for item in actual if item not in expected],
                        }
                    )
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "pf_ground_truth_repair_v1",
        "source": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "output": {"path": str(output_path.resolve()), "sha256": _sha256(output_path)},
        "rdkit_version": rdBase.rdkitVersion,
        "descriptor_functions": sorted(PROPERTY_FUNCTIONS),
        "reference_gate": {
            "path": str(ms1_reference.resolve()),
            "sha256": _sha256(ms1_reference),
            **reference_audit,
        },
        "records": records,
        "pf_records": pf_records,
        "corrected_records": corrected_records,
        "corrections": corrections,
    }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--ms1-reference", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            repair_dataset(args.input, args.output, args.audit, args.ms1_reference),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
