from __future__ import annotations

from drug_agent.scripts.repair_pf_ground_truth import parse_pf_prompt, solve_pf_prompt


def test_pf_solver_preserves_input_strings_and_order():
    prompt = """Task:
SMILES:
CC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CO

Constraints:
- MolWt <= 100
- NumHDonors <= 5
- NumHAcceptors <= 10
- MolLogP <= 5
"""
    smiles, constraints = parse_pf_prompt(prompt)
    assert smiles == ["CC", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "CO"]
    assert constraints[0] == ("MolWt", "<=", 100.0)
    assert solve_pf_prompt(prompt) == ["CC", "CO"]


def test_pf_solver_applies_additional_constraints_conjunctively():
    prompt = """SMILES:
CC
CO
CCO

Constraints:
- MolWt <= 500
- NumHDonors <= 5
- NumHAcceptors <= 10
- MolLogP <= 5
- HeteroAtoms >= 1
- HeavyAtoms <= 2
"""
    assert solve_pf_prompt(prompt) == ["CO"]
