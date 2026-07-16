import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

smiles_list = [
"O=C(CNC(=O)C1CCCCC1C(=O)O)NO",
"CCN(CC(=O)NO)C(=O)C1CCCCC1C(=O)O",
"CCN(CC(=O)O)C(=O)C1CCCCC1C(=O)O",
"CN(CC(=O)NO)C(=O)C1CCCCC1C(=O)O",
"O=C(O)C1CCCCC1C(=O)CCS",
"CN(CC(=O)NO)C(=O)C1CCCC1C(=O)O",
"O=C(O)C1CCCCC1C(=O)NCCS",
"CC(C)N(CC(=O)NO)C(=O)C1CCCCC1C(=O)O",
"O=C(O)C1CCCC1C(=O)NCCS",
"C[C@H](CC(=O)N1CCCC1C(=O)O)C(=O)O"
]


def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {}

    props["MolWt"] = Descriptors.MolWt(mol)
    props["MolLogP"] = Crippen.MolLogP(mol)
    props["HBD"] = rdMolDescriptors.CalcNumHBD(mol)
    props["HBA"] = rdMolDescriptors.CalcNumHBA(mol)
    props["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    props["HeavyAtoms"] = rdMolDescriptors.CalcNumHeavyAtoms(mol)

    return props


def lipinski_ok(p):
    return (
        p["MolWt"] <= 500
        and p["HBD"] <= 5
        and p["HBA"] <= 10
        and p["MolLogP"] <= 5
    )


rows = []

for s in smiles_list:

    p = calc_props(s)

    lip = lipinski_ok(p)

    extra = (
        p["HeavyAtoms"] <= 18
        and p["TPSA"] <= 106
    )

    final = lip and extra

    rows.append({
        "SMILES": s,
        **p,
        "Lipinski_OK": lip,
        "Extra_OK": extra,
        "Final_OK": final
    })


df = pd.DataFrame(rows)
df.to_csv("case31.csv")

print(df.to_string(index=False))


print("\nSMILES satisfying ALL constraints:\n")

selected = df[df["Final_OK"]]["SMILES"]

for s in selected:
    print(s)
