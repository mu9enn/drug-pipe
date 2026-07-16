from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

smiles_list = [
"C[C@@H]1NC(=O)[C@H](CN)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC1=O",
"C[C@@H]1NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C1=O",
"N=C(N)NCCC[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](CN)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O",
"N=C(N)NCCC[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O",
"N=C(N)NCCC[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O",
"N=C(N)NCCC[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](Cc2cnc[nH]2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O",
"C[C@@H]1NC(=O)[C@H](CN)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C1=O",
"N=C(N)NCCC[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](Cc2cnc[nH]2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](CN)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O",
"C[C@@H]1NC(=O)[C@H](CN)NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](Cc2cnc[nH]2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C1=O",
"N=C(N)NCCC[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](Cc2cnc[nH]2)NC(=O)[C@@H]2CCCN2C(=O)[C@H]2CCCN2C(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O"
]

def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

query = Chem.MolFromSmiles(smiles_list[0])
query_fp = morgan_fp(query)

best_smiles = None
best_sim = -1

print("Similarity scores:\n")

for i, smi in enumerate(smiles_list[1:], start=1):
    mol = Chem.MolFromSmiles(smi)
    fp = morgan_fp(mol)

    sim = TanimotoSimilarity(query_fp, fp)

    print(f"{i}: {sim:.4f}")

    if sim > best_sim:
        best_sim = sim
        best_smiles = smi

print("\nMost similar SMILES:")
print(best_smiles)
print("Similarity:", best_sim)
