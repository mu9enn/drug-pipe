---
name: molclaw-admet
description: Predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of the input molecules. 
license: MIT license
metadata:
    skill-author: PJLab
---

# ADMET Properties Prediction

Note: 
- Local files are not directly accessible by the server. Please upload them to the server using `molclaw-file-transfer` before execution. 
- For PDB file inputs, it is recommended to preprocess them using `molclaw-pdbfixer` before execution.
- Please refer to skill `molclaw-scp-server` to complete tool invocation.

The description of tool *pred_mol_admet*.

```tex
Predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of the input molecules from smiles list or file.
Args:
    smiles_list (List[str]): Required list of input SMILES strings; pass [] when using smiles_file
    smiles_file (str): Required path to a TXT/CSV SMILES file; pass '' when using smiles_list
Return:
    status (str): success/error
    msg (str): message
    json_content (List[Dcit]): List of dict, each containing the keys 'smiles', 'physicochemical', 'druglikeness' and 'admet_predictions', where 'admet_predictions' includes over 90 key-value pairs representing various molecular properties 
    json_file (str): Path to the json file saving the ADMET prediction results
```

How to use tool *pred_mol_admet* :

```python
response = await client.session.call_tool(
    "pred_mol_admet",
    arguments={
        "smiles_list": smiles_list,
        "smiles_file": ''
    }
)
result = client.parse_result(response)
admet_predictions = result["json_content"]
```
