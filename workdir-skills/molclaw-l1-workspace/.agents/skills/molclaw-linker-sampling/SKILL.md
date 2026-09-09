---
name: molclaw-linker-sampling
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__linkinvent_linker_sampling_by_warhead_pair_name`, `mcp__molclaw-scp__linkinvent_linker_sampling_by_warheads`. Generate new molecules sampling from the input two warhead fragments."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecule Generation from Warhead Fragments

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *linkinvent_linker_sampling_by_warheads*.

```tex
Generate new molecules sampling from the input two warhead fragments.
Args:
    warheads (str): SMILES of two warheads separated by '|', e.g., '*c1ccc(O)cc1|*N1CCNCC1'
    n (int): Number of molecules for sampling
    filter_preset (str): Required filter preset; options: ['none', 'minimal', 'default', 'strict'] (commonly 'default')
    lipinski (bool): Required flag controlling Lipinski filtering (commonly True)
    min_linker_atoms (int): Required minimum number of atoms in the linker (use 0 for no lower bound)
    max_linker_atoms (int): Required maximum number of atoms in the linker (use 0 for no upper bound)
Return:
    status (str): success/error
    msg (str): message
    save_smiles_file (str): Path to the saved SMILES file
    output_smiles_list (List[str]): List of generated SMILES strings
```

How to use tool *linkinvent_linker_sampling_by_warheads* :

```python
response = await client.session.call_tool(
    "linkinvent_linker_sampling_by_warheads",
    arguments={
        "warheads": warheads,
        "n": n,
        "lipinski": True,
        "filter_preset": filter_type,
        "min_linker_atoms": min_linker_atoms,
        "max_linker_atoms": max_linker_atoms
    }
)
result = client.parse_result(response)
output_smiles_list = result["output_smiles_list"]
```
