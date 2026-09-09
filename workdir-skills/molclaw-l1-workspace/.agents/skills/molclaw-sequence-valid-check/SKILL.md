---
name: molclaw-sequence-valid-check
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__is_valid_protein_sequence`. Check if the input protein sequence is valid."
license: MIT license
metadata:
    skill-author: PJLab
---

# Protein Sequence Valid Check

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *is_valid_protein_sequence*.

```tex
Check if the input protein sequence string is valid.
Args:
    sequences (List[str]): List of input protein sequences
Return:
    status (str): success/partial_success/error
    msg (str): message
    valid_res (List[dict]): List of dict, each containing the keys 'sequence' and 'is_valid'.
        --sequence (str): A protein sequence of the input sequences list 
        --is_valid (bool): Is the protein sequence valid or not
```

How to use tool *is_valid_protein_sequence* :

```python

response = await client.session.call_tool(
    "is_valid_protein_sequence",
    arguments={
        "sequences": sequence_list
    }
)
result = client.parse_result(response)
valid_res = result["valid_res"]
```
