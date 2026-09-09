---
name: molclaw-p2rank
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__pred_pocket_prank`. Use P2Rank to locate binding pockets in the input protein. Unless specified by the user, prioritize using fpocket."
license: MIT license
metadata:
    skill-author: PJLab
---

# Pocket Location

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *pred_pocket_prank*.

```tex
Use P2Rank to predict ligand binding pockets in the input protein.
Args:
    pdb_file_path (str): Path to the protein structure file (PDB format)
Return:
    status (str): success/error
    msg (str): message
    pred_pockets (List[dict]): List of dict, each containing pocket confidence and center position information. The first pocket (pred_pockets[0]) has the highest score and is usually used for molecular docking.
        --site_id (str): Pocket id
        --probability (float): Predicted confidence score (0~1) of the pocket
        --center_x (float): Center X of the pocket
        --center_y (float): Center Y of the pocket 
        --center_z (float): Center Z of the pocket
```

How to use tool *pred_pocket_prank* :

```python
response = await client.session.call_tool(
    "pred_pocket_prank",
    arguments={
        "pdb_file_path": pdb_file_path
    }
)
result = client.parse_result(response)
pred_pockets = result["pred_pockets"]
```
