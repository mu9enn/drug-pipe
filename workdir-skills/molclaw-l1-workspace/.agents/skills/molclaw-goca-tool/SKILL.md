---
name: molclaw-goca-tool
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__goca_pipeline`. Run GoCa coarse-grained protein MD pipeline and collect key simulation artifacts from a unified run directory."
license: MIT license
metadata:
    skill-author: PJLab
---

# GoCa Pipeline

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

- GoCa executable path is fixed by the managed wrapper to `/data/lwj/wll/code/drug/GoCa/GoCa`.


## Usage

### 1. GoCa Pipeline
The description of tool *goca_pipeline*.

```tex
Runs GoCa coarse-grained setup and optional full MD workflow for protein structure relaxation and trajectory generation.
Args:
    protein_pdb (str): Input protein PDB path, required.
    full_md (bool): Whether to run EM, production MD, and post-processing, default True.
    temperature (float): GoCa reduced temperature used for MD, default 45.0.
    md_time (float): MD simulation length in ps, default 12000.0.
    gpu_ids (str | None): Optional GROMACS GPU device IDs, default None.
    dry_run (bool): Create tracked run directory and return normalized parameters without execution, default False.
Return:
    status (str): success, partial_success, or error.
    msg (str): Human-readable run summary.
    output_dir (str): Run-specific directory under tool_result/goca_pipeline_result.
    work_dir (str): Relative GoCa working directory under output_dir.
    protein_pdb (str): Resolved input protein PDB absolute path.
    full_md (bool): Effective full_md value used by wrapper.
    temperature (float): Effective reduced temperature used by wrapper.
    md_time (float): Effective MD time in ps used by wrapper.
    gpu_ids (str | None): Effective GPU IDs used by wrapper.
    dry_run (bool): Effective dry_run value used by wrapper.
    key_files (dict): Key output files relative to output_dir.
    analysis_dir (str | None): Analysis directory relative to output_dir when generated.
```

How to use tool *goca_pipeline* :

```python
response = await client.session.call_tool(
    "goca_pipeline",
    arguments={
        "protein_pdb": "/path/to/input.pdb",
        "full_md": True,
        "md_time": 1000.0,
        "temperature": 45.0,
        "gpu_ids": None,
        "dry_run": False
    }
)
result = client.parse_result(response)
key_output = result["output_dir"]

```

#### Example parameter sets

```python
# 1) Main mode
{
    "protein_pdb": "/path/to/input.pdb",
    "full_md": True,
    "md_time": 1000.0,
    "temperature": 45.0,
    "gpu_ids": None,
    "dry_run": True
}

# 2) Variant mode
{
    "protein_pdb": "relative/path/to/protein.pdb",
    "full_md": False,
    "md_time": 50000.0,
    "temperature": 50.0,
    "gpu_ids": "0",
    "dry_run": False
}
```
