# KG Task Spec v0.2

`KGTaskSpec v0.2` is the native contract for tool-kg Stage3 DAG sampling output.

## JSON Schema (contract)

```json
{
  "task_id": "kg_run_20260524_214801_sample_0001",
  "task_type": "kg_sampled",
  "question": "Natural language question for the execution agent.",
  "source": {
    "type": "molclaw_kg",
    "kg_project_root": "/path/to/tool-kg",
    "kg_run_id": "run_20260524_214801",
    "sample_file": "/path/to/sample_success_v2.jsonl",
    "sample_index": 1,
    "sample_id": "sample_0001"
  },
  "toolchain": {
    "tools": ["retrieve_protein_sequence", "pred_protein_structure_esmfold"],
    "edges": [
      {
        "source_tool": "retrieve_protein_sequence",
        "target_tool": "pred_protein_structure_esmfold",
        "edge_type": "generates_full_input_for",
        "confidence": 0.91,
        "pair_id": "pair::retrieve_protein_sequence__to__pred_protein_structure_esmfold",
        "view": "core",
        "relation_status": "valid"
      }
    ],
    "hops": 2,
    "start_tool": "retrieve_protein_sequence",
    "end_tool": "pred_protein_structure_esmfold"
  },
  "expected_trajectory": {
    "schema_version": "trajectory_v2_graph",
    "workflow_graph": {
      "nodes": [
        {"node_id": "llm::plan::0", "type": "llm", "llm_role": "plan"},
        {"node_id": "tool::retrieve_protein_sequence", "type": "tool", "tool_id": "retrieve_protein_sequence"},
        {"node_id": "tool::pred_protein_structure_esmfold", "type": "tool", "tool_id": "pred_protein_structure_esmfold"}
      ],
      "edges": [
        {"edge_id": "edge1", "source": "llm::plan::0", "target": "tool::retrieve_protein_sequence", "relation": "routes_to_next"}
      ]
    },
    "execution_plan": {
      "topological_order": ["llm::plan::0", "tool::retrieve_protein_sequence", "tool::pred_protein_structure_esmfold"],
      "tool_order": ["retrieve_protein_sequence", "pred_protein_structure_esmfold"]
    },
    "final_deliverable": "Expected scientific output summary"
  },
  "execution": {
    "allowed_tools": "all_molclaw",
    "must_follow_expected_trajectory": false,
    "leak_toolchain_to_agent": false
  },
  "evaluation": {
    "mode": "none",
    "notes": "KG-sampled exploratory task; execute and audit only."
  },
  "metadata": {
    "schema_version": "kg_task_spec_v0.2",
    "created_by": "tool-kg Stage3",
    "trajectory_schema_version": "trajectory_v2_graph"
  }
}
```

## Required fields
- `task_id`, `task_type`, `question`
- `source.kg_run_id`, `source.sample_index`, `source.sample_id`
- `toolchain.tools`, `toolchain.edges`, `toolchain.hops`
- `expected_trajectory.schema_version=trajectory_v2_graph`
- `expected_trajectory.workflow_graph.nodes/edges`
- `expected_trajectory.execution_plan.topological_order/tool_order`
- `metadata.schema_version`

## Execution constraints
- Do **not** leak `toolchain` or `expected_trajectory` into runtime prompt.
- Store full `KGTaskSpec` in run artifacts (`question.json`) for offline audit.
- `task=kg` follows e2e-like semantics: no benchmark evaluator, no reward.
