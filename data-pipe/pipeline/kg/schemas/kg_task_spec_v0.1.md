# KG Task Spec v0.1

`KGTaskSpec` defines how sampled toolchain questions from `tool-kg` are converted into executable tasks for `data-pipe`.

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
    "sample_file": "/path/to/sample_success.jsonl",
    "sample_index": 1,
    "sample_id": "sample_0001"
  },
  "toolchain": {
    "tools": ["retrieve_protein_sequence", "is_valid_protein_sequence", "pred_protein_structure_esmfold"],
    "edges": [
      {
        "source_tool": "retrieve_protein_sequence",
        "target_tool": "is_valid_protein_sequence",
        "edge_type": "generates_input_for",
        "confidence": 0.95,
        "support_scope": "full",
        "pair_id": "pair::retrieve_protein_sequence__to__is_valid_protein_sequence"
      }
    ],
    "hops": 2,
    "start_tool": "retrieve_protein_sequence",
    "end_tool": "pred_protein_structure_esmfold"
  },
  "expected_trajectory": {
    "steps": [
      {
        "step_index": 1,
        "tool_id": "retrieve_protein_sequence",
        "step_goal": "Retrieve protein sequence"
      }
    ]
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
    "schema_version": "kg_task_spec_v0.1",
    "created_by": "tool-kg Stage3",
    "difficulty": "unknown",
    "question_payload": {},
    "raw_kg_sample": {}
  }
}
```

## Required fields
- `task_id`, `task_type`, `question`
- `source.kg_run_id`, `source.sample_index`, `source.sample_id`
- `toolchain.tools`, `toolchain.edges`, `toolchain.hops`
- `expected_trajectory`
- `metadata.schema_version`

## Execution constraints
- Do **not** leak `toolchain` or `expected_trajectory` into runtime prompt.
- Store full `KGTaskSpec` in run artifacts (`question.json`) for offline audit.
- `task=kg` follows e2e-like semantics: no benchmark evaluator, no reward.
