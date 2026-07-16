# Slime MCP-Tool Virtual Screening SFT Data Schema Handoff

## 0. Purpose

This document defines the final standardized data schema for generating SFT training data for a slime-based MCP-tool virtual screening agent.

The target consumer is the data pipeline / post-processing project. The post-processing code should convert raw successful agent trajectories into a clean `.jsonl` dataset that can be directly consumed by slime SFT through `slime.rollout.sft_rollout.generate_rollout` and `sft_loss`.

The same schema should also preserve enough metadata for later RL rollout, reward computation, evaluation, filtering, and error analysis.

---

## 1. Design Goals

The schema is designed around six principles.

### 1.1 Slime compatibility

Each training file should be a `.jsonl` file. Each line is one JSON object. The main training field is `messages`, passed to slime with:

```bash
--prompt-data /path/to/vs_mcp_sft.jsonl
--input-key messages
--loss-type sft_loss
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--calculate-per-token-loss
--disable-compute-advantages-and-returns
--debug-train-only
```

If `tools` and `metadata` are used by the training script or downstream custom code, configure the corresponding keys when supported:

```bash
--tool-key tools
--metadata-key metadata
```

If the local slime version does not expose `--metadata-key`, keep the field name as `metadata` because slime’s default metadata key is commonly `metadata`.

### 1.2 Train only model-generated content

The model should learn to generate:

- assistant reasoning, if retained;
- assistant tool calls;
- assistant final answer;
- optional assistant report-writing steps.

The model should not learn to generate:

- system prompt;
- user task input;
- MCP tool observations;
- docking scores returned by tools;
- ADMET scores returned by tools;
- benchmark ground truth;
- hidden evaluation labels.

Therefore, every message must carry an explicit `step_loss_mask`:

```text
system/user/tool/environment messages: step_loss_mask = 0
assistant messages: step_loss_mask = 1
```

### 1.3 Preserve scientific provenance

Every SFT sample must be traceable back to:

- original task source;
- raw trajectory file;
- target protein / target identifier;
- candidate molecule set;
- MCP tools used;
- intermediate tool outputs;
- final answer;
- quality checks;
- benchmark information, if available.

This is stored in `metadata`, not necessarily exposed in `messages`.

### 1.4 Avoid leakage

Ground-truth ranking, wet-lab labels, IC50/Ki/EC50 values, active/inactive labels, and evaluation scores must not appear in the user prompt or assistant trajectory unless they were genuinely available to the agent during the original task.

They may be stored in `metadata.ground_truth` for later reward/eval, but must not be inserted into `messages`.

### 1.5 Make SFT and RL data compatible

The SFT dataset should include complete successful trajectories. Later RL prompt data can be derived from the same record by keeping only:

- `system` message;
- initial `user` task message;
- available `tools`;
- `metadata` needed for reward.

This allows one canonical data-processing pipeline rather than separate incompatible SFT/RL schemas.

### 1.6 Make validation deterministic

The post-processing code should be able to deterministically reject samples with:

- invalid JSON;
- invalid message role;
- missing required fields;
- invalid or unsupported tool names;
- malformed tool-call JSON;
- missing tool observation after a tool call;
- final answer not parseable as the required output format;
- candidate molecules missing from final ranking;
- hallucinated tool outputs;
- leaked ground truth in prompt.

---

## 2. Output File Contract

The post-processing pipeline should produce the following files.

```text
outputs/
├── vs_mcp_sft_train.jsonl
├── vs_mcp_sft_valid.jsonl
├── vs_mcp_rl_prompts_train.jsonl        optional, derived from SFT records
├── vs_mcp_rl_prompts_valid.jsonl        optional, derived from SFT records
├── rejected_samples.jsonl
├── dataset_manifest.json
└── schema_validation_report.md
```

### 2.1 `vs_mcp_sft_train.jsonl`

Main SFT training file. One line per successful trajectory.

### 2.2 `vs_mcp_sft_valid.jsonl`

Held-out validation SFT file. Same schema as train.

### 2.3 `vs_mcp_rl_prompts_train.jsonl`

Optional RL prompt file derived from the SFT records. It should contain only the initial task prompt and metadata, not the successful assistant trajectory.

### 2.4 `rejected_samples.jsonl`

Every rejected raw trajectory should be saved here with rejection reason. This is mandatory for debugging data quality.

### 2.5 `dataset_manifest.json`

Dataset-level statistics and provenance.

### 2.6 `schema_validation_report.md`

Human-readable summary of validation results.

---

## 3. Top-Level JSON Schema

Each line in the final SFT `.jsonl` file should follow this top-level structure:

```json
{
  "schema_version": "vs-mcp-sft-v1",
  "id": "vs_sft_000001",
  "messages": [...],
  "tools": [...],
  "metadata": {...}
}
```

### 3.1 Required top-level fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `schema_version` | string | yes | Fixed version string. Use `vs-mcp-sft-v1` for the first implementation. |
| `id` | string | yes | Globally unique sample id. Stable across reruns if raw input does not change. |
| `messages` | array | yes | OpenAI-style conversation messages with explicit `step_loss_mask`. |
| `tools` | array | yes | Tool schemas available to the agent for this sample. Can be full list or task-relevant subset. |
| `metadata` | object | yes | Non-training metadata for provenance, reward, filtering, and evaluation. |

### 3.2 Optional top-level fields

| Field | Type | Description |
|---|---:|---|
| `tags` | array[string] | Human-readable tags, e.g. `['CARA', 'EGFR', 'docking_success']`. |
| `split` | string | `train`, `valid`, or `test`. Prefer also storing this in `metadata.split`. |
| `created_at` | string | ISO timestamp for processed record creation. |

---

## 4. `messages` Field

`messages` is the only field directly used as SFT input when `--input-key messages` is set.

### 4.1 Message object schema

Each message should have:

```json
{
  "role": "assistant",
  "content": "<tool_call>{...}</tool_call>",
  "step_loss_mask": 1,
  "metadata": {
    "event_id": "evt_0003",
    "event_type": "tool_call"
  }
}
```

### 4.2 Required message fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `role` | string | yes | One of `system`, `user`, `assistant`, `tool`. If tokenizer does not support `tool`, use compatibility mode described below. |
| `content` | string | yes | Message text. Must be non-empty after stripping whitespace unless explicitly allowed. |
| `step_loss_mask` | int | yes | `1` for assistant tokens to train; `0` for all non-assistant messages. |

### 4.3 Optional message fields

| Field | Type | Description |
|---|---:|---|
| `name` | string | For `tool` role, the tool name. Useful if chat template supports named tool messages. |
| `metadata` | object | Event-level metadata. Not required by slime, but useful for debugging. |

### 4.4 Allowed roles and masks

| Role | Meaning | `step_loss_mask` |
|---|---|---:|
| `system` | Global behavior contract, tool-use rules, output format. | 0 |
| `user` | Task input or environment observation fallback. | 0 |
| `assistant` | Model-generated reasoning, tool call, or final answer. | 1 |
| `tool` | MCP server observation. | 0 |

### 4.5 Preferred message order

A successful trajectory should follow this pattern:

```text
system
user
assistant tool_call
/tool observation
assistant tool_call
/tool observation
...
assistant final_answer
```

In JSON:

```json
[
  {"role": "system", "content": "...", "step_loss_mask": 0},
  {"role": "user", "content": "...", "step_loss_mask": 0},
  {"role": "assistant", "content": "<tool_call>{...}</tool_call>", "step_loss_mask": 1},
  {"role": "tool", "name": "validate_smiles", "content": "<tool_response>{...}</tool_response>", "step_loss_mask": 0},
  {"role": "assistant", "content": "<tool_call>{...}</tool_call>", "step_loss_mask": 1},
  {"role": "tool", "name": "run_docking", "content": "<tool_response>{...}</tool_response>", "step_loss_mask": 0},
  {"role": "assistant", "content": "<answer>[\"SMILES_2\", \"SMILES_1\"]</answer>", "step_loss_mask": 1}
]
```

---

## 5. Tool-Role Compatibility Mode

Some model chat templates may not support `role: "tool"`. In that case, the pipeline should support a compatibility mode.

### 5.1 Preferred mode

Use true `tool` role:

```json
{"role": "tool", "name": "run_docking", "content": "<tool_response>{...}</tool_response>", "step_loss_mask": 0}
```

### 5.2 Compatibility mode

Convert tool observations into `user` messages:

```json
{"role": "user", "content": "<observation tool_name=\"run_docking\">{...}</observation>", "step_loss_mask": 0}
```

### 5.3 Rule

Regardless of role representation, tool observations must always have:

```json
"step_loss_mask": 0
```

The model must not be trained to generate tool observations.

---

## 6. Tool Call Format

All assistant tool calls should use a strict wrapper:

```text
<tool_call>{"name":"TOOL_NAME","arguments":{...}}</tool_call>
```

### 6.1 Tool call JSON schema

Inside `<tool_call>...</tool_call>`, content must be valid JSON:

```json
{
  "name": "validate_smiles",
  "arguments": {
    "smiles_list": ["CCO", "c1ccccc1"]
  }
}
```

### 6.2 Required tool call fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `name` | string | yes | Tool name. Must exist in top-level `tools`. |
| `arguments` | object | yes | Tool arguments. Must satisfy the tool schema. |

### 6.3 Forbidden tool call behavior

Reject the sample if an assistant message:

- contains a tool call with invalid JSON;
- calls a tool not listed in `tools`;
- passes arguments not satisfying the tool schema;
- calls a tool but no subsequent observation exists;
- includes fake observation text inside the assistant message;
- claims a tool result before the corresponding tool observation.

### 6.4 Multiple tool calls in one assistant message

For v1, disallow multiple tool calls in one assistant message. Require exactly one tool call per assistant message when `event_type = tool_call`.

Reason: one tool call per assistant turn makes post-processing, replay, loss masking, and reward attribution much easier.

---

## 7. Tool Observation Format

Every tool observation should use a strict wrapper:

```text
<tool_response>{...}</tool_response>
```

The content inside the wrapper should be valid JSON whenever possible.

Example:

```json
{
  "status": "success",
  "tool_name": "validate_smiles",
  "result": {
    "valid_smiles": ["CCO", "c1ccccc1"],
    "invalid_smiles": []
  },
  "runtime_sec": 0.13
}
```

### 7.1 Required observation fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `status` | string | yes | `success`, `failed`, `timeout`, or `skipped`. |
| `tool_name` | string | yes | Must match the immediately preceding tool call. |
| `result` | object/string/null | yes | Tool output. Should be compact and parseable. |
| `runtime_sec` | number | no | Tool runtime in seconds. |
| `error` | object/string/null | no | Error details if failed. |

### 7.2 Observation length control

Tool responses can be very long. The post-processing code should support observation compression:

- keep structured scores and paths;
- remove excessive logs;
- truncate verbose stdout/stderr;
- store full raw output in external artifact path if needed;
- record truncation in metadata.

Recommended fields:

```json
{
  "status": "success",
  "tool_name": "run_docking",
  "result": {
    "scores": {
      "CCO": -6.1,
      "c1ccccc1": -7.2
    },
    "output_files": ["artifacts/docking/task_001/results.csv"]
  },
  "runtime_sec": 48.6,
  "truncated": false
}
```

---

## 8. `tools` Field

The `tools` field records tool schemas available to the model.

### 8.1 Tool schema format

Use an OpenAI-function-like schema:

```json
{
  "type": "function",
  "function": {
    "name": "validate_smiles",
    "description": "Validate and canonicalize a list of SMILES strings.",
    "parameters": {
      "type": "object",
      "properties": {
        "smiles_list": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Input candidate SMILES strings."
        }
      },
      "required": ["smiles_list"]
    }
  }
}
```

### 8.2 Required tool fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `type` | string | yes | Should be `function`. |
| `function.name` | string | yes | Tool name exactly as exposed by MCP server. |
| `function.description` | string | yes | Concise tool description. |
| `function.parameters` | object | yes | JSON schema for arguments. |

### 8.3 Full tool list vs task-relevant subset

For v1 SFT, use the task-relevant subset of tools, not all MCP tools.

Recommended:

```text
5-12 tools per sample
```

Reason:

- reduces context length;
- improves learning signal;
- avoids confusing the model with irrelevant tools;
- makes tool-call correctness easier to validate.

For later generalization experiments, a full tool registry variant can be created separately.

### 8.4 Tool registry in system prompt

Depending on tokenizer/chat-template behavior, the `tools` field may or may not be automatically rendered into the model input. To avoid ambiguity, v1 should also include a compact rendered tool registry in the `system` message.

Example:

```text
Available MCP tools:
<tools>
{"name":"validate_smiles","description":"Validate and canonicalize SMILES.","parameters":{...}}
{"name":"run_docking","description":"Run docking and return binding scores.","parameters":{...}}
</tools>

When using a tool, output exactly one JSON object inside <tool_call>...</tool_call>.
```

The top-level `tools` field remains the machine-readable source of truth. The system-prompt tool registry is the model-visible rendering.

---

## 9. `metadata` Field

`metadata` is not directly optimized by SFT loss. It is used for provenance, filtering, RL reward, evaluation, and debugging.

### 9.1 Recommended metadata schema

```json
{
  "split": "train",
  "source": {
    "dataset": "CARA",
    "task_family": "virtual_screening",
    "task_id": "cara_egfr_000001",
    "raw_trajectory_path": "raw/cara_egfr_000001.json",
    "postprocess_version": "2026-05-18-v1"
  },
  "target": {
    "target_id": "EGFR",
    "target_name": "Epidermal growth factor receptor",
    "uniprot_id": "P00533",
    "pdb_id": "1M17",
    "protein_file": "artifacts/proteins/1M17_prepared.pdbqt"
  },
  "candidates": [
    {
      "candidate_id": "mol_0001",
      "input_smiles": "CCO",
      "canonical_smiles": "CCO",
      "name": null
    }
  ],
  "ground_truth": {
    "available": true,
    "label_type": "ranking_or_activity",
    "activity_type": "IC50",
    "activity_unit": "nM",
    "values": {
      "mol_0001": 120.0
    },
    "active_labels": {
      "mol_0001": 1
    },
    "ranking_best_to_worst": ["mol_0001", "mol_0002"]
  },
  "trajectory": {
    "num_assistant_turns": 3,
    "num_tool_calls": 2,
    "tool_names": ["validate_smiles", "run_docking"],
    "all_tool_calls_executed": true,
    "all_observations_real": true,
    "contains_error_recovery": false,
    "max_turns_exceeded": false
  },
  "outputs": {
    "final_answer_format": "answer_json_smiles_ranking",
    "final_ranked_smiles": ["CCO", "c1ccccc1"],
    "final_ranked_candidate_ids": ["mol_0001", "mol_0002"],
    "result_md_path": "artifacts/results/cara_egfr_000001/result.md"
  },
  "quality": {
    "accepted": true,
    "quality_score": 0.97,
    "rejection_reasons": [],
    "warnings": [],
    "token_count_estimate": 5200,
    "observation_truncated": false,
    "ground_truth_leakage_detected": false
  }
}
```

### 9.2 Required metadata fields

The following fields should be required in v1:

```text
metadata.split
metadata.source.dataset
metadata.source.task_family
metadata.source.task_id
metadata.target.target_id
metadata.candidates
metadata.trajectory.num_tool_calls
metadata.trajectory.tool_names
metadata.trajectory.all_tool_calls_executed
metadata.outputs.final_answer_format
metadata.outputs.final_ranked_smiles
metadata.quality.accepted
metadata.quality.rejection_reasons
```

### 9.3 Ground-truth metadata

`metadata.ground_truth` is optional but recommended when benchmark labels exist.

Important rule:

```text
metadata.ground_truth may exist, but its contents must not appear in messages unless they were explicitly given to the agent as part of the original task.
```

This prevents train/eval leakage.

---

## 10. Final Answer Format

For virtual screening ranking tasks, final answer should be strict:

```text
<answer>["SMILES_1", "SMILES_2", "SMILES_3"]</answer>
```

### 10.1 Required final-answer constraints

The post-processing validator should check:

1. final answer exists;
2. final answer is inside `<answer>...</answer>`;
3. inner content is valid JSON;
4. JSON content is a list of strings;
5. every string is a candidate SMILES from the task;
6. no duplicate SMILES;
7. ranking length equals candidate count unless task explicitly requests top-k;
8. if canonicalization is used, output can be mapped back to candidate IDs;
9. no extra text outside `<answer>` unless allowed by task.

### 10.2 Candidate ID vs SMILES

The model-facing final answer should use SMILES if the benchmark requires SMILES ranking.

Internally, metadata should also store candidate IDs:

```json
"outputs": {
  "final_ranked_smiles": ["CCO", "c1ccccc1"],
  "final_ranked_candidate_ids": ["mol_0001", "mol_0002"]
}
```

Reason: SMILES strings may be canonicalized or duplicated. Candidate IDs make evaluation deterministic.

---

## 11. Complete Example Record

```json
{
  "schema_version": "vs-mcp-sft-v1",
  "id": "vs_sft_cara_egfr_000001",
  "messages": [
    {
      "role": "system",
      "content": "You are a computational medicinal chemistry agent for virtual screening. You may call MCP tools when needed. Do not fabricate tool outputs. Available MCP tools:\n<tools>\n{\"name\":\"validate_smiles\",\"description\":\"Validate and canonicalize SMILES strings.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"smiles_list\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"smiles_list\"]}}\n{\"name\":\"run_docking\",\"description\":\"Run docking for candidate ligands against a prepared target.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"target_id\":{\"type\":\"string\"},\"smiles_list\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"target_id\",\"smiles_list\"]}}\n</tools>\nWhen using a tool, output exactly one JSON object inside <tool_call>...</tool_call>. Final answer must be <answer>[...]</answer>, a JSON array of SMILES ordered from best predicted binder to worst predicted binder.",
      "step_loss_mask": 0
    },
    {
      "role": "user",
      "content": "Task: Rank the following candidate molecules for target EGFR from best predicted binder to worst predicted binder.\nCandidate SMILES: [\"CCO\", \"c1ccccc1\"]",
      "step_loss_mask": 0
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\":\"validate_smiles\",\"arguments\":{\"smiles_list\":[\"CCO\",\"c1ccccc1\"]}}</tool_call>",
      "step_loss_mask": 1,
      "metadata": {"event_type": "tool_call", "tool_name": "validate_smiles"}
    },
    {
      "role": "tool",
      "name": "validate_smiles",
      "content": "<tool_response>{\"status\":\"success\",\"tool_name\":\"validate_smiles\",\"result\":{\"valid_smiles\":[\"CCO\",\"c1ccccc1\"],\"invalid_smiles\":[],\"canonical_smiles\":{\"CCO\":\"CCO\",\"c1ccccc1\":\"c1ccccc1\"}},\"runtime_sec\":0.08}</tool_response>",
      "step_loss_mask": 0,
      "metadata": {"event_type": "tool_observation", "tool_name": "validate_smiles"}
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\":\"run_docking\",\"arguments\":{\"target_id\":\"EGFR\",\"smiles_list\":[\"CCO\",\"c1ccccc1\"]}}</tool_call>",
      "step_loss_mask": 1,
      "metadata": {"event_type": "tool_call", "tool_name": "run_docking"}
    },
    {
      "role": "tool",
      "name": "run_docking",
      "content": "<tool_response>{\"status\":\"success\",\"tool_name\":\"run_docking\",\"result\":{\"scores\":{\"CCO\":-6.1,\"c1ccccc1\":-7.2},\"score_unit\":\"kcal/mol\"},\"runtime_sec\":42.5}</tool_response>",
      "step_loss_mask": 0,
      "metadata": {"event_type": "tool_observation", "tool_name": "run_docking"}
    },
    {
      "role": "assistant",
      "content": "<answer>[\"c1ccccc1\",\"CCO\"]</answer>",
      "step_loss_mask": 1,
      "metadata": {"event_type": "final_answer"}
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "validate_smiles",
        "description": "Validate and canonicalize SMILES strings.",
        "parameters": {
          "type": "object",
          "properties": {
            "smiles_list": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["smiles_list"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "run_docking",
        "description": "Run docking for candidate ligands against a prepared target and return binding scores.",
        "parameters": {
          "type": "object",
          "properties": {
            "target_id": {"type": "string"},
            "smiles_list": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["target_id", "smiles_list"]
        }
      }
    }
  ],
  "metadata": {
    "split": "train",
    "source": {
      "dataset": "CARA",
      "task_family": "virtual_screening",
      "task_id": "cara_egfr_000001",
      "raw_trajectory_path": "raw/cara_egfr_000001.json",
      "postprocess_version": "2026-05-18-v1"
    },
    "target": {
      "target_id": "EGFR",
      "target_name": "Epidermal growth factor receptor",
      "uniprot_id": null,
      "pdb_id": null,
      "protein_file": null
    },
    "candidates": [
      {"candidate_id": "mol_0001", "input_smiles": "CCO", "canonical_smiles": "CCO", "name": null},
      {"candidate_id": "mol_0002", "input_smiles": "c1ccccc1", "canonical_smiles": "c1ccccc1", "name": null}
    ],
    "ground_truth": {
      "available": false,
      "label_type": null,
      "activity_type": null,
      "activity_unit": null,
      "values": {},
      "active_labels": {},
      "ranking_best_to_worst": []
    },
    "trajectory": {
      "num_assistant_turns": 3,
      "num_tool_calls": 2,
      "tool_names": ["validate_smiles", "run_docking"],
      "all_tool_calls_executed": true,
      "all_observations_real": true,
      "contains_error_recovery": false,
      "max_turns_exceeded": false
    },
    "outputs": {
      "final_answer_format": "answer_json_smiles_ranking",
      "final_ranked_smiles": ["c1ccccc1", "CCO"],
      "final_ranked_candidate_ids": ["mol_0002", "mol_0001"],
      "result_md_path": null
    },
    "quality": {
      "accepted": true,
      "quality_score": 1.0,
      "rejection_reasons": [],
      "warnings": [],
      "token_count_estimate": 1800,
      "observation_truncated": false,
      "ground_truth_leakage_detected": false
    }
  }
}
```

---

## 12. Post-Processing Pipeline Requirements

The post-processing pipeline should implement the following stages.

### 12.1 Load raw trajectory

Input raw trajectory may contain:

- original prompt;
- model messages;
- MCP tool calls;
- MCP tool outputs;
- artifact paths;
- final answer;
- benchmark labels;
- execution logs.

The loader should normalize them into an internal event list.

### 12.2 Normalize task fields

Normalize:

- target ID;
- target name;
- PDB / UniProt IDs;
- candidate SMILES;
- canonical SMILES;
- candidate IDs;
- dataset source;
- benchmark labels.

### 12.3 Validate tool calls

For every assistant tool call:

1. parse `<tool_call>` wrapper;
2. parse JSON;
3. check `name` exists in `tools`;
4. check arguments satisfy schema;
5. check next event is matching tool observation;
6. check observation is real tool output, not model-generated text.

### 12.4 Normalize observations

For every tool output:

- convert to `<tool_response>{...}</tool_response>`;
- compress long logs;
- keep scientific result fields;
- keep runtime and status;
- preserve artifact path if full output is external.

### 12.5 Build messages

Construct messages in this order:

1. system prompt with behavior rules and tool registry;
2. user task input;
3. assistant/tool alternating trajectory;
4. final assistant answer.

Assign `step_loss_mask`:

```python
if role == "assistant":
    step_loss_mask = 1
else:
    step_loss_mask = 0
```

### 12.6 Validate final answer

Parse final answer and map it to candidates.

Reject if:

- no final answer;
- invalid wrapper;
- invalid JSON;
- non-list output;
- duplicate molecule;
- unknown SMILES;
- missing required candidate;
- ranking cannot be mapped to candidate IDs.

### 12.7 Detect leakage

Search messages for benchmark-only information:

- IC50/Ki/EC50 values not given in task;
- active/inactive labels;
- ground-truth ranking;
- evaluation score;
- hidden dataset labels.

If leakage is found, reject or mark with:

```json
"ground_truth_leakage_detected": true
```

For SFT training, reject leakage by default.

### 12.8 Estimate token length

Estimate token count after chat-template rendering. If tokenizer is unavailable, use a conservative character-based estimate.

Recommended policies:

```text
soft warning: > 75% of max context
reject or truncate: > max context
```

If observations are truncated, record:

```json
"metadata.quality.observation_truncated": true
```

### 12.9 Write accepted and rejected samples

Accepted samples go to train/valid jsonl. Rejected samples go to `rejected_samples.jsonl` with reason.

---

## 13. Rejection Reason Taxonomy

Use fixed rejection reason strings for easier statistics.

```text
invalid_json
missing_required_top_level_field
missing_messages
invalid_message_role
invalid_step_loss_mask
missing_system_message
missing_user_task
missing_final_answer
invalid_tool_call_wrapper
invalid_tool_call_json
unknown_tool_name
invalid_tool_arguments
missing_tool_observation
mismatched_tool_observation
assistant_contains_fake_observation
tool_observation_not_real
invalid_tool_response_json
final_answer_invalid_wrapper
final_answer_invalid_json
final_answer_not_list
final_answer_unknown_smiles
final_answer_duplicate_smiles
final_answer_missing_candidates
cannot_map_smiles_to_candidate_id
ground_truth_leakage
token_length_exceeded
empty_assistant_message
empty_tool_response
raw_trajectory_failed
manual_blacklist
```

Each rejected record should include:

```json
{
  "raw_id": "...",
  "rejection_reasons": ["..."],
  "raw_trajectory_path": "...",
  "debug_excerpt": "..."
}
```

---

## 14. Dataset Manifest

The pipeline should generate `dataset_manifest.json`:

```json
{
  "schema_version": "vs-mcp-sft-v1",
  "created_at": "2026-05-18T00:00:00Z",
  "postprocess_version": "2026-05-18-v1",
  "input_sources": ["raw/cara/*.json"],
  "output_files": {
    "train": "vs_mcp_sft_train.jsonl",
    "valid": "vs_mcp_sft_valid.jsonl",
    "rejected": "rejected_samples.jsonl"
  },
  "counts": {
    "raw": 10000,
    "accepted": 8200,
    "rejected": 1800,
    "train": 7800,
    "valid": 400
  },
  "tool_stats": {
    "validate_smiles": 8200,
    "run_docking": 7900
  },
  "rejection_stats": {
    "invalid_tool_call_json": 120,
    "final_answer_unknown_smiles": 83
  },
  "token_stats": {
    "mean": 4200,
    "p50": 3900,
    "p95": 9000,
    "max": 14000
  }
}
```

---

## 15. Derived RL Prompt Schema

For later RL, derive prompt-only data from SFT records.

### 15.1 RL prompt record

```json
{
  "schema_version": "vs-mcp-rl-prompt-v1",
  "id": "vs_rl_cara_egfr_000001",
  "messages": [
    {"role": "system", "content": "...same system prompt...", "step_loss_mask": 0},
    {"role": "user", "content": "...task input...", "step_loss_mask": 0}
  ],
  "tools": [...],
  "metadata": {...}
}
```

### 15.2 Rule

RL prompt records must not contain successful assistant trajectory or tool observations.

They should preserve metadata needed by reward:

- target;
- candidates;
- benchmark labels;
- available tools;
- source task ID.

---

## 16. Recommended Slime SFT Arguments

The first SFT run can use:

```bash
SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data /path/to/vs_mcp_sft_train.jsonl
  --input-key messages
  --rollout-shuffle
  --num-epoch 3
  --rollout-batch-size 128
  --global-batch-size 128
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
```

If supported by the local slime version:

```bash
  --tool-key tools
  --metadata-key metadata
```

If tool rendering is not reliable through `--tool-key`, keep tools explicitly rendered inside the system prompt.

---

## 17. Implementation Notes for Codex / Data Pipeline

### 17.1 Pseudocode overview

```python
def process_raw_trajectory(raw):
    record = {}
    record["schema_version"] = "vs-mcp-sft-v1"
    record["id"] = make_stable_id(raw)

    tools = build_tool_schemas(raw)
    task = normalize_task(raw)
    events = normalize_events(raw)

    validate_event_sequence(events, tools)

    messages = []
    messages.append(build_system_message(tools))
    messages.append(build_user_message(task))

    for event in events:
        if event.type == "assistant_tool_call":
            messages.append(build_assistant_tool_call_message(event))
        elif event.type == "tool_observation":
            messages.append(build_tool_observation_message(event))
        elif event.type == "assistant_final_answer":
            messages.append(build_final_answer_message(event))

    validate_loss_masks(messages)
    final_outputs = parse_and_validate_final_answer(messages, task.candidates)
    leakage = detect_ground_truth_leakage(messages, task.ground_truth)
    if leakage:
        reject("ground_truth_leakage")

    metadata = build_metadata(raw, task, events, final_outputs)

    record["messages"] = messages
    record["tools"] = tools
    record["metadata"] = metadata

    validate_record(record)
    return record
```

### 17.2 Stable ID rule

Use deterministic IDs:

```text
vs_sft_{dataset}_{task_id}_{short_hash(raw_trajectory_or_canonical_task)}
```

Example:

```text
vs_sft_cara_cara_egfr_000001_a13f92c8
```

### 17.3 Split rule

Split by task/target, not by individual trajectory, when possible.

Reason: if the same target/candidate set appears in train and valid, validation may overestimate generalization.

Recommended:

```text
train: 90%
valid: 10%
```

For benchmark evaluation, keep a separate test split that is not used for SFT or RL training.

---

## 18. Critical Do-Not Rules

The post-processing pipeline must not:

1. put tool observations into assistant messages;
2. set `step_loss_mask = 1` for tool observations;
3. fabricate missing docking/ADMET/protein outputs;
4. include ground-truth labels in model-visible prompt unless originally provided;
5. silently drop candidates from final ranking;
6. silently convert failed tool calls into successful calls;
7. hide rejected samples;
8. allow unknown tools;
9. allow unparseable final answers;
10. train on trajectories where the final answer cannot be mapped back to candidate IDs.

---

## 19. Minimal Acceptance Checklist

A sample is accepted only if all checks pass:

```text
[ ] top-level fields exist: schema_version, id, messages, tools, metadata
[ ] messages is non-empty and starts with system + user
[ ] every message has role, content, step_loss_mask
[ ] non-assistant messages have step_loss_mask = 0
[ ] assistant messages have step_loss_mask = 1
[ ] every tool call is valid JSON
[ ] every tool call name exists in tools
[ ] every tool call has matching real observation
[ ] final answer exists and is parseable
[ ] final ranking maps to candidate IDs
[ ] no duplicate final candidates
[ ] no hidden ground-truth leakage in messages
[ ] token length is within configured max context
[ ] metadata.quality.accepted = true
```

---

## 20. Summary

The final SFT schema is:

```text
one jsonl line = one complete successful MCP-tool trajectory
messages = model-visible conversation with explicit loss masks
tools = machine-readable tool schemas
metadata = non-training provenance/reward/eval information
assistant turns = trainable
tool observations = non-trainable
final answer = strict ranked SMILES JSON array
```

This schema gives the data pipeline a stable post-processing target and keeps the later RL stage compatible with the same task, tool, and benchmark metadata.

