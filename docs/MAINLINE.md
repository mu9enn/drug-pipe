# Mainline

## Authorities

| Fact | Authority |
| --- | --- |
| Tool schema and names | The exact tool manifest visible to the student at deployment/rollout |
| Raw assistant decisions/calls/results | Claude stream-json; one `message.id` is one decision |
| Training path normalization | Source-aware deterministic Python contract |
| Semantic trajectory | `drug_agent_semantic_trajectory_v1` |
| Reasoning prose edits | Restricted `semantic_reasoning_patch_v1` |
| Qwen system prompt and native message projection | Qwen3.5 adapter |
| Wire-format rendering | Checkpoint tokenizer `apply_chat_template(messages, tools=tools)` |
| Serving parser implementation | Active Qwen3.5 checkpoint + active SGLang version configuration |
| ToolRL reward | `drug_agent.toolrl.molclaw_reward` over native parsed decisions |
| Real tool execution | Data-Pipe collector and explicit online rollout/debug only |

Tool-KG keeps its existing schema/adjudication/sampling authorities. Benchmark labels and evaluator metrics remain
audit data and never enter training prompts.

## Data flow

```text
clean Claude execution cwd
        ↓
immutable raw stream-json + outer rollout metadata
        ├──→ uncleaned Qwen3.5 native message projection (audit only)
        │
        └──→ Python deterministic filtering/path sanitation/observation compaction
                    ↓
             model-agnostic semantic canonical     ← permanent mother dataset
        ↓
mandatory LLM reasoning patch (failure is pending; mother data remains valid)
        ↓
        Qwen3.5 structured SFT view
                ↓
        tokenizer.apply_chat_template + qwen3_5 loss mask
                ↓
              Slime SFT
```

There is no XML ReAct stage or rendered-text reverse parser anywhere in the SFT path.

Every primary trajectory JSONL has a sibling `.pretty.json` JSON-array rendering for human review. Pretty files
are not data authorities and are never consumed by loaders.

## Collector boundary

Before Claude starts, its cwd contains only runtime material explicitly supplied by
`workdir-skills/molclaw-trajectory-execution/`. The task is passed through the normal user prompt.
`question.json`, `prompt.txt`, `run_meta.json`, `complete_session.jsonl`, attempt manifests and other collector
sidecars remain in the outer rollout directory. Agent-created scientific artifacts may appear in the cwd after
launch.

Each invocation has an immutable `attempts/attempt_NNNN/complete_session.jsonl`; the selected stream is copied
byte-for-byte to the sample-level `complete_session.jsonl`. Runner diagnostics never get appended to raw events.
Each stream has a `complete_session.pretty.json` sidecar for reading. Non-JSON Claude runtime diagnostics are
represented there as explicit line-numbered diagnostic records; the byte-identical JSONL remains authoritative.
`run_meta.json` and `selected_attempt_artifacts.json` both bind `question_sha256`, `user_prompt_sha256`,
`system_prompt_sha256`, `selected_session_sha256` and `source_dataset_sha256`; Python clean rejects a mismatch.
缺失任一绑定字段也会拒绝该样本；正式入口不保留 legacy partial-binding fallback。

## Semantic boundary

For stage-by-stage audit, raw events are also projected once into
`claude_raw_qwen35_native_projection_v1`. This projection preserves raw skill/runtime calls, paths and long
observations and marks every cleaning flag false. It is not the mother dataset; no downstream stage reverse-parses
it. Deployment-visible tool schemas are added only by the final Qwen adapter, because the raw Claude stream is not
their authority.

Semantic data contains only user task, assistant decisions, tool calls, tool observations, final response and
source provenance. Claude system/runtime instructions can be retained in audit metadata but are excluded from the
training body. Qwen system prompts and tool schemas are not semantic fields.

The canonicalizer groups all fragments with the same Claude `message.id`, preserves parallel calls as one
`tool_calls[]`, and matches results through `tool_use_id`. Removing a teacher-runtime interaction removes that
decision and its results; it never merges assistant decisions across an observation. Scientific calls are not
filtered against a student deployment manifest at this stage.

Python cleaning removes L2/L3/CLAUDE/collector-runtime actions. A read-only mixed Bash that contains both L1 and
teacher-runtime inspection is projected within the same assistant decision into native `Read` calls for its L1
`SKILL.md` targets; the original Bash-to-derived-call relation stays in audit. Attempt-workdir files become normal
cwd-relative paths, L1 paths become `skills/L1_tools/...`, and server paths from the user/MCP remain unchanged.
A real raw `Bash pwd` is preserved and maps its workdir chain to one trajectory-specific absolute root; Python
never invents a pwd call. Oversized observations retain scalar/path evidence reused downstream.

Mandatory LLM cleaning may only edit reasoning and prepend one task-level high-level plan to the first decision.
Calls, arguments, observations, final response, event order and provenance are immutable.

An LLM/provider/patch failure creates a pending clean materialization, not an invalid semantic sample. Only
successful patches enter the cleaned materialization; the semantic mother dataset is always retained.

## SFT boundary

The Qwen3.5 adapter injects the final training system prompt and a deployment-aligned tool set. `all` exposes the
full manifest; `trajectory-plus-distractors` exposes the six local discovery tools, all used MolClaw tools and a
deterministic equal-size distractor set. Tool selection and the deployment hash belong to the materialization
manifest, never to semantic. SFT stores
structured `messages` and `tools`, never rendered XML/text. Slime receives `--tool-key tools` and uses
`--loss-mask-type qwen3_5`, so system/user/tool observations are non-trainable and assistant
reasoning/action/final spans are trainable.

ToolRL is not materialized by the SFT pipeline. The existing ToolRL rows are derived directly from semantic assistant decisions. Their labels contain structured
`target_assistant`, `target_tool_calls` or `target_final_answer`; no custom string parser reconstructs them.
Rows remain decision-level records, but rollout sampling is trajectory-atomic: all ordinals for one `source_id`
must be present, contiguous and contained in one rollout batch. A batch may contain multiple complete
trajectories. Batch shuffling never shuffles individual decisions, and a trajectory is never split or padded with
duplicated decisions. Capacity rejection also applies to the whole trajectory. ToolRL launch requires
`global_batch_size == rollout_batch_size * n_samples_per_prompt`, so the complete rollout batch is not split across
optimizer updates.
The native reasoning/tool parser names are required launcher configuration, not schema fields. The launcher must
round-trip tokenizer-rendered targets through the installed SGLang parsers before starting Ray/GPU training.

Formal SFT remains offline. Real MCP calls are restricted to collection and explicit online evaluation or debug
paths. Legacy XML-dependent ToolRL/online runtime is explicitly deferred to the ToolRL-specific refactor and is
not linked from the SFT entrypoint.
