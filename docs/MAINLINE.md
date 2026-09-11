> 2026-09-08：常规样本处理要求、阶段分工及验证发布见 [DATA_PIPE_REGULAR_CLEANING.md](DATA_PIPE_REGULAR_CLEANING.md)。训练和 rjob 继续暂停。

# Mainline

> 当前 SFT 发布和公平测评协议以 [EXPERIMENT_ALIGNMENT.md](EXPERIMENT_ALIGNMENT.md) 为准：512 条完整训练轨迹、默认 87 道测试题、保留全部 112 题做隔离。旧数据与旧结果属于历史实验。

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

ToolRL is not materialized by the SFT pipeline. V8 derives one structured row from each real semantic assistant
decision. A multi-call response remains one target with ordered `target_tool_calls`; no custom string parser
reconstructs it. The prompt contains only history before the current decision.

The default selector is now the three-stage [NeMo recent-history workflow](NEMO_SELECTOR.md):
prepare comparison text and native comparison conditions once; encode with Qwen3-Embedding-8B and run official
NeMo semantic deduplication at one explicit threshold; restore original records by ID and apply an independently
bound training-length audit. The old 0.6B short-description selector and policy-trial selector are separate
experiments, not prerequisites. Tool decisions can compare across task types; final answers remain task-specific.
Selection never rewrites history or labels. Unselected rows are similar recent-context examples, not proven
invalid actions or strict duplicates. There is no 20% quota.

Selected rows retain canonical `trajectory_index, decision_ordinal` order. Ordinal holes are allowed, trajectories
may cross rollout batch boundaries, and the reader wraps at dataset end to fill a requested batch (it does not
return a short tail batch or permanently delete the tail). The launcher must omit
`--rollout-shuffle`; each decision independently owns its four-response GRPO group. Tool content matching remains
order-insensitive one-to-one with multiplicity, with only a small content-gated pairwise teacher-order bonus.
Length limits are chosen from native Qwen template/tokenizer audits and never implemented by truncating or deleting
an otherwise valid teacher action.
The native reasoning/tool parser names are required launcher configuration, not schema fields. The launcher must
round-trip tokenizer-rendered targets through the installed SGLang parsers before starting Ray/GPU training.

Formal SFT remains offline. Real MCP calls are restricted to collection and explicit online evaluation or debug
paths. Legacy XML-dependent ToolRL/online runtime is explicitly deferred to the ToolRL-specific refactor and is
not linked from the SFT entrypoint.
