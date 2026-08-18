# ToolRL v6-final 开训前审计

## 结论

v6-final 已将 decision 语义从“一个 assistant message 不可拆”收缩为 ordered
reasoning/action segmentation。同一 multi-tool unit 在 SFT、ToolRL gold、reward parser 和
production runtime 中均使用单一 `<tool_call>` container 和空白分隔的 JSON objects。

## 1. Full / mol segmentation

| view | trajectories | assistant messages | expanded decisions | multi-action turns | interleaved turns | thoughtless actions | incomplete thought tails | tool / final | multi-tool decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 605 | 11,915 | 11,913 | 4 | 8 | 1,934 | 10 | 11,308 / 605 | 2,481 |
| mol | 365 | 5,329 | 5,329 | 0 | 0 | 428 | 0 | 4,964 / 365 | 1,527 |

`multi-action turns` 表示真正展开为多个可训 decision 的 turn。`interleaved turns`
还包含“action + thought-only tail”。full 的 10 个 incomplete thought 由 4 个尾段和
6 个独立 thought-only assistant message 组成；原文保留作 history，但不进入
SFT action loss 或 RL decision。

## 2. 原 8 个 interleaved turn 的实际切分

| source / assistant index | Decision 1 | Decision 2 / tail |
|---|---|---|
| `react_kg_00d86c196654705a:14` | thought + `calculate_pdb_structural_geometry`, `fpocket_toolkit` | state 带 Decision 1 精确 prefix；thought + `Read`, `calculate_pdb_structural_geometry`, `fpocket_toolkit` |
| `react_kg_b98c972b9210c604:18` | thoughtless `analyze_protein_ligand_interactions` | thought-only incomplete tail |
| `react_kg_b98c972b9210c604:24` | thought + `interaction_visualizer` | thought-only incomplete tail |
| `react_kg_b98c972b9210c604:26` | thoughtless `interaction_visualizer` | thought-only incomplete tail |
| `react_kg_414e16e2dc83b2cf:14` | thought + `calculate_mol_structure_complexity` | thought-only incomplete tail |
| `react_kg_de86fdffc2563cc8:104` | thoughtless 3× `molecule_docking_quickvina_fullprocess` | state 带 Decision 1 精确 prefix；thought + 1× 同工具 |
| `react_kg_de86fdffc2563cc8:132` | thoughtless 1× `molecule_docking_quickvina_fullprocess` | state 带 Decision 1 精确 prefix；thought + 1× 同工具 |
| `react_kg_de86fdffc2563cc8:150` | thoughtless 1× `molecule_docking_quickvina_fullprocess` | state 带 Decision 1 精确 prefix；thought + `Bash` |

这些 turn 不再因“没有 observation”丢弃。Decision 2 的 prompt 是原 conversation
history 加上 Decision 1 的精确 assistant prefix，没有伪造 user turn 或 observation。

## 3. SFT ↔ ToolRL serializer parity

SFT 和 ToolRL 共用 `toolrl_turn.serialize_decision`。multi-action message 在 base SFT
trajectory 中 loss-masked，每个 action 另生成 prefix-conditioned SFT record，并使用
`loss_char_start` 仅监督当前 action。full 因此是 605 条 canonical trajectories + 12
条 supplemental records = 617 条 SFT records；mol 仍为 365。

full 全量 parity 结果：11,913 个 SFT action targets 对 11,913 个
ToolRL gold actions，2,481 个 multi-tool actions，全部 byte-exact，mismatch=0。
mol 全量 parity 结果：5,329 对 5,329，其中 1,527 个 multi-tool
actions，同样全部 byte-exact，mismatch=0。

## 4. ToolRL output ↔ production runtime parser

production runtime 显式调用 `parse_runtime_decision(..., strict_toolrl_turn=True)`，与
reward parser 共用 strict grammar。end-to-end 结果：

| input | runtime | reward parser |
|---|---|---|
| `A\nB\nC` | valid，3 invocations | valid，3 invocations |
| `A,B,C` | invalid | invalid |
| `[A,B,C]` | invalid | invalid |
| 多个 container | invalid | invalid |

## 5. Runtime response limit quarantine

`max_response_length=16,384` 之上的 gold action 不可在当前 action space 中完整生成，
baseline 和 production 均禁止截断 label，排除原因为
`target_action_exceeds_runtime_response_limit`。

| source | raw decisions | over-limit gold actions | max gold-action tokens |
|---|---:|---:|---:|
| full | 11,913 | 46 | 445,297 |
| mol | 5,329 | 13 | 45,571 |

## 6. Accounting 定义

- `decision_count == grpo_group_count`：每个 prompt/decision 对应一个 n=4 group。
- `sampled_response_count = decision_count × 4`。
- `rollout_batch_count = decision_count / RBS`，当前 RBS=4。

因此 11,900 decisions 表示 11,900 groups、47,600 sampled responses 和 2,975 rollout
batches，而不是 2,975 groups。v6-final 的确定性物化结果是：

| source / view | decisions = GRPO groups | sampled responses | rollout batches | tool / final |
|---|---:|---:|---:|---:|
| full / official baseline | 11,864 | 47,456 | 2,966 | 11,263 / 601 |
| full / production | 4,804 | 19,216 | 1,201 | 4,202 / 602 |
| mol / official baseline | 5,316 | 21,264 | 1,329 | 4,952 / 364 |
| mol / production | 2,252 | 9,008 | 563 | 1,888 / 364 |

official baseline 只做官方语义对照与配置解析；当前正式训练使用
`drug_pipe_production`，不是 baseline reward/profile。
