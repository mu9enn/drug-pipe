# MolClaw-KG

MolClaw MCP 工具的 Tool Knowledge Graph 构建工程（fixed pruning taxonomy + directional agent adjudication）。

## 当前协议（已切换）

1. 固定 taxonomy 真源：`configs/stage_taxonomy.json`
2. stage 只用于 pruning，不参与 confidence 打分
3. pairwise 判边为**单向调用**（每个 allowed direction 一次调用）
4. validator 已从主流程移除
5. confidence 采用：`confidence_raw = agent_confidence`
6. 生成类边拆分为：
   - `generates_full_input_for`
   - `generates_partial_input_for`
7. `requires_intermediate` 不再作为 edge_type，仅作为 `negative_reason`
8. 主图字段瘦身；重解释信息写入 sidecar：
   - `tool_cards_debug.jsonl`
   - `edge_debug_sidecar.jsonl`
9. Stage3 默认模式为 `simple_toolchain_question`；旧 `dag_closure` 仅保留为显式调用的 legacy 模式

## 目录

- `configs/`
  - `stage_taxonomy.json`
  - `edge_ontology_v1.yaml`
  - `rules_v1.yaml`
  - `prompts/tool_card_agent_v1.md`
  - `prompts/pairwise_adjudication_v1.md`
  - `prompts/toolchain_question_sampler_v1.md`
- `skills_full/`（项目内置技能文档，保证项目独立运行）
- `src/molclaw_kg/`
- `scripts/`
  - `run_pipeline_stage1_toolcards.sh`
  - `run_pipeline_stage2_graph.sh`
  - `run_full_pipeline.sh`
  - `run_sample_questions.sh`

## 环境准备

```bash
git clone <YOUR_MOLCLAW_KG_REPOSITORY_URL> molclaw-kg
cd molclaw-kg
python3 -m pip install -r requirements.txt
cp .env.example .env
```

`.env` 至少包含：

- `MOLCLAW_SCP_API_KEY`
- `MOLCLAW_AGENT_PROVIDER`
- `MOLCLAW_AGENT_MCP_SERVER_NAME`
- `MOLCLAW_AGENT_MCP_SERVER_URL`
- `MOLCLAW_AGENT_MCP_AUTH_HEADER`
- `MOLCLAW_AGENT_MCP_AUTH_TOKEN`

## 运行

### 一键串行（Stage1 + Stage2，均带 alert-rerun）

```bash
bash scripts/run_full_pipeline.sh [run_id] --max-alert-rerun-rounds 3 --max-workers 1
```

继续已有 run：

```bash
bash scripts/run_full_pipeline.sh run_x --resume --max-alert-rerun-rounds 3 --max-workers 3
```

### 分阶段

Stage1（`snapshot -> doc-chunks -> tool-cards`）

```bash
bash scripts/run_pipeline_stage1_toolcards.sh run_x --alert-rerun --max-alert-rerun-rounds 3 --max-workers 3
```

继续已有 run：

```bash
bash scripts/run_pipeline_stage1_toolcards.sh run_x --resume --max-workers 3
```

Stage2（`candidates -> adjudicate -> score -> views -> provenance -> export -> audit -> eval-logs -> manifest`）

```bash
bash scripts/run_pipeline_stage2_graph.sh run_x --alert-rerun --max-alert-rerun-rounds 3 --max-workers 3
```

继续已有 run：

```bash
bash scripts/run_pipeline_stage2_graph.sh run_x --resume --max-workers 3 --alert-rerun --max-alert-rerun-rounds 3
```

## Stage3：KG 后处理采样

Stage3 将随机 anchor walk 仅作为科学任务启发。Agent 可以删除不必要工具，
或增加检索、转换和 provider 工具；Python 随后严格验证真实 Science-KB
grounding、ToolKG/skills 边证据、依赖闭包和问题可执行性，并确定性生成
`trajectory_v2_graph`。

首次使用前，单独下载一次官方 UniProt reviewed-human 与 GtoPdb 快照，再结合
机器上已有的 BindingDB 数据构建固定本地 Science-KB：

```bash
bash scripts/download_science_kb_sources.sh
python scripts/build_science_kb.py --replace
```

下载/构建脚本都不接入采样主流程。采样过程只读使用该 KB，不会自动下载、
更新或在缺失时回退到自由编造模式。

```bash
bash scripts/run_sample_questions.sh run_x \
  --sample-size 10 \
  --min-hops 2 \
  --max-hops 4 \
  --partial-policy closure_required \
  --edge-profile core_strict \
  --max-repair-rounds 2 \
  --sampling-mode dag_closure
```

调试线性模式（仅 debug）：

```bash
bash scripts/run_sample_questions.sh run_x --sample-size 5 --sampling-mode linear_debug
```

输出：

- `runs/<run_id>/sample_workdir/`
- `runs/<run_id>/sample_results/sample_attempts.jsonl`
- `runs/<run_id>/sample_results/sample_success.jsonl`
- `runs/<run_id>/sample_results/sample_success_v2.jsonl`
- `runs/<run_id>/sample_results/input_closure_report.jsonl`
- `runs/<run_id>/sample_results/grounding_records.jsonl`
- `runs/<run_id>/sample_results/skills_supported_edges.jsonl`
- `runs/<run_id>/sample_results/workflow_quality_report.json`
- `runs/<run_id>/sample_results/questions.csv`
- `runs/<run_id>/sample_results/sampling_meta.json`

## 关键产物

- `tool_snapshot.jsonl`
- `doc_chunks.jsonl`
- `tool_cards.jsonl`
- `tool_cards_debug.jsonl`
- `candidate_pairs.jsonl`
- `pair_pruned_by_stage.jsonl`
- `pair_adjudications.jsonl`
- `scored_edges.jsonl`
- `graph_all.jsonl` / `graph_core.jsonl` / `graph_expanded.jsonl` / `graph_uncertain.jsonl` / `graph_negative.jsonl`
- `edge_debug_sidecar.jsonl`
- `graph_all.csv`（edge-level）
- `<run_id>.csv`（pair-level）
- `graph_all.graphml`

## 说明

- Stage1/Stage2 都支持自动 alert-rerun（最多 3 轮，可参数化）。
- `--max-workers` 控制 stage1 tool-card 与 stage2 adjudicate 的并行度，默认 1。
- `--resume` 只恢复同一个 `run_id`，不会自动把新 run 当成旧 run。
- Stage2 alert-rerun 仅处理解析/格式失败样本，不重跑一般语义 `invalid/uncertain`。
- 历史 logs 仅用于评测，不用于主图真值构建。
- Stage3 默认只从 `core` 边采 anchor；partial 边必须有明确 mapping，且最终
  workflow 必须闭包后才能进入 success。
- Stage3 不执行远程 MolClaw 科学工具，只允许 Agent 查询只读本地 Science-KB。

### Stage3 simple question sampling mode

`simple_toolchain_question` 是面向高产出率的默认模式：

- 只从 `relation_status=valid` 的边采样 simple hidden toolchain；
- hidden toolchain 仅作为内部 blueprint，不会拼入公开问题；
- Agent 只需根据紧凑工具卡、边证据和少量 Science-KB facts 生成自然问题；
- Python 从完整 Science-KB seed 索引随机选择 target-ligand grounding，并限制同一
  target/compound 的优先重复次数；
- Agent 输出仅包含 `status/public_question_text/question_payload/rationale`；
- 程序将工具名或显式顺序暴露记录为 soft warning，不再因此丢弃可 rollout 问题；
- 明确要求用户后续补输入、使用 placeholder 或虚构路径的问题会被拒绝；
- JSON 格式错误时可执行 JSON-only repair；
- 主循环以目标成功问题数为终止条件，而不是固定尝试次数。

运行示例：

```bash
bash scripts/run_sample_questions.sh run_x \
  --target-successes 20 \
  --max-attempts 200 \
  --min-hops 2 \
  --max-hops 4 \
  --json-repair-rounds 1 \
  --grounding-selection random_seeded \
  --science-kb-topk 5 \
  --max-repeat-target 2 \
  --max-repeat-compound 2 \
  --seed 42
```

旧 closure 模式不再默认启用，需要时显式运行：

```bash
bash scripts/run_sample_questions.sh run_x \
  --sampling-mode dag_closure \
  --sample-size 20
```

输出：

- `runs/<run_id>/sample_results/sample_success_simple.jsonl`
- `runs/<run_id>/sample_results/sample_attempts_simple.jsonl`
- `runs/<run_id>/sample_results/questions_simple.csv`
- `runs/<run_id>/sample_results/simple_sampling_meta.json`
- `runs/<run_id>/sample_workdir/simple_toolchain_question/`
