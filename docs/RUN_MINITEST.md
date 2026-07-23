# Mainline Mini-Test：逐步运行与人工检查

本页用于逐步检查：

```text
Tool-KG
→ grounded task
→ Data-Pipe 真实执行
→ Python 筛选
→ Python 结构化
→ restricted-patch LLM clean
→ canonical ReAct
→ SFT / ToolRL / GAD 数据
→ 可选 GPU smoke
→ 可选 online MCP replay
```

默认只处理：

- 2 个 Tool-KG 工具；
- 1 个 directed pair；
- 1 个 grounded task；
- 1 个真实 rollout；
- 1 条 canonical ReAct。

GRPO 至少需要同一 prompt 的 2 个 responses，所以 ToolRL/GAD GPU smoke 中
`N_SAMPLES_PER_PROMPT=2` 是唯一的数量例外。

请在同一个终端中按编号执行。每一步都包含“运行”和“检查”；检查不符合预期时，
先停止并查看该步产物，不要继续执行后面的命令。

本文默认以下文件已经准备好：

```text
/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/tool-kg/.env
/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/data-pipe/.env
```

不要把整页一次性复制执行。Tool-KG、Data-Pipe execution、LLM clean 和 online replay
会真实调用 Claude/MCP；GPU smoke 会重启本机 Ray/SGLang，只能在独占 GPU 节点运行。

---

## A. 一次性初始化

### A1. 设置本次 mini-test 目录

运行：

```bash
export DRUG_PIPE_ROOT=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe
export MINI_ROOT="$DRUG_PIPE_ROOT/minitest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MINI_ROOT"
echo "$MINI_ROOT"
```

人工检查：

- 输出的是一个本次专用的新目录；
- 后续所有临时结果都写入该目录；
- 若需要继续之前的 run，应先手动把 `MINI_ROOT` 指向原目录。

### A2. 检查两个 `.env`

运行：

```bash
test -s "$DRUG_PIPE_ROOT/tool-kg/.env" && echo "tool-kg .env: OK"
test -s "$DRUG_PIPE_ROOT/data-pipe/.env" && echo "data-pipe .env: OK"
```

预期看到两行 `OK`。

这里只检查文件存在，不打印 secret。

### A3. 加载 Tool-KG 环境

运行：

```bash
cd "$DRUG_PIPE_ROOT/tool-kg"
set -a
source .env
set +a
```

检查必要入口：

```bash
command -v claude
test -n "${MOLCLAW_SCP_API_KEY:-}" && echo "MOLCLAW_SCP_API_KEY: SET"
test -n "${MOLCLAW_SCP_SERVER_URL:-}" && echo "MOLCLAW_SCP_SERVER_URL: SET"
```

如果 server URL 在 `.env` 中使用兼容变量 `MOLCLAW_SCP_MCP_URL`，则改用：

```bash
test -n "${MOLCLAW_SCP_SERVER_URL:-${MOLCLAW_SCP_MCP_URL:-}}" && echo "MCP URL: SET"
```

### A4. 检查 Stage3 Science-KB

运行：

```bash
test -s science_kb/processed/science_kb.sqlite && echo "Science-KB sqlite: OK"
test -s science_kb/manifests/science_kb_manifest.json && echo "Science-KB manifest: OK"
```

若任一文件缺失，先停止。不要进入 task sampling。

---

## B. Tool-KG：逐阶段检查

### B1. 创建 run ID 和两个工具的 allowlist

运行：

```bash
export KG_RUN_ID=${KG_RUN_ID:-minitest_$(date +%Y%m%d_%H%M%S)}
export KG_RUN_DIR="$DRUG_PIPE_ROOT/tool-kg/runs/$KG_RUN_ID"
export KG_TOOL_IDS="$MINI_ROOT/kg_tool_ids.txt"
export KG_PAIR_IDS="$MINI_ROOT/kg_pair_ids.txt"
```

写入两个工具：

```bash
printf '%s\n' retrieve_protein_structure_by_gene_name fix_pdb > "$KG_TOOL_IDS"
cat "$KG_TOOL_IDS"
```

定义后续每一步共用的 CLI：

```bash
kg() {
  PYTHONPATH=src python -m molclaw_kg.cli \
    --project-root "$PWD" \
    --run-id "$KG_RUN_ID" \
    --api-key "$MOLCLAW_SCP_API_KEY" \
    --max-workers 1 \
    "$@"
}
```

### B2. MCP snapshot

运行：

```bash
kg snapshot
```

检查：

```bash
test -s "$KG_RUN_DIR/tool_snapshot.jsonl"
wc -l "$KG_RUN_DIR/tool_snapshot.jsonl"
```

人工检查：

- snapshot 应包含完整 MCP inventory，不应只有 2 行；
- 若 MCP 连接、认证或 schema 获取失败，在这里停止。

确认两个目标工具存在：

```bash
grep -E '"tool_id": "(retrieve_protein_structure_by_gene_name|fix_pdb)"' \
  "$KG_RUN_DIR/tool_snapshot.jsonl"
```

预期匹配到两个工具。

### B3. 只构建两个 Tool Cards

运行：

```bash
kg tool-cards --tool-ids-file "$KG_TOOL_IDS"
```

检查数量：

```bash
wc -l "$KG_RUN_DIR/tool_cards.jsonl"
python -m json.tool "$KG_RUN_DIR/tool_cards_meta.json"
```

预期：

- `tool_cards.jsonl` 为 2 行；
- `agent_failure_count=0`；
- `alert_count=0` 或每个 alert 都能被合理解释。

逐条人工查看：

```bash
sed -n '1p' "$KG_RUN_DIR/tool_cards.jsonl" | python -m json.tool
sed -n '2p' "$KG_RUN_DIR/tool_cards.jsonl" | python -m json.tool
```

重点检查：

- `primary_stage`/`scheduling_stages` 来自 taxonomy；
- `schema_slots` 保留 MCP 的 name/raw type/required/default/enum；
- `slot_annotations` 没有伪造 schema slot；
- unknown slot 没有被静默删除；
- skill-derived slot 带有 evidence。

若有 alert：

```bash
cat "$KG_RUN_DIR/tool_card_alerts.jsonl"
cat "$KG_RUN_DIR/tool_card_rerun_targets.txt"
```

先处理 alert，再继续。

### B4. 生成 taxonomy-directed candidates

运行：

```bash
kg candidates
```

检查统计：

```bash
python -m json.tool "$KG_RUN_DIR/candidate_meta.json"
```

写入本次唯一目标 pair：

```bash
printf '%s\n' \
  'pair::retrieve_protein_structure_by_gene_name__to__fix_pdb' \
  > "$KG_PAIR_IDS"
```

确认 pair 被 taxonomy 调度：

```bash
grep -Ff "$KG_PAIR_IDS" "$KG_RUN_DIR/candidate_pairs.jsonl" | python -m json.tool
```

重点检查：

- `taxonomy_supporting_stage_pairs` 非空；
- candidate 中没有 semantic/name/format compatibility score；
- candidate 中没有 suggested edge type。

如果没有匹配，先停止。这表示当前 taxonomy 没有调度该方向，不能绕过 gate 强行 adjudicate。

### B5. 只 adjudicate 一个 directed pair

运行：

```bash
kg adjudicate --pair-ids-file "$KG_PAIR_IDS"
```

先检查 Claude 原始标准化结果：

```bash
grep -Ff "$KG_PAIR_IDS" "$KG_RUN_DIR/pair_adjudications.jsonl" | python -m json.tool
```

再检查 issue/alert：

```bash
test ! -s "$KG_RUN_DIR/pair_adjudication_alerts.jsonl" \
  && echo "pair adjudication alerts: none" \
  || cat "$KG_RUN_DIR/pair_adjudication_alerts.jsonl"
```

重点检查：

- `pair_id` 方向正确；
- relation status 与 edge type 的组合符合 ontology；
- valid edge 有明确 slot mapping/evidence；
- Python 没有用默认 edge type 修复 Claude 语义。

`negative` 或 `uncertain` 可能是合理科学结论，不等于代码失败。但这样的 run 无法用这一条边
继续采样；请先人工判断结论是否合理。

### B6. 生成 canonical edge decisions

运行：

```bash
kg canonical-edges
```

检查：

```bash
grep -Ff "$KG_PAIR_IDS" "$KG_RUN_DIR/canonical_edges.jsonl" | python -m json.tool
```

确认 canonical projection 没有改变 Claude 的：

- `relation_status`；
- `edge_types`；
- slot mapping；
- confidence；
- rationale/evidence。

### B7. 发布正式 Tool-KG 结果

运行：

```bash
kg finalize
```

检查正式目录：

```bash
find "$KG_RUN_DIR/results" -maxdepth 1 -type f -printf '%f\n' | sort
```

预期核心文件：

```text
edge_decisions.jsonl
graph.jsonl
run_manifest.json
tool_catalog.jsonl
```

仅在存在问题时才应出现 `issues.jsonl`。

查看唯一 decision：

```bash
grep -Ff "$KG_PAIR_IDS" "$KG_RUN_DIR/results/edge_decisions.jsonl" \
  | python -m json.tool
```

查看 graph：

```bash
cat "$KG_RUN_DIR/results/graph.jsonl"
```

继续条件：

- decision 为 `valid`；
- `eligible_for_sampling=true`；
- `graph.jsonl` 中存在该 pair。

如果 Claude 合理地给出 `negative/uncertain`，本次 Tool-KG mini-test 已经有效完成，但不要继续
用该 run 采 task。可以设置：

```bash
export KG_TASK_RUN_ID="replace_with_existing_valid_run_id"
```

否则：

```bash
export KG_TASK_RUN_ID="$KG_RUN_ID"
```

### B8. 只采 1 个 grounded task

运行：

```bash
bash scripts/run_sample_questions.sh "$KG_TASK_RUN_ID" \
  --sampling-profile simple_default \
  --target-successes 1 \
  --max-attempts 3 \
  --min-hops 1 \
  --max-hops 1 \
  --json-repair-rounds 0 \
  --science-kb-topk 1 \
  --max-repeat-target 1 \
  --max-repeat-compound 1 \
  --seed 42
```

设置实际采样 run 目录：

```bash
export KG_TASK_RUN_DIR="$DRUG_PIPE_ROOT/tool-kg/runs/$KG_TASK_RUN_ID"
```

检查 sampling meta：

```bash
python -m json.tool "$KG_TASK_RUN_DIR/intermediate/stage3/sampling_meta.json"
```

确认：

- `sampling_profile=simple_default`；
- `target_successes=1`；
- `success_count=1`；
- resolved config 与显式 override 一致；
- manifest 记录了 config/prompt hash。

查看公开 task：

```bash
test "$(wc -l < "$KG_TASK_RUN_DIR/results/tasks.jsonl")" -eq 1
head -n 1 "$KG_TASK_RUN_DIR/results/tasks.jsonl" | python -m json.tool
```

重点人工检查：

- question 没有泄露工具名、工具顺序或 hidden workflow；
- hidden trajectory 中每条边都引用 canonical `pair_id`；
- grounding 值来自 Science-KB；
- 没有依赖 debug sidecar 或 legacy graph view。

如果没有成功 task，检查：

```bash
cat "$KG_TASK_RUN_DIR/intermediate/stage3/sample_attempts.jsonl"
```

不要为了得到 1 条成功记录而跳过 validator。

---

## C. Data-Pipe：一个 task、一个 rollout、逐层清洗

### C1. 加载 Data-Pipe `.env`

运行：

```bash
cd "$DRUG_PIPE_ROOT/data-pipe"
set -a
source .env
set +a
```

检查：

```bash
command -v claude
```

Data-Pipe launcher 也会自动读取这个 `.env`。这里显式 source，是为了让后续手动命令使用同一环境。

### C2. 导出 1 个 Data-Pipe KG task

运行：

```bash
mkdir -p "$MINI_ROOT/kg_tasks"
python pipeline/kg/scripts/build_kg_task_dataset.py \
  --kg-run-dir "$KG_TASK_RUN_DIR" \
  --output-dir "$MINI_ROOT/kg_tasks" \
  --max-samples 1 \
  --no-include-raw-sample
```

检查：

```bash
export KG_TASK_FILE="$MINI_ROOT/kg_tasks/kg_sampled_tasks.jsonl"
test "$(wc -l < "$KG_TASK_FILE")" -eq 1
head -n 1 "$KG_TASK_FILE" | python -m json.tool
```

确认公开 question、task contract 和 hidden expected trajectory 的职责分离正确。

### C3. 真实执行 1 个 task、1 个 rollout

运行：

```bash
export DATA_RESULTS_ROOT="$MINI_ROOT/data_pipe_results"
bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file "$KG_TASK_FILE" \
  --n-cases 1 \
  --num-rollouts 1 \
  --parallel-rollouts 1 \
  --results-root "$DATA_RESULTS_ROOT" \
  --skip-provider-switch 1
```

找到本次 raw run：

```bash
export DATA_RUN_DIR=$(
  find "$DATA_RESULTS_ROOT" -type f -name run_config.json -printf '%h\n' \
  | sort | tail -n 1
)
echo "$DATA_RUN_DIR"
```

检查：

```bash
test -n "$DATA_RUN_DIR"
python -m json.tool "$DATA_RUN_DIR/run_config.json"
find "$DATA_RUN_DIR" -type f -name complete_session.jsonl -print
```

查看 execution summary：

```bash
test ! -f "$DATA_RUN_DIR/run_summary.jsonl" \
  || head -n 1 "$DATA_RUN_DIR/run_summary.jsonl" | python -m json.tool
```

人工检查：

- return code/timeout；
- 是否真实调用 MolClaw 工具；
- observation 是否来自真实执行；
- final 是否声称了工具未产生的结果。

如果执行失败，不要进入清洗；先检查该 run 下的日志和 `complete_session.jsonl`。

### C4. Step 1-2：Python 筛选与 Python 结构化

同一个命令先按 A/B/C gate 筛选 raw 样本，再只对通过样本执行 event pairing、ReAct
construction、artifact sanitization 和 observation compaction。Evaluator 与 invariants
只写 audit，不改变准入。输出状态只有 `python_valid/rejected`，不会提前产生 accepted。

运行：

```bash
export PYTHON_CLEAN="$MINI_ROOT/python_clean"
PYTHONPATH=. python -m pipeline.cleaning.python_clean \
  --results-root "$DATA_RUN_DIR" \
  --output-root "$PYTHON_CLEAN"
```

默认保留 MolClaw 与受支持的 `Read/Write/Edit/Bash/Grep/Glob/L1 Skill`。如需差异检查，
另选输出目录并增加 `--only-molclaw-tool`；两次 manifest 的
`python_valid_count/rejected_count` 应一致。

检查汇总：

```bash
python -m json.tool "$PYTHON_CLEAN/run_manifest.json"
wc -l \
  "$PYTHON_CLEAN/python_drafts.jsonl" \
  "$PYTHON_CLEAN/python_audit.jsonl" \
  "$PYTHON_CLEAN/rejected.jsonl"
```

查看 audit：

```bash
head -n 1 "$PYTHON_CLEAN/python_audit.jsonl" | python -m json.tool
```

人工确认：

- `execution_valid`、`task_answer_valid`、`training_trace_valid` 分开记录；
- evaluator metrics 只在 audit 中；
- `python_status=python_valid`，且不存在 `final_status`；
- raw path、return code、ground truth 没有进入 training messages；
- bare tool name、reasoning、observation、rich final 都被保留。

如果 Python clean 输出被拒绝，先查看：

```bash
head -n 1 "$PYTHON_CLEAN/rejected.jsonl" | python -m json.tool
```

不要把 `python_drafts.jsonl` 当作最终训练输入。

### C5. Step 3：restricted-patch LLM clean

只有在 C4 的 Python draft 和 audit 合理后再运行。Claude 在隔离 workdir 中只写
`llm_clean_patch.json`；Python 应用白名单 thought/final-summary edits，并验证 immutable
facts 和最终 schema。LLM timeout、坏 JSON、缺 patch 或 unsafe patch 都回退 Python draft，
不改变 A/B/C 准入结论。

运行：

```bash
export REACT_LLM_CLEAN="$MINI_ROOT/react_llm_clean"
PYTHONPATH=. python -m pipeline.cleaning.llm_clean \
  --input "$PYTHON_CLEAN/python_drafts.jsonl" \
  --python-audit "$PYTHON_CLEAN/python_audit.jsonl" \
  --output-root "$REACT_LLM_CLEAN" \
  --limit 1
```

检查：

```bash
python -m json.tool "$REACT_LLM_CLEAN/run_manifest.json"
head -n 1 "$REACT_LLM_CLEAN/curation_audit.jsonl" | python -m json.tool
```

比较 LLM clean 前后 canonical messages：

```bash
diff -u \
  <(head -n 1 "$PYTHON_CLEAN/python_drafts.jsonl") \
  <(head -n 1 "$REACT_LLM_CLEAN/react_trajectories.jsonl") \
  || true
```

默认模式允许：

- reasoning 语言更连贯；
- L1 skill 和真实受支持的本地文件操作；
- `run_log.md`、`result.md` 的真实写入；
- L2/L3 teacher-only 编排被清理；
- final 表达更清晰。

不允许：

- tool call 顺序变化；
- observation 数值变化；
- task prediction 变化；
- 新增 raw trace 中不存在的科学结论。

若 LLM 回退，检查 audit 中的 `llm_clean.status/findings` 以及对应 debug workdir：

```bash
find "$REACT_LLM_CLEAN/debug" -maxdepth 2 -type f -print
```

### C6. Final output fail-fast（不是清洗阶段）

最终训练输入固定为 final gate 发布的文件：

```bash
export REACT_SOURCE="$REACT_LLM_CLEAN/react_trajectories.jsonl"
test "$(wc -l < "$REACT_SOURCE")" -eq 1 && echo "canonical ReAct: 1 accepted"
```

如果文件为空就在这里停止。没有独立“选择器”，也不能回退到 Python draft 冒充 accepted。

---

## D. Slime：逐项派生训练数据

### D1. 设置隔离的数据与运行目录

先设置变量，再 source Slime 环境：

```bash
export DRUG_AGENT_DATA_ROOT="$MINI_ROOT/slime_data"
export DRUG_AGENT_RUNS_ROOT="$MINI_ROOT/slime_runs"
source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd "$SLIME"
```

准备 canonical 输入：

```bash
mkdir -p "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"
cp "$REACT_SOURCE" "$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl"
export REACT="$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl"
```

### D2. 检查 SFT canonical messages

运行：

```bash
PYTHONPATH=. python drug_agent/data/validate_sft_messages.py \
  --input "$REACT" \
  --protocol react_json
```

人工查看 1 条：

```bash
PYTHONPATH=. python drug_agent/data/sample_debug.py --input "$REACT" --num 1
```

### D3. 派生 ToolRL decision records

运行：

```bash
mkdir -p "$DRUG_AGENT_DATA_ROOT/toolrl"
PYTHONPATH=. python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$REACT" \
  --output "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.toolrl_steps.jsonl" \
  --skipped-report "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.skipped.jsonl" \
  --report "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.report.json"
```

检查：

```bash
python -m json.tool "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.report.json"
head -n 1 "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.toolrl_steps.jsonl" \
  | python -m json.tool
```

确认 state 只包含当前 assistant decision 之前的 history，不包含 target 或 future observation。

### D4. 派生 GAD decision records

运行：

```bash
bash drug_agent/gad/scripts/prepare_gad_step_data.sh
```

检查：

```bash
python -m json.tool "$DRUG_AGENT_DATA_ROOT/gad/gad_steps.report.json"
head -n 1 "$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl" | python -m json.tool
```

ToolRL 与 GAD 数量可以不同。只需确认二者使用同一个 history-only decision boundary。

---

## E. 可选 GPU smoke：每种方法单独运行

如果本次只检查数据主线，可以在 D4 结束。以下步骤必须在独占 GPU 节点逐项运行。

### E1. GPU 与模型前置检查

运行：

```bash
nvidia-smi -L
echo "DATA=$DATA"
```

检查所需模型：

```bash
find "$DATA" -maxdepth 2 -type d \
  \( -name '*Qwen3.5-4B*' -o -name '*Qwen3.5-0.8B*' \) -print
```

4B smoke 固定 TP=4，需要 4 张可用 GPU。

### E2. SFT：一个最小更新

运行：

```bash
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=1 \
SAVE_INTERVAL=1 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/sft_minitest" \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_smoke.sh
```

检查：

```bash
find "$DRUG_AGENT_RUNS_ROOT/sft_minitest" -maxdepth 2 -type f | sort
```

确认至少有训练日志和一次 checkpoint。

### E3. ToolRL：一个 prompt、两个 responses

运行：

```bash
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
SAVE_INTERVAL=1 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/toolrl_minitest" \
bash drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_smoke.sh
```

检查：

```bash
find "$DRUG_AGENT_RUNS_ROOT/toolrl_minitest" -maxdepth 2 -type f | sort
```

确认 reward/trajectory 日志存在，并且 formal launcher 没有调用 MCP executor。

### E4. GAD Stage2：只生成一个 negative

运行：

```bash
PROMPT_DATA="$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl" \
GAD_NEGATIVE_CACHE="$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl" \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
bash drug_agent/gad/scripts/generate_stage2_negatives.sh
```

检查：

```bash
test -s "$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl"
head -n 1 "$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl" \
  | python -m json.tool
```

### E5. GAD Stage2：只训练一轮 discriminator

运行：

```bash
PAIRS="$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl" \
DISCRIMINATOR_OUTPUT_DIR="$DRUG_AGENT_RUNS_ROOT/gad_discriminator_minitest" \
DISCRIMINATOR_EPOCHS=1 \
DISCRIMINATOR_BATCH_SIZE=1 \
DISCRIMINATOR_SAVE_INTERVAL=1 \
bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh
```

检查：

```bash
test -s "$DRUG_AGENT_RUNS_ROOT/gad_discriminator_minitest/latest/gad_state.pt"
```

### E6. GAD Stage3：终端 A 启动 discriminator

运行：

```bash
DISCRIMINATOR_RESUME="$DRUG_AGENT_RUNS_ROOT/gad_discriminator_minitest/latest" \
DISCRIMINATOR_OUTPUT_DIR="$DRUG_AGENT_RUNS_ROOT/gad_discriminator_online_minitest" \
GAD_DISCRIMINATOR_HOST=127.0.0.1 \
GAD_DISCRIMINATOR_PORT=8100 \
DISCRIMINATOR_SAVE_INTERVAL=1 \
bash drug_agent/gad/scripts/serve_discriminator.sh
```

保持终端 A 运行。

### E7. GAD Stage3：终端 B 检查 health

终端 B 需要重新执行 A1、D1 中的环境变量设置，然后运行：

```bash
curl -fsS http://127.0.0.1:8100/health | python -m json.tool
```

只有 health 成功后才继续。

### E8. GAD Stage3：终端 B 跑一个最小更新

运行：

```bash
GAD_DISCRIMINATOR_URL=http://127.0.0.1:8100 \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
SAVE_INTERVAL=1 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/gad_stage3_minitest" \
bash drug_agent/gad/scripts/run_stage3_gad_grpo_smoke.sh
```

检查：

```bash
find "$DRUG_AGENT_RUNS_ROOT/gad_stage3_minitest" -maxdepth 2 -type f | sort
test -s "$DRUG_AGENT_RUNS_ROOT/gad_stage3_minitest/gad_trajectories.jsonl"
```

---

## F. 可选 online MCP replay：只重放一个 tool call

此步骤不属于 formal training，会真实访问 MCP。

运行：

```bash
cd "$SLIME"
export DRUG_AGENT_ALLOW_TOOL_ENV=1
PYTHONPATH=. python drug_agent/tools_debug/debug_replay_trajectory.py \
  --input-jsonl "$REACT" \
  --index 0 \
  --max-tool-calls 1 \
  --run-name replay_minitest
```

检查：

```bash
find "$DRUG_AGENT_RUNS_ROOT/replay_minitest" -maxdepth 2 -type f -print
```

确认真实工具返回与原 observation 的语义一致；允许运行时间、artifact path 等运行时字段不同。

---

## G. 最终人工验收清单

完成到哪个阶段，就检查对应项目：

- [ ] Tool snapshot 是真实、完整的 MCP inventory。
- [ ] Tool Card 没有覆盖 MCP schema facts 或 taxonomy stage。
- [ ] Candidate 只由 taxonomy 调度。
- [ ] Claude adjudication 是 edge semantics 唯一 authority。
- [ ] `graph.jsonl` 是 `edge_decisions.jsonl` 的纯 projection。
- [ ] Grounded question 没有泄露 hidden toolchain。
- [ ] Data-Pipe raw trace 包含真实 tool call/observation。
- [ ] Python 筛选、Python 结构化、restricted LLM patch 与 A/B/C final projection 职责分离。
- [ ] LLM clean 没有改变 tool call、observation 数值或 prediction。
- [ ] Training JSONL 与 audit sidecar 分离。
- [ ] ToolRL/GAD state 不包含 future observation。
- [ ] Formal SFT/ToolRL/GAD training 没有访问 MCP。
- [ ] Online replay 只有显式 opt-in 后才能访问 MCP。

本 mini-test 验证的是主线接口、authority 边界和最小可运行性，不代表模型质量或全量数据 parity。
