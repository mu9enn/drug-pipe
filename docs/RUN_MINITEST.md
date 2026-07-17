# Mainline Mini-Test

本页把 `Tool-KG → grounded task → Data-Pipe execution/evaluation/curation → SFT/ToolRL/GAD → online MCP replay` 限制到最小可行规模。默认只处理 1 个 task、1 个 rollout、1 个训练 prompt；Tool-KG 关系需要两个端点，GRPO 需要同一 prompt 的 2 个采样，这是两处不能再降为 1 的例外。

命令会真实调用 Claude、MolClaw/MCP 和 GPU，不属于普通单元测试。建议逐节执行并人工检查产物，不要把整页一次性粘贴到共享训练节点。ToolRL/GAD launcher 会重启本机 Ray/SGLang，只应在独占节点运行。

## 0. 公共变量与前置检查

```bash
export DRUG_PIPE_ROOT=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe
export MINI_ROOT=${MINI_ROOT:-/tmp/drug_pipe_minitest}
mkdir -p "$MINI_ROOT"

if [ -f "$DRUG_PIPE_ROOT/tool-kg/.env" ]; then
  set -a
  source "$DRUG_PIPE_ROOT/tool-kg/.env"
  set +a
fi

test -x "$(command -v claude)"
test -n "${MOLCLAW_SCP_API_KEY:-}"
test -n "${MOLCLAW_SCP_SERVER_URL:-}"
```

Tool-KG 的 `.env` 也会自动加载。Stage 3 还要求已构建固定 Science-KB：

```bash
test -s "$DRUG_PIPE_ROOT/tool-kg/science_kb/processed/science_kb.sqlite"
test -s "$DRUG_PIPE_ROOT/tool-kg/science_kb/manifests/science_kb_manifest.json"
```

GPU 阶段要求 `Qwen3.5-4B`、`Qwen3.5-4B_torch_dist` 和 `Qwen3.5-0.8B` 已在 `$DATA` 下。若 4B torch-dist 尚未准备：

```bash
cd "$DRUG_PIPE_ROOT/slime-wd/slime"
bash drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh
```

## 1. Tool-KG：两个工具、一条关系、一个 grounded question

快照仍需列出完整 MCP schema；Claude tool-card 只处理两个工具，关系裁决只处理一条有向 pair。

```bash
cd "$DRUG_PIPE_ROOT/tool-kg"
export KG_RUN_ID=${KG_RUN_ID:-minitest_$(date +%Y%m%d_%H%M%S)}
export KG_RUN_DIR="$PWD/runs/$KG_RUN_ID"
export KG_TOOL_IDS="$MINI_ROOT/kg_tool_ids.txt"
export KG_PAIR_IDS="$MINI_ROOT/kg_pair_ids.txt"

printf '%s\n' \
  retrieve_protein_structure_by_gene_name \
  fix_pdb > "$KG_TOOL_IDS"

bash scripts/run_pipeline_stage1_toolcards.sh "$KG_RUN_ID" \
  --tool-ids-file "$KG_TOOL_IDS" \
  --max-workers 1

run_kg_cli() {
  PYTHONPATH=src python -m molclaw_kg.cli \
    --project-root "$PWD" \
    --run-id "$KG_RUN_ID" \
    --api-key "$MOLCLAW_SCP_API_KEY" \
    --max-workers 1 \
    "$@"
}

run_kg_cli candidates

printf '%s\n' \
  'pair::retrieve_protein_structure_by_gene_name__to__fix_pdb' > "$KG_PAIR_IDS"
grep -Fx -f "$KG_PAIR_IDS" <(
  python - "$KG_RUN_DIR/candidate_pairs.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            print(json.loads(line)["pair_id"])
PY
)

run_kg_cli adjudicate --pair-ids-file "$KG_PAIR_IDS"
run_kg_cli canonical-edges
run_kg_cli views
run_kg_cli provenance
run_kg_cli export
run_kg_cli manifest

python - "$KG_RUN_DIR/graph_all.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
if not any(row.get("relation_status") == "valid" for row in rows):
    raise SystemExit("mini pair was not adjudicated valid; inspect the adjudication before continuing")
PY

bash scripts/run_sample_questions.sh "$KG_RUN_ID" \
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

人工检查：

```bash
python - "$KG_RUN_DIR/graph_all.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

python - "$KG_RUN_DIR/sample_results/sample_success_simple.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
print(json.dumps(row, ensure_ascii=False, indent=2)[:10000])
PY
```

若这条关系被合理裁决为 `negative`/`uncertain`，这本身是有效 mini-test 结果，但同一个 run 无法采 grounded question。先检查 `pair_adjudications.jsonl`、`tool_cards.jsonl` 和 `graph_all.jsonl`；确认不是模型/格式故障后，可将下一节的 `KG_TASK_RUN_ID` 指向另一个已有有效边的完成 run。

## 2. Grounded task：导出一个、执行一个、curate 一个

```bash
export KG_TASK_RUN_ID=${KG_TASK_RUN_ID:-$KG_RUN_ID}

cd "$DRUG_PIPE_ROOT/data-pipe"
python pipeline/kg/scripts/build_kg_task_dataset.py \
  --kg-run-dir "$DRUG_PIPE_ROOT/tool-kg/runs/$KG_TASK_RUN_ID" \
  --output-dir "$MINI_ROOT/kg_tasks" \
  --max-samples 1 \
  --no-include-raw-sample

test "$(wc -l < "$MINI_ROOT/kg_tasks/kg_sampled_tasks.jsonl")" -eq 1

bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file "$MINI_ROOT/kg_tasks/kg_sampled_tasks.jsonl" \
  --n-cases 1 \
  --num-rollouts 1 \
  --parallel-rollouts 1 \
  --results-root "$MINI_ROOT/data_pipe_results" \
  --skip-provider-switch 1

bash scripts/run_postprocess.sh \
  --results-root "$MINI_ROOT/data_pipe_results" \
  --output-root "$MINI_ROOT/react"
```

curator 会调用统一 evaluator；`kg` 是可执行性/工具使用评测，不另跑 benchmark evaluator。人工检查原始 run、评测、接收与拒绝原因：

```bash
export DATA_RUN_DIR=$(
  find "$MINI_ROOT/data_pipe_results" -type f -name run_config.json -printf '%h\n' |
  sort |
  tail -n 1
)

test -n "$DATA_RUN_DIR"
python -m json.tool "$DATA_RUN_DIR/run_config.json"
python -m json.tool "$DATA_RUN_DIR/trajectories/dataset_summary.json"
python -m json.tool "$MINI_ROOT/react/curation_report.json"

cd "$DRUG_PIPE_ROOT/slime-wd/slime"
PYTHONPATH=. python drug_agent/data/sample_debug.py \
  --input "$MINI_ROOT/react/react_trajectories.jsonl" \
  --num 1
```

必须有一条 accepted canonical ReAct 才继续训练：

```bash
test "$(wc -l < "$MINI_ROOT/react/react_trajectories.jsonl")" -eq 1
```

若为 0，查看 `$MINI_ROOT/react/react_rejected.jsonl` 和 `$DATA_RUN_DIR/trajectories/react_rejected.jsonl`，修正真实执行问题后重跑；不要为了打通训练而绕过 curator。

## 3. Slime：用新默认路径派生三类训练数据

先设置隔离目录，再 source 环境；这样 launcher 会正式使用本次切换后的 canonical 默认名。

```bash
export DRUG_AGENT_DATA_ROOT="$MINI_ROOT/slime_data"
export DRUG_AGENT_RUNS_ROOT="$MINI_ROOT/slime_runs"
source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd "$SLIME"

mkdir -p "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"
cp "$MINI_ROOT/react/react_trajectories.jsonl" \
  "$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl"

PYTHONPATH=. python drug_agent/data/validate_sft_messages.py \
  --input "$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl" \
  --protocol react_json

PYTHONPATH=. python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl" \
  --output "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.toolrl_steps.jsonl" \
  --skipped-report "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.skipped.jsonl" \
  --report "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.report.json"

bash drug_agent/gad/scripts/prepare_gad_step_data.sh

test -s "$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl"
test -s "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.toolrl_steps.jsonl"
test -s "$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl"

python -m json.tool "$DRUG_AGENT_DATA_ROOT/toolrl/react_trajectories.report.json"
python -m json.tool "$DRUG_AGENT_DATA_ROOT/gad/gad_steps.report.json"
```

## 4. GPU：各训练方法只跑一个最小更新

### SFT

输入不再传 `PROMPT_DATA`，用于验证新默认路径。4B smoke 固定 TP=4，因此需要 4 GPU。

```bash
cd "$SLIME"
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=1 \
SAVE_INTERVAL=1 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/sft_minitest" \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_smoke.sh
```

检查 checkpoint 和训练日志：

```bash
find "$DRUG_AGENT_RUNS_ROOT/sft_minitest" -maxdepth 2 -type f | sort
```

### ToolRL

一个 prompt 的 GRPO 最小组仍需 2 个 response，因此 `N_SAMPLES_PER_PROMPT=2`、`GLOBAL_BATCH_SIZE=2`。

```bash
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
SAVE_INTERVAL=1 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/toolrl_minitest" \
bash drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_smoke.sh
```

```bash
find "$DRUG_AGENT_RUNS_ROOT/toolrl_minitest" -maxdepth 2 -type f | sort
```

### GAD Stage 2

先生成一个 prompt 的 student negative，再做一轮、batch 1 的 discriminator warmup。

```bash
PROMPT_DATA="$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl" \
GAD_NEGATIVE_CACHE="$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl" \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
bash drug_agent/gad/scripts/generate_stage2_negatives.sh

PAIRS="$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl" \
DISCRIMINATOR_OUTPUT_DIR="$DRUG_AGENT_RUNS_ROOT/gad_discriminator_minitest" \
DISCRIMINATOR_EPOCHS=1 \
DISCRIMINATOR_BATCH_SIZE=1 \
DISCRIMINATOR_SAVE_INTERVAL=1 \
bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh

test -s "$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.minitest.jsonl"
test -s "$DRUG_AGENT_RUNS_ROOT/gad_discriminator_minitest/latest/gad_state.pt"
```

### GAD Stage 3

终端 A 启动独立 discriminator：

```bash
cd "$SLIME"
DISCRIMINATOR_RESUME="$DRUG_AGENT_RUNS_ROOT/gad_discriminator_minitest/latest" \
DISCRIMINATOR_OUTPUT_DIR="$DRUG_AGENT_RUNS_ROOT/gad_discriminator_online_minitest" \
GAD_DISCRIMINATOR_HOST=127.0.0.1 \
GAD_DISCRIMINATOR_PORT=8100 \
DISCRIMINATOR_SAVE_INTERVAL=1 \
bash drug_agent/gad/scripts/serve_discriminator.sh
```

终端 B 等 `/health` 成功后跑一个 prompt、两个 response：

```bash
curl -fsS http://127.0.0.1:8100/health | python -m json.tool

cd "$SLIME"
GAD_DISCRIMINATOR_URL=http://127.0.0.1:8100 \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
SAVE_INTERVAL=1 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/gad_stage3_minitest" \
bash drug_agent/gad/scripts/run_stage3_gad_grpo_smoke.sh
```

```bash
find "$DRUG_AGENT_RUNS_ROOT/gad_stage3_minitest" -maxdepth 2 -type f | sort
test -s "$DRUG_AGENT_RUNS_ROOT/gad_stage3_minitest/gad_trajectories.jsonl"
```

## 5. Online MCP：重放一个历史 tool call

这一步不启动训练模型，只把 canonical ReAct 中第一个 assistant tool action 真实重放到 MCP；默认输入即 `$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl`。

```bash
cd "$SLIME"
export DRUG_AGENT_ALLOW_TOOL_ENV=1
PYTHONPATH=. python drug_agent/tools_debug/debug_replay_trajectory.py \
  --index 0 \
  --max-tool-calls 1 \
  --run-name replay_minitest
```

```bash
find "$DRUG_AGENT_RUNS_ROOT/replay_minitest" -maxdepth 2 -type f -print
```

至此检查的是完整主线接口，而不是模型质量。人工验收时重点看：Tool-KG pair 是否有证据、grounded question 是否隐藏工具链、raw trace 是否真实执行、curator 拒绝原因是否合理、三类训练各自产生一次 checkpoint/trajectory，以及 online replay 的工具返回是否成功。
