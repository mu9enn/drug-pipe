# Run Commands

以下只列当前主线。把示例路径替换为本机实际目录；大规模 Claude、MCP 与 GPU 命令应手动确认后运行。

所有下列 Claude 主线命令会自动保留每次 invocation 的原始合并 stream：

```text
<workdir>/attempts/attempt_NNNN/complete_session.jsonl
```

MCP-ready retry 会递增 `NNNN`，不会覆盖旧流；顶层
`<workdir>/complete_session.jsonl` 是最终采用 attempt 的字节级副本，供现有 parser
继续读取。不要编辑 attempt 文件或向其中追加 runner 诊断。该布局仅对新运行生效。

## Tool-KG

完整构图会调用 MCP/Claude：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/tool-kg
bash scripts/run_full_pipeline.sh run_x --max-workers 1
```

已有 run 续跑：

```bash
bash scripts/run_full_pipeline.sh run_x --resume --max-workers 1
```

从 canonical graph 采 grounded questions：

```bash
PYTHONPATH=src python -m molclaw_kg.cli \
  --project-root "$PWD" --run-id run_x \
  sample-questions --sampling-profile simple_default \
  --target-successes 10 --max-attempts 40 --seed 42
```

未显式提供的参数来自 `configs/question_sampling.yaml`；需要旧 DAG 路径时必须使用
`--sampling-profile dag_legacy`。Stage3 只依赖 canonical
`graph.jsonl + edge_decisions.jsonl + tool_catalog.jsonl`。

正式 KG 结果位于 `runs/run_x/results/`：`tool_catalog.jsonl`、`edge_decisions.jsonl`、`graph.jsonl`、可选 `tasks.jsonl`、`run_manifest.json`，有问题时另有 `issues.jsonl`。

历史 KG 确定性迁移，不调用 Claude/MCP：

```bash
PYTHONPATH=src python -m molclaw_kg.cli migrate-kg \
  --source-dir /path/to/historical/kg-run \
  --output-dir /tmp/drug_pipe_convergence/migrated/kg
```

## Data-Pipe

真实执行：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/data-pipe
bash pipeline/claude_agent/run_execute.sh \
  --run-dataset --task vs --dataset-csv molbench/molbench-vs-30.csv
```

三段逻辑清洗与 canonical ReAct（前两段由同一个 Python 命令完成）：

```bash
PYTHONPATH=. python -m pipeline.cleaning.python_clean \
  --results-root results/<run_dir> \
  --output-root results/cleaning_work

PYTHONPATH=. python -m pipeline.cleaning.llm_clean \
  --input results/cleaning_work/python_drafts.jsonl \
  --python-audit results/cleaning_work/python_audit.jsonl \
  --output-root results/cleaned
```

默认 Python 结构化保留 MolClaw 和受支持的本地文件工具。仅需要 MolClaw 轨迹时，给
`python_clean` 或 `run_cleaning.sh` 显式增加 `--only-molclaw-tool`；该参数不改变
A/B/C gate 的 accepted/rejected 数量。

也可用 `bash scripts/run_cleaning.sh` 连续执行同样三段逻辑。LLM 只写
`llm_clean_patch.json`，Python 验证并应用白名单 prose edits；stdout 不作为数据接口。

输出训练接口为 `results/cleaned/react_trajectories.jsonl`；审计为同目录
`curation_audit.jsonl`，A/B/C gate 拒绝记录为 `rejected.jsonl`。LLM 失败时回退
Python draft，不产生 quarantine。

历史 trace 确定性迁移：

```bash
python pipeline/postprocess/migrate_trace.py legacy-sft \
  --source /path/to/legacy_sft.jsonl \
  --output-dir /tmp/drug_pipe_convergence/migrated/react

python pipeline/postprocess/migrate_trace.py raw-reference \
  --source-root /path/to/historical/results \
  --output-dir /tmp/drug_pipe_convergence/migrated/react_raw

# 可选：只保留 MolClaw call/observation
python pipeline/postprocess/migrate_trace.py raw-reference \
  --source-root /path/to/historical/results \
  --output-dir /tmp/drug_pipe_convergence/migrated/react_raw_molclaw_only \
  --only-molclaw-tool
```

`raw-reference` 只产生 `python_drafts.jsonl`，随后仍必须运行 `pipeline.cleaning.llm_clean`。

## Slime 数据派生

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime
source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh

REACT_SOURCE=/path/to/data-pipe/results/postprocess_candidates/react_trajectories.jsonl
OUT=$DRUG_AGENT_DATA_ROOT
mkdir -p "$OUT"
cp "$REACT_SOURCE" "$OUT/react_trajectories.jsonl"
REACT=$OUT/react_trajectories.jsonl

PYTHONPATH=. python drug_agent/data/validate_sft_messages.py \
  --input "$REACT" --protocol react_json

PYTHONPATH=. python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$REACT" \
  --output "$OUT/toolrl/react_trajectories.toolrl_steps.jsonl" \
  --skipped-report "$OUT/toolrl/react_trajectories.skipped.jsonl" \
  --report "$OUT/toolrl/react_trajectories.report.json"

PYTHONPATH=. python -m drug_agent.gad.data \
  --input "$REACT" \
  --output "$OUT/gad/gad_steps.jsonl" \
  --skipped-report "$OUT/gad/gad_steps.skipped.jsonl" \
  --report "$OUT/gad/gad_steps.report.json"
```

## Formal offline training

这些命令启动 GPU/Ray。launcher 默认读取 canonical 文件名；这里仍显式传入路径，便于正式 run 留下清楚的数据来源。

SFT 4B full：

```bash
PROMPT_DATA="$REACT" \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
```

ToolRL 4B full：

```bash
PROMPT_DATA="$OUT/toolrl/react_trajectories.toolrl_steps.jsonl" \
bash drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_full.sh
```

GAD Stage2/Stage3：

```bash
PROMPT_DATA="$OUT/gad/gad_steps.jsonl" \
bash drug_agent/gad/scripts/generate_stage2_negatives.sh

bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh

DISCRIMINATOR_RESUME=/path/to/discriminator/checkpoint \
bash drug_agent/gad/scripts/serve_discriminator.sh

PROMPT_DATA="$OUT/gad/gad_steps.jsonl" \
GAD_DISCRIMINATOR_URL=http://DISCRIMINATOR_HOST:8100 \
bash drug_agent/gad/scripts/run_stage3_gad_grpo_full.sh
```

Resume 使用各 launcher 已有的 `RESUME_DIR`、`TOOLRL_RESUME`、`STUDENT_RESUME`、`DISCRIMINATOR_RESUME` 变量；不要把普通初始化 checkpoint 当成 resume。

## Online MCP debug

以下命令会真实访问工具环境：

```bash
export DRUG_AGENT_ALLOW_TOOL_ENV=1
PYTHONPATH=. python drug_agent/tools_debug/debug_mcp_tools.py --list-tools
PYTHONPATH=. python drug_agent/tools_debug/debug_one_task.py --input-jsonl "$REACT" --index 0
PYTHONPATH=. python drug_agent/tools_debug/debug_replay_trajectory.py --input-jsonl "$REACT" --index 0
```

## 非侵入式检查

```bash
cd tool-kg
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v

cd ../data-pipe
PYTHONPATH=. python -m unittest discover -s pipeline/evaluate/tests -p 'test_*.py' -v
PYTHONPATH=. python -m unittest discover -s pipeline/cleaning/tests -p 'test_*.py' -v
PYTHONPATH=. python -m unittest discover -s pipeline/kg/tests -p 'test_*.py' -v

cd ../slime-wd/slime
PYTHONPATH=. python -m unittest -v \
  drug_agent.tests.test_decision_extractor \
  drug_agent.gad.tests.test_data \
  drug_agent.tests.test_offline_training
PYTHONPATH=. python drug_agent/toolrl/tests/run_toolrl_tests.py
PYTHONPATH=. python drug_agent/tools_debug/audit_offline_training.py
```
