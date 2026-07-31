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

未显式提供的参数来自 `configs/question_sampling.yaml`。Stage3 只依赖 canonical
`graph.jsonl + edge_decisions.jsonl + tool_catalog.jsonl`，并只支持 `simple_default` profile。

正式 KG 结果位于 `runs/run_x/results/`：`tool_catalog.jsonl`、`edge_decisions.jsonl`、`graph.jsonl`、可选 `tasks.jsonl`、`run_manifest.json`，有问题时另有 `issues.jsonl`。

## Data-Pipe

真实执行：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/data-pipe
bash pipeline/claude_agent/run_execute.sh \
  --run-dataset --task vs --dataset-csv molbench/molbench-vs-30.csv
```

执行 canonical KG tasks 时，Launcher 从 `data-pipe/.env` 读取 endpoint/auth，复制仓库根目录的
`molclaw-skills` 到每个 task workspace，并通过 `--strict-mcp-config` 只注册 `molclaw-scp`：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/data-pipe

KG_RUN_DIR=/path/to/drug-pipe/tool-kg/runs/run_x
KG_INPUT_DIR="$PWD/pipeline/kg/data/run_x"
python pipeline/kg/scripts/build_kg_task_dataset.py \
  --kg-run-dir "$KG_RUN_DIR" \
  --output-dir "$KG_INPUT_DIR"
KG_TASK_FILE="$KG_INPUT_DIR/kg_sampled_tasks.jsonl"
KG_RESULTS_ROOT="$PWD/results/kg_run_x"

set -a
source .env
set +a

test -n "$MOLCLAW_SCP_MCP_URL"
test -n "$MOLCLAW_SCP_MCP_AUTH"
command -v claude

# 可选；MolClaw per-tool 硬超时，单位毫秒。默认 4 小时。
export MOLCLAW_MCP_TOOL_TIMEOUT_MS=14400000
# 更重的批次可显式改成 6 小时：21600000

PROVIDER="${CC_SWITCH_PROVIDER:-manual}" \
bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file "$KG_TASK_FILE" \
  --n-cases 20 \
  --num-rollouts 1 \
  --parallel-rollouts 1 \
  --max-workers 2 \
  --results-root "$KG_RESULTS_ROOT" \
  --skip-provider-switch 1
```

这一步只执行 raw rollout，不自动运行 Python clean 或 LLM clean。每次 Claude invocation 都在
sample workspace 的 `attempts/attempt_0001/complete_session.jsonl` 留存原始 stream-json，并复制
selected attempt 到顶层 `complete_session.jsonl`。

Launcher 会把 `MOLCLAW_MCP_TOOL_TIMEOUT_MS` 作为数值型 `timeout` 写入临时
`molclaw-scp` server 配置，并将有效值记录到 `run_config.json` 和每题 `run_meta.json`。
它不设置全局 `MCP_TOOL_TIMEOUT`，也不保存 MCP endpoint 的认证信息。默认值为
`14400000`（4 小时）；该值必须是至少 1000 的整数。客户端 timeout 后远端计算可能仍在
继续，因此不要把自动重试作为超时修复。

`--max-workers` 是 Data-Pipe 的全局 Claude invocation 上限：它同时覆盖不同 task row 和同一
task 的多个 rollout。`--parallel-rollouts` 暂时保留为兼容参数；未显式传
`--max-workers` 时才作为 worker 数使用。LLM clean 也接受同名参数，例如：

```bash
bash scripts/run_cleaning.sh --results-root <raw-run> --max-workers 2
```

Tool-KG 的 Tool Card 和 edge adjudication 已使用根 CLI 的 `--max-workers`。Stage 3 的
success-first question sampler 带有共享的去重/配额状态，目前仍保持串行；不能把根参数误认为
它已安全并行化。

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

也可用 `bash scripts/run_cleaning.sh` 连续执行同样三段逻辑。每条 Python-valid draft
都会由 LLM 自行检查 prose；LLM 只写 `llm_clean_patch.json`，Python 只允许修改已有
thought/final summary 并用 immutable/schema checks 保护执行事实；stdout 不作为数据接口。

输出训练接口为 `results/cleaned/react_trajectories.jsonl`；审计为同目录
`curation_audit.jsonl`，A/B/C gate 拒绝记录为 `rejected.jsonl`。LLM 失败时回退
Python draft，不产生 quarantine。

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
4-GPU worker 的 4B 实跑经验、当前 token 长度统计和 8×H200 的 27B 参数决策见
[`SLIME_TRAINING_SETTINGS.md`](SLIME_TRAINING_SETTINGS.md)。

在一台干净的 4-GPU worker 上，使用当前 373 条数据依次运行 SFT、ToolRL 和 pure GAD：

```bash
cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime
bash drug_agent/scripts/run_qwen3_5_4b_sft_toolrl_gad_serial.sh
```

总控脚本会重新确定性派生 3028 条 ToolRL steps 和 3234 条 GAD steps。算法权重关系是
`SFT -> ToolRL` 与 `SFT -> GAD` 两个分支；“串行运行”不表示 GAD 从 ToolRL checkpoint 初始化。
Pure GAD 阶段使用 3 张卡训练 TP1/DP3 generator，并保留第 4 张卡运行同源 4B discriminator。
任一阶段失败都会停止后续阶段，日志和 checkpoint 统一写入
`outputs/slime_drug_agent_runs/Qwen3.5-4B_current373_serial_<timestamp>/`。

SFT 4B full：

```bash
PROMPT_DATA="$REACT" \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
```

ToolRL 4B full：

```bash
PROMPT_DATA="$OUT/toolrl/react_trajectories.toolrl_steps.jsonl" \
TOOLRL_REWARD_MODE=official \
bash drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_full.sh
```

GAD Stage2/Stage3：

```bash
PROMPT_DATA="$OUT/gad/gad_steps.jsonl" \
bash drug_agent/gad/scripts/generate_stage2_negatives.sh

GENERATOR_WARMUP_LOAD=/path/to/completed/sft/checkpoint \
DISCRIMINATOR_MODEL_PATH="$DATA/Qwen3.5-4B" \
bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh

DISCRIMINATOR_RESUME=/path/to/gad_discriminator_warmup/latest \
bash drug_agent/gad/scripts/serve_discriminator.sh

PROMPT_DATA="$OUT/gad/gad_steps.jsonl" \
GAD_REWARD_MODE=pure \
GAD_DISCRIMINATOR_URL=http://DISCRIMINATOR_HOST:8100 \
STUDENT_WARMUP_LOAD=/path/to/completed/sft/checkpoint \
DISCRIMINATOR_WARMUP_LOAD=/path/to/gad_discriminator_warmup/latest \
GAD_WARMUP_MANIFEST=/path/to/gad_discriminator_warmup/warmup_manifest.json \
bash drug_agent/gad/scripts/run_stage3_gad_grpo_full.sh
```

`GAD_REWARD_MODE=rule` 不需要 discriminator service；`pure`（默认）和 `hybrid` 必须连接由
manifest 指定 warmup checkpoint 启动的 service。标准配置为同源 Qwen3.5-4B discriminator；
0.8B 只能通过显式 `DISCRIMINATOR_MODEL_PATH` 作为 efficiency variant 使用。

Resume 使用各 launcher 已有的 `RESUME_DIR`、`TOOLRL_RESUME`、`STUDENT_RESUME`、`DISCRIMINATOR_RESUME` 变量；不要把普通初始化 checkpoint 当成 resume。

## Online MCP debug

以下命令会真实访问工具环境：

```bash
export DRUG_AGENT_ALLOW_TOOL_ENV=1
PYTHONPATH=. python drug_agent/tools_debug/debug_mcp_tools.py --env-file ../../data-pipe/.env --list-tools
PYTHONPATH=. python drug_agent/tools_debug/debug_one_task.py --env-file ../../data-pipe/.env --input-jsonl "$REACT" --index 0
PYTHONPATH=. python drug_agent/tools_debug/debug_replay_trajectory.py --env-file ../../data-pipe/.env --input-jsonl "$REACT" --index 0
```

## Checkpoint → MolBench online evaluation

先在 worker 的 Python 环境安装一次在线评测依赖；不要在 formal training launcher 中安装：

```bash
cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime
python -m pip install -r drug_agent/requirements_online_eval.txt
```

指定任意 Slime torch-distributed checkpoint 执行 186 题 held-out 评测：

```bash
MODEL_CHECKPOINT=/path/to/slime/checkpoint_root \
MOLBENCH_ROOT=/root/slime_sxy/group-space/sunxiangyu/drug_wd/MolClaw/molbench \
MAX_WORKERS=2 \
MAX_STEPS=0 \
TASK_TIMEOUT_SEC=10800 \
TEMPERATURE=0.0 \
bash drug_agent/scripts/run_molbench_eval.sh
```

这也是脚本默认值：`MAX_STEPS=0` 表示不限制 assistant decision steps；
`TASK_TIMEOUT_SEC=10800` 以每题 3 小时总超时作为死循环和异常长任务的终止保护。

若 GPU worker 无外网，先在可访问集群 HTTP proxy 的 no-GPU 开发机启动已有纯字节 relay：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd
tmux new-session -d -s molclaw-relay-13208 \
  "MCP_RELAY_LISTEN_HOST=0.0.0.0 MCP_RELAY_LISTEN_PORT=13208 \
   bash molclaw-mcp-relay/run_relay.sh"
```

然后在 GPU worker 启动评测时增加 relay 地址；MCP URL 和 HTTPS 校验保持不变：

```bash
MOLCLAW_PROXY_URL=http://<no-gpu-ip>:13208 \
MODEL_CHECKPOINT=/path/to/slime/checkpoint_root \
bash drug_agent/scripts/run_molbench_eval.sh
```

评测脚本会把大小写 HTTP(S) proxy 变量传播到 Ray actor。`NO_PROXY` 不得包含 MolClaw
endpoint；relay 只用于 online debug/evaluation，不进入 SFT、ToolRL 或 GAD formal training。

checkpoint 根目录必须含 `latest_checkpointed_iteration.txt` 和对应 iteration 的 `common.pt`。
脚本会通过 Slime actor→SGLang 权重同步测指定 iteration，不会把 torch-dist 目录误当 HF
模型，也不会静默回退 base model。4B/9B 可按路径名推断 profile；其他模型必须显式提供
`HF_CHECKPOINT`、`MODEL_ARGS_FILE`、`NUM_GPUS`、TP 和 PP。完整科学评测不会被测试命令自动启动。

评测 preflight 捕获的实时 catalog 可用于未来训练数据迁移和派生数据再生成：

```bash
TOOL_CATALOG=/path/to/eval_run/tool_catalog.json \
INPUT="$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl" \
OUTPUT_ROOT="$DRUG_AGENT_DATA_ROOT/live_tool_catalog_v1" \
bash drug_agent/scripts/migrate_and_regenerate_live_tool_data.sh
```

迁移只影响未来数据，不修改已经训练完成的 checkpoint。迁移后必须先阅读
`migration_report.json`、`migration_rejected.jsonl` 和 `derived_data_manifest.json`，再决定是否冻结新训练集。

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
