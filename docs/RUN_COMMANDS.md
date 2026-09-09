# Run Commands

> 当前 SFT 发布和公平测评协议以 [EXPERIMENT_ALIGNMENT.md](EXPERIMENT_ALIGNMENT.md) 为准：512 条完整训练轨迹、默认 87 道测试题、保留全部 112 题做隔离。旧数据与旧结果属于历史实验。

以下只列当前主线。把示例路径替换为本机实际目录；大规模 agent、MCP 与 GPU 命令应手动确认后运行。

所有下列 agent 主线命令会自动保留每次 invocation 的原始轨迹：

```text
<workdir>/attempts/attempt_NNNN/complete_session.jsonl
<workdir>/attempts/attempt_NNNN/complete_session.pretty.json
```

MCP-ready retry 会递增 `NNNN`，不会覆盖旧流；顶层
`<workdir>/complete_session.jsonl` 是最终采用 attempt 的字节级副本，供现有 parser
继续读取；同目录 `complete_session.pretty.json` 只用于人工阅读。raw 中的非 JSON Claude runtime
diagnostic 会在 pretty 文件中变成带原始行号的显式 diagnostic record。不要编辑 attempt 文件或向
其中追加 runner 诊断。DeepSeek Harness 保存的是它自己的 canonical durable session JSONL，
不是伪装成 Claude schema 的 stream；下游 parser 会在内存中归一化，原始文件保持不变。
该布局仅对新运行生效。

### DeepSeek Harness

五个 work-scene（tool-card、tool-edge、task-generation、trajectory-execution、prose-curation）
和 `react-step-context-summarization` 都支持 `claude|deepseek` 选择。DSH 直连 OpenAI-compatible
模型接口，不需要切换全局 `cc-switch` 状态：

```bash
export DSH_BIN=/absolute/path/to/dsh
export DSH_NODE_BIN=/absolute/path/to/node   # Node 22.19.x or >=24

bash pipeline/claude_agent/run_execute.sh \
  --harness deepseek --provider dsv4flash --run-dataset \
  --task kg --dataset-csv /path/to/tasks.csv
```

`--dsh-model` 默认是 `deepseek-v4-flash`。DSH MCP 临时 patch 权限为 `0600`，进程结束即删除；
API key 和 MCP header 不写入 metadata。若 assistant 文本泄漏 DSML pseudo-markup，本次 attempt
会以 `dsml_tool_call_leaked_as_text` 失败，避免把坏轨迹静默当成成功数据。凭据优先读取
`DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY`；未设置时按 `--provider` 从只读的
`~/.cc-switch/cc-switch.db` 加载，因此不需要改变当前 provider。

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
  --project-root "$PWD" --run-id run_x --max-workers 4 --harness deepseek \
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
`workdir-skills/molclaw-trajectory-execution/.claude` 到每个 task workspace，并通过 `--strict-mcp-config` 只注册 `molclaw-scp`。完整执行 system prompt 与用户任务分别传给 Claude CLI：

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

Tool-KG 的 Tool Card、edge adjudication 和 Stage 3 success-first question sampler 都使用根
CLI 的 `--max-workers`。Stage 3 支持 1–4 并发：主线程顺序规划 graph walk、grounding seed
和重复配额，各 Claude worker 使用独立 runtime/workdir，最后按 attempt 序号归并。不要对同一
run directory 同时启动多个 sampler。

Raw → pre-clean native audit projection → answer-contract validation/Python clean → mandatory reasoning clean → native skill augmentation → Qwen3.5 SFT：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/data-pipe

DEPLOYMENT_TOOLS=/path/to/the/exact/student-visible-tool-manifest.json
bash scripts/run_cleaning.sh \
  --results-root results/<run_dir> \
  --output-root results/cleaned \
  --harness deepseek \
  --deployment-tool-set "$DEPLOYMENT_TOOLS" \
  --tool-visibility all
```

`DEPLOYMENT_TOOLS` 必须等于未来 rollout/deployment 真正可提供给 student 的工具集合。可改用
`--tool-visibility trajectory-plus-distractors`：保留六个本地工具、全部已用 MolClaw 工具及等量的
deterministic distractors。Semantic 构建不读取这个 manifest，因此更换 visibility/manifest 只需重新
materialize SFT，不必回到 raw。

Claude runner 与 Python clean 共用 `pipeline/output_contracts.py` 这一份答案协议。runner 在调用
Claude 之前即把题目规范成最终训练版本，写入 `question.json`、`prompt.txt` 并实际发送给 teacher；
后续 clean 不再把 prompt A 改成 prompt B，只验证该 prompt 未漂移，并要求 teacher final 已严格遵循
对应的 `answer_smiles` / `selected_smiles` / `ranked_smiles` / `result` 加 `evidence` 两字段 JSON。
字段错误、额外 `task_type` 或缺失 evidence 的样本进入 `rejected.jsonl`，不会合成答案或伪造证据。

默认输出：

```text
results/qwen35_native_raw/
├── qwen35_native_raw.jsonl                 # 未清洗 Qwen message 投影，仅供审计
├── qwen35_native_raw.pretty.json
└── projection_audit.jsonl
results/semantic_work/
├── semantic_trajectories.jsonl             # 永久母数据
├── semantic_trajectories.pretty.json
├── python_audit.jsonl
└── rejected.jsonl
results/cleaned/
├── semantic_trajectories.jsonl             # 仅成功 immutable patch
├── semantic_trajectories.pretty.json
├── llm_pending.jsonl                       # provider/patch 失败，母数据仍有效
├── qwen35_sft.jsonl
├── qwen35_sft.pretty.json
└── materialization_manifest.json
results/l1_augmented/
├── semantic_trajectories.jsonl             # native skill 首次使用协议
├── semantic_trajectories.pretty.json
├── augmentation_audit.jsonl
└── augmentation_manifest.json
```

所有 `.jsonl` 仍是一行一个 record；同目录 `.pretty.json` 是等价的缩进 JSON array，只供人工阅读，
不作为任何 loader 或下游转换输入。`qwen35_native_raw` 保留未过滤调用、未匿名化路径和未压缩
observation，并明确不是 semantic mother dataset。

正式脚本没有 `--skip-llm-clean` 或 `--no-high-level-plan`。失败样本只进入 pending；不会把 Python-only
semantic 冒充完成清洗的 SFT 数据。

## Slime 数据验证与派生

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime
source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh

SFT=/path/to/data-pipe/results/cleaned/qwen35_sft.jsonl
MODEL=$DATA/Qwen3.5-9B

PYTHONPATH=. python drug_agent/data/validate_sft_messages.py \
  --input "$SFT" --model "$MODEL"

```

SFT launcher 直接读取 structured `messages/tools`，不再接受目录、单个 JSON 或旧 ReAct flatten
入口。ToolRL materialization/rollout 不属于本轮 SFT 主线，待 ToolRL 专项统一处理。

## Formal offline training

这些命令启动 GPU/Ray。launcher 默认读取 canonical 文件名；这里仍显式传入路径，便于正式 run 留下清楚的数据来源。
4-GPU worker 的 4B 实跑经验、当前 token 长度统计和 8×H200 的 27B 参数决策见
[`SLIME_TRAINING_SETTINGS.md`](SLIME_TRAINING_SETTINGS.md)。

当前 structured SFT 入口：

```bash
PROMPT_DATA="$SFT" \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
```

ToolRL v8 数据准备是独立主线，直接消费 semantic 与 Qwen adapter view，不经过旧 ReAct/XML：

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime

python -m drug_agent.scripts.materialize_toolrl_v8 \
  --semantic /path/to/semantic_trajectories.jsonl \
  --qwen-sft /path/to/qwen35_sft.jsonl \
  --output-root /path/to/v8_toolrl/01_all_decisions

python -m drug_agent.scripts.select_toolrl_v8 encode \
  --input /path/to/v8_toolrl/01_all_decisions/all_decisions.jsonl \
  --output-root /path/to/v8_toolrl/03_embedding_cache \
  --model Qwen/Qwen3-Embedding-0.6B

python -m drug_agent.scripts.select_toolrl_v8 calibrate \
  --input /path/to/v8_toolrl/01_all_decisions/all_decisions.jsonl \
  --embedding-root /path/to/v8_toolrl/03_embedding_cache \
  --output /path/to/v8_toolrl/threshold_calibration.json \
  --distance-thresholds 0.05 0.08 0.10 0.12 0.15
```

阈值必须查看实际 calibration 与最大组后再传给 `select`。正式输出保持 canonical trajectory 顺序，
launcher 不传 `--rollout-shuffle`，默认每个 decision 生成 4 个候选。

如果需要把 selected 数量约束到目标预算，必须同时满足“每个 homogeneous cluster 至少一个代表”。
先提高 `--distance-threshold`，直到 cluster 数不超过预算，再设置 `--budget`；不能只给一个小于
cluster 数的预算。严格预算可能把 final-answer supervision 压成每组一条，可用
`--min-final-records` 保持明确的 final 下限。例如 12,634 条取约 20%：

```bash
python -m drug_agent.scripts.select_toolrl_v8 select \
  --input /path/to/all_decisions.jsonl \
  --embedding-root /path/to/embedding_cache \
  --output-root /path/to/strict_20pct/selected \
  --distance-threshold 0.35 \
  --budget 2527 \
  --min-final-records 121
```

这里 threshold 决定同质边界，budget 决定最终总量，min-final-records 防止总预算破坏目标类型覆盖。
具体数值必须由当前数据的 threshold sweep 和类型分布重新计算，不是跨数据集常量。

可选模型试答 selector 与默认 embedding 路径分离。`select_toolrl_v8_trials prepare` 只导出两次试答
请求；当前 serving/parser/reward 环境需返回每次工具与参数内容正确度 `content_scores`。随后 `select`
按内容差距优先、固定种子少量补取并恢复 canonical 顺序。预筛回答不得复用为正式 RL rollout。

## Online MCP debug

以下命令会真实访问工具环境：

```bash
export DRUG_AGENT_ALLOW_TOOL_ENV=1
PYTHONPATH=. python drug_agent/tools_debug/debug_mcp_tools.py --env-file ../../data-pipe/.env --list-tools
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

若进程中断，使用原来的 `RUN_NAME` 和完全相同的 checkpoint、数据、工具、skills、模型拓扑及生成
参数显式续跑：

```bash
RUN_NAME=<原运行名> \
RESUME_EVAL=1 \
MODEL_CHECKPOINT=/path/to/slime/checkpoint_root \
MOLBENCH_ROOT=/path/to/molbench \
MAX_WORKERS=2 \
bash drug_agent/scripts/run_molbench_eval.sh
```

每题完成后会立即写入 `task_results/`，同时原子刷新 `partial_results.jsonl` 和 `progress.json`。
resume fingerprint 不一致会在占用 GPU 前失败；默认不带 `RESUME_EVAL=1` 时也不会覆盖已有 task
checkpoint。只有全量题目完成后才生成正式 `predictions.jsonl`、`traces.jsonl` 和官方 metrics。

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

评测 preflight 捕获的 catalog 若代表未来 student 的真实 visibility，应把它作为新的
`--deployment-tool-set` 从 semantic mother dataset 重新物化 structured views。不要调用旧 XML
migration/converter，也不要把未验证的工具 alias 写回数据。

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
  drug_agent.tests.test_structured_qwen_pipeline \
  drug_agent.tests.test_local_tools
```

旧 ToolRL/GAD regression suite 不属于本轮 structured SFT 验收；应在后续 ToolRL 专项按其当时的
native parser/reward contract 单独执行。
