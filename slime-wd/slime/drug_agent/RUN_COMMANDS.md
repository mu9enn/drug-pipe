# SFT

## Qwen3.5-4B SFT smoke on four GPUs

The current ReAct SFT data includes samples around 10k tokens. `MAX_TOKENS_PER_GPU` controls
dynamic microbatch packing but does not truncate one long sample. A 4B TP=1 run can therefore OOM
when SFT loss clones full-vocabulary logits.

Use the dedicated conservative wrapper. Its defaults are `TP=4`, `DP=1`, `RBS=1`, and `GBS=1`:

Do not enable PyTorch `expandable_segments` for this run. Colocated SGLang uses
`TorchMemorySaver`, which is incompatible with expandable allocator segments. The wrapper and
generic SFT entrypoint remove incompatible inherited values automatically.

```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd "$SLIME"

# Run this once only if $DATA/Qwen3.5-4B_torch_dist is absent.
bash drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh

bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_smoke.sh \
  2>&1 | tee "$DRUG_AGENT_RUNS_ROOT/qwen3_5_4b_sft_smoke_tp4.log"
```

The generic SFT entrypoint rejects invalid configurations before starting Ray. Its formal smoke
constraints are:

```text
RBS >= GBS
RBS % GBS == 0
NUM_GPUS % (TP * PP * CP * EP) == 0
GBS % DP == 0
DP = NUM_GPUS / (TP * PP * CP * EP)
```

After TP4 passes, TP2/DP2 is a less conservative throughput experiment:

```bash
TENSOR_MODEL_PARALLEL_SIZE=2 \
ROLLOUT_BATCH_SIZE=2 \
GLOBAL_BATCH_SIZE=2 \
SAVE_DIR="$DRUG_AGENT_RUNS_ROOT/Qwen3.5-4B_drug_sft_smoke_tp2" \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_smoke.sh
```

Before running the Qwen3.5-4B SFT/ToolRL commands below, make sure the Megatron checkpoint exists:

```bash
bash drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh
```

结论是：

1. `0.8B`、`4B`、`27B` 这条 SFT 路径已经准备好了。
2. 我刚把 [drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh#L76) 参数化成了可复用的入口，现在只要切 `MODEL_ARGS_FILE`，同一个脚本就能跑 `qwen3.5-0.8B / 4B / 27B`。
3. `9B` 这档现在还不完整，因为仓库里没有 `scripts/models/qwen3.5-9B.sh`。我只看到了 `scripts/models/glm4-9B.sh`，那是另一条模型线，不是这套 Qwen3.5 drug_agent 训练配置。

我也顺手把说明补到了 [README.md](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/README.md#L249) 和 [runbook](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/SLIME_DRUG_AGENT_RUNBOOK_zh.md#L189)。

下面给你一套可直接用的命令。

**通用准备**
```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd $SLIME

export DRUG_AGENT_RUNS_ROOT=$DRUG_AGENT_RUNS_ROOT
export PROMPT_DATA=/root/slime_sxy/group-space/sunxiangyu/slime-wd/data/mcp_sft_all.train.jsonl
```

**0.8B Smoke**
```bash
export MODEL_ARGS_FILE=scripts/models/qwen3.5-0.8B.sh
export HF_CHECKPOINT=$DATA/Qwen3.5-0.8B
export REF_LOAD=$DATA/Qwen3.5-0.8B_torch_dist
export SAVE_DIR=$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_sft_smoke

NUM_GPUS=2 \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=4 \
NUM_EPOCH=1 \
MAX_TOKENS_PER_GPU=4096 \
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
```

**0.8B Full**
```bash
export MODEL_ARGS_FILE=scripts/models/qwen3.5-0.8B.sh
export HF_CHECKPOINT=$DATA/Qwen3.5-0.8B
export REF_LOAD=$DATA/Qwen3.5-0.8B_torch_dist
export SAVE_DIR=$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_sft_full

NUM_GPUS=2 \
NUM_ROLLOUT=2 \
ROLLOUT_BATCH_SIZE=8 \
GLOBAL_BATCH_SIZE=8 \
NUM_EPOCH=1 \
MAX_TOKENS_PER_GPU=8192 \
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
```

**4B Smoke**
```bash
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_smoke.sh \
  2>&1 | tee "$DRUG_AGENT_RUNS_ROOT/qwen3_5_4b_sft_smoke_tp4.log"
```

**4B Full**

After TP4 smoke passes, launch the one-epoch full run:

```bash
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
```

Defaults: `TP=4`, `DP=1`, `RBS=GBS=4`, one epoch, full activation recompute, cosine LR from
`1e-5` to `1e-6`, 10% warmup, and checkpoint every 10 rollout steps. A timestamped `SAVE_DIR`
is created under `$DRUG_AGENT_RUNS_ROOT`.

Resume an interrupted run with:

```bash
RESUME_DIR=/path/to/Qwen3.5-4B_drug_sft_full_TIMESTAMP \
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
```

**27B debug/teacher only**

`$DATA/Qwen3.5-27B` is retained for SGLang rollout debugging and teacher use. A
matching Megatron `Qwen3.5-27B_torch_dist` checkpoint is not available, so this
runbook does not advertise 27B actor-training commands.

**9B**
- 现在不能给你一条“真正的 Qwen3.5-9B”命令，因为仓库里没有 `scripts/models/qwen3.5-9B.sh`。
- 如果你指的是 `glm4-9B`，那是另一套模型配置，不能直接套到这条 Qwen3.5 drug_agent 路径。
- 如果你把 9B 的 checkpoint 和 spec 名字给我，我可以马上把这条也补成同样的统一入口。

# ToolRL

`drug_agent/toolrl/` 已经落地了，且不改 slime core / trainer。

**这次实际做了什么**
- 新增 step-level ToolRL 转换器
  - [convert_react_to_toolrl_steps.py](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/convert_react_to_toolrl_steps.py)
- 新增 tool-call 解析与非 MolClaw 过滤
  - [parse_tool_calls.py](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/parse_tool_calls.py)
- 新增配置驱动的归一化与匹配规则
  - [normalization.py](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/normalization.py)
  - [tool_schema_config.yaml](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/tool_schema_config.yaml)
- 新增 ToolRL reward
  - [molclaw_reward.py](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/molclaw_reward.py)
- 新增可直接执行的训练脚本
  - [run_toolrl_grpo.sh](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/scripts/run_toolrl_grpo.sh)
  - [run_toolrl_grpo_smoke.sh](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/scripts/run_toolrl_grpo_smoke.sh)
  - [run_toolrl_grpo_learn.sh](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/scripts/run_toolrl_grpo_learn.sh)
- 新增测试集和无 `pytest` runner
  - [run_toolrl_tests.py](/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/slime/drug_agent/toolrl/tests/run_toolrl_tests.py)

**我本地验证过的结果**
- `python -m py_compile ...` 通过
- `python drug_agent/toolrl/tests/run_toolrl_tests.py` 通过
- 真实 `mcp_sft_all` 转换结果：
  - `kept_rows = 2469`
  - `skipped_rows = 657`
  - 预览样本的 `target_tool_call_count` 已正确回到 `1`
- 没有跑 GPU smoke

**可直接执行的命令**

先做数据转换：

```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd $SLIME

export TOOLRL_DATA_ROOT=${TOOLRL_DATA_ROOT:-$DRUG_AGENT_DATA_ROOT/toolrl}
export DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$DRUG_AGENT_RUNS_ROOT/toolrl}
mkdir -p "$TOOLRL_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"

python drug_agent/toolrl/convert_react_to_toolrl_steps.py \
  --input /root/slime_sxy/group-space/sunxiangyu/slime-wd/data/mcp_sft_all \
  --output "$TOOLRL_DATA_ROOT/mcp_sft_all.toolrl_steps.jsonl" \
  --skipped-report "$TOOLRL_DATA_ROOT/mcp_sft_all.skipped.jsonl" \
  --report "$TOOLRL_DATA_ROOT/mcp_sft_all.report.json"
```

做本地自检：

```bash
python drug_agent/toolrl/tests/run_toolrl_tests.py
```

如果你那边环境里有 `pytest`，也可以用：

```bash
python -m pytest -q drug_agent/toolrl/tests
```

0.8B smoke：

```bash
PROMPT_DATA=$TOOLRL_DATA_ROOT/mcp_sft_all.toolrl_steps.jsonl \
MODEL_ARGS_FILE=scripts/models/qwen3.5-0.8B.sh \
HF_CHECKPOINT=$DATA/Qwen3.5-0.8B \
REF_LOAD=$DATA/Qwen3.5-0.8B_torch_dist \
SAVE_DIR=$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_toolrl_grpo_smoke \
NUM_GPUS=2 \
GLOBAL_BATCH_SIZE=4 \
ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=1 \
NUM_ROLLOUT=1 \
MAX_TOKENS_PER_GPU=4096 \
bash drug_agent/toolrl/scripts/run_toolrl_grpo_smoke.sh \
  2>&1 | tee $DRUG_AGENT_RUNS_ROOT/qwen3.5-0.8B_toolrl_grpo_smoke.log
```

0.8B learn：

```bash
PROMPT_DATA=$TOOLRL_DATA_ROOT/mcp_sft_all.toolrl_steps.jsonl \
MODEL_ARGS_FILE=scripts/models/qwen3.5-0.8B.sh \
HF_CHECKPOINT=$DATA/Qwen3.5-0.8B \
REF_LOAD=$DATA/Qwen3.5-0.8B_torch_dist \
SAVE_DIR=$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_toolrl_grpo_learn \
NUM_GPUS=2 \
GLOBAL_BATCH_SIZE=16 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES_PER_PROMPT=2 \
NUM_ROLLOUT=8 \
MAX_TOKENS_PER_GPU=8192 \
bash drug_agent/toolrl/scripts/run_toolrl_grpo_learn.sh \
  2>&1 | tee $DRUG_AGENT_RUNS_ROOT/qwen3.5-0.8B_toolrl_grpo_learn.log
```

4B smoke：

```bash
PROMPT_DATA=$TOOLRL_DATA_ROOT/mcp_sft_all.toolrl_steps.jsonl \
MODEL_ARGS_FILE=scripts/models/qwen3.5-4B.sh \
HF_CHECKPOINT=$DATA/Qwen3.5-4B \
REF_LOAD=$DATA/Qwen3.5-4B_torch_dist \
SAVE_DIR=$DRUG_AGENT_RUNS_ROOT/Qwen3.5-4B_toolrl_grpo_smoke \
NUM_GPUS=2 \
GLOBAL_BATCH_SIZE=4 \
ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=1 \
NUM_ROLLOUT=1 \
MAX_TOKENS_PER_GPU=4096 \
bash drug_agent/toolrl/scripts/run_toolrl_grpo_smoke.sh \
  2>&1 | tee $DRUG_AGENT_RUNS_ROOT/qwen3.5-4B_toolrl_grpo_smoke.log
```

4B learn：

```bash
PROMPT_DATA=$TOOLRL_DATA_ROOT/mcp_sft_all.toolrl_steps.jsonl \
MODEL_ARGS_FILE=scripts/models/qwen3.5-4B.sh \
HF_CHECKPOINT=$DATA/Qwen3.5-4B \
REF_LOAD=$DATA/Qwen3.5-4B_torch_dist \
SAVE_DIR=$DRUG_AGENT_RUNS_ROOT/Qwen3.5-4B_toolrl_grpo_learn \
NUM_GPUS=2 \
GLOBAL_BATCH_SIZE=16 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES_PER_PROMPT=2 \
NUM_ROLLOUT=8 \
MAX_TOKENS_PER_GPU=8192 \
bash drug_agent/toolrl/scripts/run_toolrl_grpo_learn.sh \
  2>&1 | tee $DRUG_AGENT_RUNS_ROOT/qwen3.5-4B_toolrl_grpo_learn.log
```

27B remains a debug/teacher-only model because there is no matching Megatron
checkpoint for actor training.

`9B` 我没有给你硬编命令，因为仓库里没有 `scripts/models/qwen3.5-9B.sh`，我不想给你一个不存在的入口。
