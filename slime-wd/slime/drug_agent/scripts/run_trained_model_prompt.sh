#!/usr/bin/env bash
set -euo pipefail

# Run this script on a GPU worker with the shared group-space mount.
# It loads a Slime torch-distributed checkpoint through the eval-only
# actor -> SGLang weight-sync path, then executes one fresh prompt with the
# same XML ReAct parser and real tool environment used by formal evaluation.

if [[ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

# Existing trained checkpoints (set MODEL_CHECKPOINT to the checkpoint root,
# not its iter_NNNNNNN child directory):
#
# 9B SFT:
# /root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/outputs/slime_drug_agent_runs/Qwen3.5-9B_current373_full_20260727_110712/sft
#
# 9B ToolRL (default below):
# /root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/outputs/slime_drug_agent_runs/Qwen3.5-9B_current373_full_20260727_110712/toolrl_procfsfix_20260728_091212
#
# There is currently no completed retained GAD checkpoint in the mainline.

MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-9B_current373_full_20260727_110712/toolrl_procfsfix_20260728_091212}
TASK_TYPE=${TASK_TYPE:-e2e}
TASK_ID=${TASK_ID:-trained_model_real_tools_smoke}
PROMPT_SUITE_FILE=${PROMPT_SUITE_FILE:-}

# Edit this question directly, or override it with QUESTION='...' bash ...
DEFAULT_QUESTION='Complete a real tool-use smoke test and ground every claim in actual observations.

1. Use Read to inspect skills/L1_tools/molclaw-fix-pdb/SKILL.md.
2. Call the MolClaw tool fix_pdb with input_path /data/lwj/wll/code/DrugAgentTools/sxy_sum/tests/fixtures/protein.pdb, remove_water=true, remove_heterogens=true, and add_hydrogens=false.
3. Use Write to create run_log.md in the task workspace, recording the returned canonical output artifact and a short factual status.
4. Use Read to read run_log.md back.
5. Return an e2e final_answer whose result reports the repaired-protein artifact and the verified local log artifact.

Do not claim that any step succeeded before receiving its observation. Do not skip the real tool calls.'
QUESTION=${QUESTION:-$DEFAULT_QUESTION}

RUN_NAME=${RUN_NAME:-prompt_toolrl9b_$(date +%Y%m%d_%H%M%S)}
DRUG_AGENT_EVAL_ROOT=${DRUG_AGENT_EVAL_ROOT:-${OUTPUTS_ROOT:-$WD/outputs}/slime_drug_agent_evals}
PROMPT_INPUT_ROOT=${PROMPT_INPUT_ROOT:-$DRUG_AGENT_RUNS_ROOT/manual_prompt_inputs}
PROMPT_FILE=${PROMPT_FILE:-$PROMPT_INPUT_ROOT/${RUN_NAME}.txt}
if [[ -z "$PROMPT_SUITE_FILE" ]]; then
  mkdir -p "$(dirname "$PROMPT_FILE")"
  printf '%s\n' "$QUESTION" > "$PROMPT_FILE"
  EVAL_MODE=single_prompt
else
  EVAL_MODE=prompt_suite
fi

# 9B topology used by the retained checkpoint. Override these only when the
# target worker/model profile has been deliberately validated.
export NUM_GPUS=${NUM_GPUS:-8}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
export MAX_WORKERS=${MAX_WORKERS:-1}
export MAX_STEPS=${MAX_STEPS:-0}
export TASK_TIMEOUT_SEC=${TASK_TIMEOUT_SEC:-10800}
export TEMPERATURE=${TEMPERATURE:-0.0}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
export MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-32768}
export DRUG_AGENT_ENABLE_LOCAL_TOOLS=1

export EVAL_MODE
export MODEL_CHECKPOINT TASK_TYPE TASK_ID RUN_NAME PROMPT_FILE PROMPT_SUITE_FILE DRUG_AGENT_EVAL_ROOT

echo "[single-prompt] model=$MODEL_CHECKPOINT"
echo "[single-prompt] input=${PROMPT_SUITE_FILE:-$PROMPT_FILE}"
echo "[single-prompt] run=$DRUG_AGENT_EVAL_ROOT/$RUN_NAME"
exec bash "$SLIME/drug_agent/scripts/run_molbench_eval.sh"
