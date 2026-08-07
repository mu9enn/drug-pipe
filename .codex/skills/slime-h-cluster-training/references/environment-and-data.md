# Environment, data, and cluster operations

## Contents

- Project and worker paths
- Data contracts
- Safe worker workflow
- HF-to-torch_dist conversion
- Reproducibility records

## Project and worker paths

Use these as discovery anchors, not immutable facts:

| Context | Slime repository |
|---|---|
| Login host | `/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime` |
| Worker | `/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime` |

The `rjob` mount maps the login-host GPFS tree to `/root/slime_sxy/group-space/sunxiangyu` in the container. Source `slime_env.sh` before launch; the repository scripts support both `/root/...` and `/home/...` variants.

The original model locations were:

- Qwen3.5-27B: `.../drug-pipe/cached/archive_20260730_150855/models/slime-wd/data/Qwen3.5-27B`
- Qwen3.6-35B-A3B: the shared Hugging Face cache under `group-space/gpfs2-shared-public/huggingface/zskj-hub/` or its worker-visible `group-space/huggingface/` alias.
- Qwen3.5-122B-A10B-FP8: `.../group-space/sunxiangyu/slime_wd/data/Qwen3.5-122B-A10B-FP8`

Always verify `config.json`, `model.safetensors.index.json`, and all indexed shards. Paths and pod hostnames can change.

## Data contracts

The historical canonical v1 corpus used by the first 27B/122B experiments
contains:

| Method | Path relative to `outputs/slime_drug_agent_data/live_tool_catalog_v1` | Records |
|---|---|---:|
| SFT | `react_trajectories.jsonl` | 364 |
| ToolRL | `toolrl/toolrl_steps.jsonl` | 3182 |
| GAD | `gad/gad_steps.jsonl` | 3147 |
| Audit | `migration_audit.jsonl` | audit only |
| Rejected | `migration_rejected.jsonl` | never train |

The current canonical v2 corpus is the default for new launches:

| Method | Path relative to `outputs/slime_drug_agent_data/live_tool_catalog_v2` | Records | SHA-256 |
|---|---|---:|---|
| SFT | `react_trajectories.jsonl` | 365 | `deeddf45b5f8d38d4a0d8d09228501876948693329e8e8fba20b23b8e4eb4dbd` |
| ToolRL | `toolrl/toolrl_steps.jsonl` | 3187 | `21aec7f5729329131d239a9352b069f02dfc77acba2ca6623094804e63eb00a4` |
| GAD | `gad/gad_steps.jsonl` | 3152 | `f47a5b01fe43f8091feec89c7b6546ae49ddd239a08f8e5468e976dddfbcb0b8` |

V2 removed benchmark overlap and unsafe migrations, canonicalized missing
human `organism` to the server-authoritative default, removed the standalone
`Skill` tool, and deduplicated adjacent synonymous thoughts. Preserve its
migration, rejection, thought-dedup and derived-data manifests with every run.

The 122B LoRA production launcher used compacted v1
`toolrl_steps_ctx10240.jsonl` and `gad_steps_ctx10240.jsonl`; it is historical,
not evidence that v2 was trained. Resolve the actual launcher inputs, count and
hash them instead of silently mixing corpus versions.

With the Qwen3.5 tokenizer and `enable_thinking=false`, v2 is long despite its
small record count. Measured SFT rendered lengths were p50 14,132, p95 65,612,
maximum 94,016 and mean 23,873 tokens. ToolRL/GAD prompts had p50 about 9.4K,
p95 about 33K and maxima near 89.5K; prompt plus target reached about 94K.
Record batch size is not token batch size, and `max_tokens_per_gpu` does not
truncate a single oversized sample.

Do not encode the current v2 counts into topology or optimizer settings. The
9B serial launcher derives rollout/update counts after reading the final files.
For any non-divisible tail, use `materialize_batch_aligned_sft.py` for SFT and
`materialize_batch_aligned_jsonl.py` for ToolRL/GAD; append only the minimum
audited duplicates and preserve source/output hashes. Do not edit canonical
data or silently drop records.

## Safe worker workflow

Before launch:

```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime
MODEL_PROFILE=qwen35-27b-8xh200 source drug_agent/scripts/qwen3_large_profile.sh
EXPECTED_GPUS="$NUM_GPUS" HF_CHECKPOINT="$HF_CHECKPOINT" \
  bash drug_agent/scripts/preflight_large_model_worker.sh
bash drug_agent/scripts/validate_qwen3_large_profile.sh
```

Also inspect:

```bash
nvidia-smi
free -h
df -h "$GROUP_SPACE"
ray job list 2>/dev/null || true
tmux list-sessions 2>/dev/null || true
ps -eo pid,ppid,etime,stat,rss,args --sort=-rss | head -40
```

Do not launch into busy GPUs unless explicitly intended. `guard_ray_restart.sh` prevents this project's launchers from restarting Ray while another Ray job is active, but older/external scripts may not honor it.

Use a tmux window rather than a detached untracked shell. Put one serial driver in one window; record its name and Ray submission ID. Do not equate a live tmux shell with a live training child.

## HF-to-torch_dist conversion

Use the repository converter wrapper because it checks source shards, disk space, path separation, locking, and partial output:

```bash
MODEL_PROFILE=qwen36-35b-4xh200 source drug_agent/scripts/qwen3_large_profile.sh
NUM_GPUS="$NUM_GPUS" MODEL_ARGS_FILE="$MODEL_ARGS_FILE" \
HF_CHECKPOINT="$HF_CHECKPOINT" SAVE_DIR="$REF_LOAD" \
  bash drug_agent/scripts/prepare_qwen3_torch_dist.sh
```

Rules:

1. Write to an empty directory distinct from the HF source.
2. Never let two converters target the same output.
3. Verify the tracker contains `release` and at least one nonempty `.distcp` shard exists.
4. Load under the final topology; distributed checkpoints can reshard, but only a real load proves model-bridge compatibility.
5. Run a true optimizer step and a separate checkpoint-save gate.

The official 122B-FP8 bridge must recognize Qwen3.5 fused expert keys and dequantize/import them safely. Verify the local patches and `drug_agent/tests/test_large_model_profiles.py` before reconverting.

## Reproducibility records

Every run root should contain or point to:

- `resolved_config.env` or `serial_config.env`
- exact launcher command and environment overrides
- input dataset paths, counts, hashes if practical, and audit files
- HF/torch_dist/SFT/rollout checkpoint lineage
- repository commit plus dirty diff summary
- GPU model/count, host memory limit, image tag, library versions
- Ray job ID, tmux session/window, stage status markers
- stage logs, health snapshots, and checkpoint pointer files

Never promote a stage solely because a shell command returned zero. Require its semantic artifact: checkpoint tracker, complete negative cache, discriminator manifest, adapter file, or explicit gate `PASS` marker.
