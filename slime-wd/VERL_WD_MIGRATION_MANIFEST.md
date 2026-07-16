# verl_wd → slime-wd migration manifest

Migration completed on 2026-07-16. The result is `safe_to_delete: true`; the
source directory was deliberately not deleted.

## Migrated assets

- Qwen3.5 0.8B, 4B, and 27B Hugging Face models now live under `data/`.
- The 0.8B and 4B Megatron `torch_dist` checkpoints now live under `data/`.
- `pipelined_data` now lives under `data/`.
- The two 0.8B slime smoke checkpoints now live under `outputs/slime_runs/`.
- ToolRL step data now lives under `outputs/slime_drug_agent_data/toolrl/`.
- Six slime/DrugAgent logs now live under
  `outputs/slime_drug_agent_runs/historical_logs/`.
- The MolClaw 81-tool preflight is integrated at
  `slime/drug_agent/tools_debug/molclaw_preflight/`; its May 2026 reports are
  archived under `outputs/molclaw_preflight/history/20260525/`.

All large directory moves stayed on device 36 and retained their original
inode, confirming rename-based migration rather than duplicate copies. Primary
migrated model, data, and checkpoint assets total 98,015,227,890 bytes.

## Path and data result

- Active code uses `DATA_ROOT` in Python and `$DATA`, `$OUTPUTS_ROOT`,
  `$DRUG_AGENT_DATA_ROOT`, and `$DRUG_AGENT_RUNS_ROOT` in Shell.
- Active code, environment scripts, documentation, and symlinks contain no
  runtime dependency on `VERL_DATA` or `verl_wd`.
- SFT defaults to `data/mcp_sft_all.train.jsonl`; the legacy invalid action-JSON
  derivative is not referenced by a training entrypoint.
- GRPO data and the schema report were regenerated from the new paths.
- Qwen3.5-27B is debug/teacher-only because no corresponding `torch_dist`
  checkpoint exists.

## Verification

- Three HF models loaded config and tokenizer in offline mode.
- Both `torch_dist` checkpoints have markers, metadata, common state, and all
  expected non-empty shards (2 for 0.8B; 4 for 4B).
- Strict SFT validation: 516 valid, 0 invalid.
- ToolRL validation: 2469 valid, 0 invalid.
- GRPO: 125 rows, distributed as ac=43, pf=71, vs=11.
- DrugAgent, ToolRL, and GAD regressions passed; one FastAPI-only GAD test was
  skipped by its existing environment guard.
- All affected Shell scripts passed `bash -n`.
- MolClaw coverage, case generation, report merge, and report comparison passed
  offline.
- With `verl_wd` temporarily renamed and unavailable, all offline gates passed;
  the directory was then restored automatically.
- No process or open file was using `verl_wd` during the deletion audit.

## What remains in verl_wd

The remaining content is classified as verl-only code/data, three old Conda
environments, installation wheels, unrelated benchmark/model data, recreatable
caches, or obsolete duplicates. In particular, the remaining
`data/slime_drug_agent_data` does not contain the migrated ToolRL dataset.

The old tree still contains `verl-agent/molclaw_drug_agent/env_molclaw_mcp.sh`,
which held a plaintext MCP credential. No plaintext key was migrated, and the
preflight notebooks were redacted. Rotate that credential before deleting the
old directory.

The pre-existing broken `slime/.agents/skills` symlink is unrelated to this
migration and was intentionally left unchanged.
