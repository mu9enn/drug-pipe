# data-pipe -> verl Training Bundle v0.1

This document defines the portable export contract produced in `data-pipe` and consumed downstream by `verl_wd` with minimal changes.

## Goal

- Provide a stable training-data handoff from existing `data-pipe` artifacts.
- Keep `data-pipe` independent from `verl_wd` code and runtime paths.
- Export schema-validated JSONL first; parquet is optional best-effort.

## Bundle Layout

The exporter writes:

- `exports/mol_pipeline_to_verl_bundle_v0.1/`
- `exports/mol_pipeline_to_verl_bundle_v0.1.tar.gz`

Important files:

- `bundle_manifest.json`
- `sft/mcp_sft_{train,valid}.raw.jsonl`
- `sft/mcp_sft_{train,valid}.normalized_json_action.jsonl`
- `rl_prompts/mcp_rl_prompts_{train,valid}.verl_ready.jsonl`
- `sft/sft_validation_report.{json,md}`
- `rl_prompts/rl_prompt_validation_report.{json,md}`
- `reports/export_summary.{json,md}`
- `reports/bundle_validation_report.json`
- `schemas/*.md`

## SFT Contract

- `messages` roles are restricted to `system|user|assistant`.
- Assistant turns are strict JSON strings and must parse with `json.loads`.
- Assistant action type is one of:
  - `tool_call`
  - `final_answer`
- Observation turns are represented as `role=user` with `<observation>...</observation>` wrappers.

## RL Prompt Contract

Each row includes:

- `data_source`
- `prompt` (`list[dict]`, not a plain string)
- `ability`
- `reward_model`
- `extra_info`
- `env_kwargs`

`env_kwargs.task` includes task payload only (`task_id/task_type/instruction/inputs/allowed_tools/max_steps/data_source`) and excludes endpoint/token secrets.

## KG Handling

- Default export includes `vs/ac/pf` from SFT outputs.
- If `results/kg_sampled` exists and `--include-kg-if-present` is enabled (default), KG prompt records are appended into RL prompt exports.
- KG inclusion is best-effort and does not change existing `vs/ac/pf` execution logic.

## Security Policy

- Bundle must not include `.env` or `.mcp.json`.
- Bundle must not include explicit API tokens or authorization headers.
- Security scan result is recorded in `bundle_manifest.json` and `reports/bundle_validation_report.json`.

## Commands

Direct export:

```bash
python pipeline/postprocess/export_verl_training_bundle.py \
  --scan-output-root results/used_molclaw_accepted_hit_0518 \
  --sft-output-dir results/used_molclaw_accepted_hit_0518/sft_outputs \
  --results-root results \
  --output-dir exports/mol_pipeline_to_verl_bundle_v0.1 \
  --include-kg-if-present
```

Validate only:

```bash
python pipeline/postprocess/validate_verl_training_bundle.py \
  --bundle-dir exports/mol_pipeline_to_verl_bundle_v0.1
```

One-shot build + validate + tar:

```bash
`scripts/build_verl_bundle.sh` has been removed. Invoke `export_verl_training_bundle.py` and `validate_verl_training_bundle.py` explicitly when this optional legacy handoff is needed.
```

## Out of Scope

- Running `verl` trainers
- Modifying `verl_wd` code
- Connecting to online MolClaw environments
- Defining production RL reward
