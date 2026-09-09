# SFT data contract

> 当前 SFT 发布和公平测评协议以 [EXPERIMENT_ALIGNMENT.md](../docs/EXPERIMENT_ALIGNMENT.md) 为准：512 条完整训练轨迹、默认 87 道测试题、保留全部 112 题做隔离。旧数据与旧结果属于历史实验。

The cleaning pipeline produces one semantic trajectory and derives model-specific
SFT views from it. New Claude trajectories and historical migrations must obey
the same rules; do not generate one answer protocol and rewrite it later.

## Skill namespace

- `.agents/skills/<name>/SKILL.md` is the workspace layout, and the harness
  catalog is the authoritative skill namespace.
- The first use of a MolClaw tool is preceded by a structured `skill` call with
  the exact catalog name. A successfully loaded shared skill is not reloaded.
- Skill names must be canonical in structured calls and in all assistant targets:
  reasoning, final text, and tool arguments such as `Write` log content.
- Known old L1 aliases are rewritten to the current catalog name. An unavailable
  historical L2/L3 name used explicitly as a skill is converted to a plain
  legacy-note label so it cannot be learned as a callable name. Real paths,
  server identifiers, and artifact identifiers are preserved.
- An unknown structured `skill` call rejects the sample. If an unknown skill is
  encountered at inference time, the agent re-checks the current catalog and
  retries an exact listed name before invoking the scientific tool.

These rules are implemented in `pipeline.cleaning.skill_native_augmentation`,
which runs after reasoning cleanup and before SFT materialization.

## Final answers

- The teacher sees the same task-type-specific JSON contract retained in SFT.
- Terminal targets contain exactly the requested answer field plus `evidence`;
  `task_type` and redundant answer fields are not generated.
- The task-specific contract also validates exact public candidates, selection
  counts and complete rankings. Molecular canonicalization is confined to
  isolation audits and evidence-based historical repair.

These rules are implemented in `pipeline.output_contracts` and applied before
teacher generation for new data, with `release_aligned` providing audited historical repairs and isolation.
`output_contract_alignment` is retained only for older persisted envelope migrations;
it does not recover candidates and cannot replace the publication gates.
