# MolClaw L1 evaluation workspace template

This directory is the complete filesystem payload for one isolated evaluation
workspace. The evaluation runner copies its contents for every new inference
session and prefixes the benchmark question with `prompt_prefix.md`.

Layout:

```text
prompt_prefix.md
.agents/skills/
  <skill-name>/SKILL.md
```

`.agents/skills/` is the cross-harness skill location. Each L1 skill lives
directly beneath it without an extra wrapper or L1/L2/L3 parent level. The DSH
`molclaw-skill-eval` preset retains its native skill catalog and loader while
removing only the interactive `ask_user_question` tool. The model loads the
selected L1 instructions through `skill` before the first corresponding
MolClaw call.
