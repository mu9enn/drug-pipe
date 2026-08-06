# Claude Code work scenes

| Scene | Workdir bundle | Invoked skill |
|---|---|---|
| `molclaw-tool-card-annotation` | `molclaw-tool-card-annotation/` | `/annotate-tool-card` |
| `molclaw-tool-edge-adjudication` | `molclaw-tool-edge-adjudication/` | `/judge-tool-edge` |
| `grounded-molclaw-task-generation` | `grounded-molclaw-task-generation/` | `/generate-toolchain-question` |
| `molclaw-trajectory-execution` | `molclaw-trajectory-execution/` | `/execute-molclaw-trajectory` plus selected L3/L2/L1 skills |
| `drug-trajectory-prose-curation` | `drug-trajectory-prose-curation/` | `/clean-drug-trajectory` |

The scene identifier and Workdir bundle directory name are deliberately identical. At runtime, only the selected bundle's `.claude` payload is copied into the isolated Claude Code workdir. The bundle's short `system_prompt.md` is passed with `--system-prompt`; a separate short user prompt carries the concrete task. Runtime JSON files are generated per invocation, while stable contracts and representative examples live under each skill's `references/` directory.
