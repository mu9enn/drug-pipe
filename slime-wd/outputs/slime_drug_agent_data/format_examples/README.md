# Drug-Agent training format examples

These files are small, versioned schema references. They are not a training
dataset and are the only files below `slime-wd/outputs/` allowed into Git.
The four main examples use indented JSON for human review. Production datasets
remain newline-delimited JSONL.

The source is `react_pf_303eea476077228e`, the shortest eligible record in the
current 373-record canonical dataset. Its final reasoning and structured final
answer are one assistant generation, and it produces both tool-call and
final-answer decision examples for ToolRL and GAD.

## Files

- `source/canonical_react.json`: one canonical Data-Pipe record after LLM clean.
- `sft/sft_messages.json`: one SFT record. It is intentionally the same
  canonical record because Slime SFT directly consumes `messages`; chat-template
  rendering, tokenization, and assistant loss masking happen in the loader.
- `toolrl/toolrl_steps.json`: two history-only ToolRL states: one tool-call
  decision and one final-answer decision.
- `toolrl/skipped.jsonl`: converter audit rows for decisions outside ToolRL's
  MolClaw/final decision set; empty for this example.
- `toolrl/report.json`: ToolRL conversion counts.
- `gad/gad_steps.json`: a JSON array containing two history-only GAD states,
  one tool-call decision and one final-answer decision.
- `gad/skipped.jsonl`: empty for this sample.
- `gad/report.json`: GAD conversion counts.

## Historical status

These ReAct examples are archived references for the deferred ToolRL/GAD refactor. They are not accepted by the
current SFT launcher. Current SFT data is produced by `pipeline.cleaning.materialize_sft` as structured
`messages + tools` JSONL/Parquet; there is no directory/JSON flatten step.

ToolRL uses
`drug_agent.toolrl.convert_react_to_toolrl_steps.convert_react_to_toolrl_steps`.
GAD uses:

```bash
cd slime-wd/slime
jq -c . \
  ../outputs/slime_drug_agent_data/format_examples/source/canonical_react.json \
  > /tmp/canonical_react.jsonl
PYTHONPATH=. python -m drug_agent.gad.data \
  --input /tmp/canonical_react.jsonl \
  --output /tmp/gad_steps.jsonl \
  --skipped-report /tmp/gad_skipped.jsonl \
  --report /tmp/gad_report.json
```

ToolRL and GAD both consume the historical shared history-only decision extractor. Their XML dependencies are
not part of the current SFT mainline and will be handled in the ToolRL-specific refactor.
