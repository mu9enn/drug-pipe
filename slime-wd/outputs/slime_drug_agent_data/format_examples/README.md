# Drug-Agent training format examples

These files are small, versioned schema references. They are not a training
dataset and are the only files below `slime-wd/outputs/` allowed into Git.
The four main examples use indented JSON for human review. Production datasets
remain newline-delimited JSONL.

The source is `react_pf_303eea476077228e`, the shortest serialized record among
the 98 completed LLM-clean trajectories available from copied0721 when this
example was generated. It has six messages and produces non-empty views for
SFT, ToolRL, and GAD.

## Files

- `source/canonical_react.json`: one canonical Data-Pipe record after LLM clean.
- `sft/sft_messages.json`: one SFT record. It is intentionally the same
  canonical record because Slime SFT directly consumes `messages`; chat-template
  rendering, tokenization, and assistant loss masking happen in the loader.
- `toolrl/toolrl_steps.json`: one history-only ToolRL decision state whose
  target contains three MolClaw calls.
- `toolrl/skipped.jsonl`: the final-answer decision omitted by ToolRL because it
  has no MolClaw target call.
- `toolrl/report.json`: ToolRL conversion counts.
- `gad/gad_steps.json`: a JSON array containing two history-only GAD states,
  one tool-call decision and one final-answer decision.
- `gad/skipped.jsonl`: empty for this sample.
- `gad/report.json`: GAD conversion counts.

## Canonical transformations

SFT materialization:

```bash
cd slime-wd/slime
jq -c . \
  ../outputs/slime_drug_agent_data/format_examples/source/canonical_react.json \
  > /tmp/canonical_react.jsonl
PYTHONPATH=. python -m drug_agent.data.materialize_sft_jsonl \
  --input /tmp/canonical_react.jsonl \
  --output /tmp/sft_messages.jsonl
```

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

ToolRL and GAD both consume the shared history-only decision extractor. The
target assistant response and every future observation are excluded from the
state.
