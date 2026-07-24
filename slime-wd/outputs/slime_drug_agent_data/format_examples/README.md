# Drug-Agent training format examples

These files are small, versioned schema references. They are not a training
dataset and are the only files below `slime-wd/outputs/` allowed into Git.

The source is `react_pf_303eea476077228e`, the shortest serialized record among
the 98 completed LLM-clean trajectories available from copied0721 when this
example was generated. It has six messages and produces non-empty views for
SFT, ToolRL, and GAD.

## Files

- `source/canonical_react.jsonl`: canonical Data-Pipe output after LLM clean.
- `sft/sft_messages.jsonl`: the SFT view. It is intentionally the same
  canonical record because Slime SFT directly consumes `messages`; chat-template
  rendering, tokenization, and assistant loss masking happen in the loader.
- `toolrl/toolrl_steps.jsonl`: one history-only ToolRL decision state whose
  target contains three MolClaw calls.
- `toolrl/skipped.jsonl`: the final-answer decision omitted by ToolRL because it
  has no MolClaw target call.
- `toolrl/report.json`: ToolRL conversion counts.
- `gad/gad_steps.jsonl`: two history-only GAD states, one tool-call decision and
  one final-answer decision.
- `gad/skipped.jsonl`: empty for this sample.
- `gad/report.json`: GAD conversion counts.

## Canonical transformations

SFT materialization:

```bash
cd slime-wd/slime
PYTHONPATH=. python -m drug_agent.data.materialize_sft_jsonl \
  --input ../outputs/slime_drug_agent_data/format_examples/source/canonical_react.jsonl \
  --output ../outputs/slime_drug_agent_data/format_examples/sft/sft_messages.jsonl
```

ToolRL uses
`drug_agent.toolrl.convert_react_to_toolrl_steps.convert_react_to_toolrl_steps`.
GAD uses:

```bash
cd slime-wd/slime
PYTHONPATH=. python -m drug_agent.gad.data \
  --input ../outputs/slime_drug_agent_data/format_examples/source/canonical_react.jsonl \
  --output ../outputs/slime_drug_agent_data/format_examples/gad/gad_steps.jsonl \
  --skipped-report ../outputs/slime_drug_agent_data/format_examples/gad/skipped.jsonl \
  --report ../outputs/slime_drug_agent_data/format_examples/gad/report.json
```

ToolRL and GAD both consume the shared history-only decision extractor. The
target assistant response and every future observation are excluded from the
state.
