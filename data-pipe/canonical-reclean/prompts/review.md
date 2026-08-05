# Canonical ReAct prose review

Work only inside the current isolated directory. Read `request.json`, then
write exactly one `review.json`. Do not modify `request.json`.

Every entry in `segments` is an existing editable prose segment. Return one
review for every coordinate, in the same order. Use one of these actions:

- `keep`: the prose is already useful and non-redundant;
- `replace`: rewrite it and provide `replacement`;
- `delete`: it contains no unique scientific information and the surrounding
  assistant decision still contains an immutable tool call or final answer.

Rewrite or delete repeated paragraphs, repeated rankings, repeated tool
catalogs, repeated progress summaries, repeated conclusions, bloated preambles,
and historical cleaning corruption such as repeated “I cannot complete this
thought”. Merge near-duplicate reasoning while retaining the union of unique
scientific content.

Preserve concrete scientific motivation, target identities, parameters,
measurements, uncertainty, tool failure diagnosis, alternative hypotheses, and
replanning. A failed tool call followed by a useful adjustment is valuable and
must not be hidden. Do not shorten prose merely because it is detailed.

The task prompt, preceding observation excerpt, and immutable terminal-action
excerpt are read-only evidence. Each excerpt includes the SHA256 and original
length; a long context may contain an explicit omission marker. Never infer
anything from omitted content. Never invent facts or numbers. Never copy protocol tags,
absolute paths, tool-call JSON, observations, or final-answer JSON into a
replacement. Tool calls, observations, structured predictions, artifacts,
roles, and message order are outside your authority.

If `previous_findings` is non-empty, correct every listed quality problem.

Output shape:

```json
{
  "schema_version": "canonical_reclean_review_v1",
  "record_id": "...",
  "reviews": [
    {
      "message_index": 2,
      "segment_type": "thought",
      "segment_index": 0,
      "action": "replace",
      "replacement": "Concise evidence-grounded reasoning.",
      "rationale": "Removed repeated progress narration."
    }
  ]
}
```

For `keep` and `delete`, omit `replacement`. The file must contain JSON only.
