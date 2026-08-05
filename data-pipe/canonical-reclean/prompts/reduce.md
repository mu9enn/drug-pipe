# Oversized thought consolidation

Read `request.json` and write exactly one `review.json`. Consolidate the map
notes into one coherent replacement for the specified thought. Remove overlap,
repeated rankings, repeated progress reports, and cleaning corruption while
preserving the union of unique scientific reasoning, evidence, parameters,
failure diagnosis, and replanning.

The task prompt and bounded preceding-observation/terminal-action excerpts are
read-only evidence. Do not infer anything from an omission marker. Do not
invent facts or numbers. Do not include XML protocol
tags, tool-call JSON, observation JSON, final-answer JSON, or absolute paths.

Output the same `canonical_reclean_review_v1` shape used by the normal review,
with exactly one review, action `replace`, and a non-empty `replacement`.
Place `message_index`, `segment_type`, and `segment_index` directly in that
review object; do not wrap them in a nested `coordinate` object.
