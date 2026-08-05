# Oversized thought evidence extraction

Read `request.json` and write exactly one `chunk_notes.json` file. This is one
chunk of an oversized historical thought. Extract the unique scientific
content needed to reconstruct useful reasoning. Remove repeated prose and
historical cleaning corruption, but retain concrete targets, parameters,
measurements, uncertainty, failure diagnosis, alternative hypotheses, and
replanning.

The accompanying observation and terminal-action contexts may be bounded
head/tail excerpts. Do not infer anything from an omission marker.

Do not invent facts. Do not reproduce long repeated tables when their unique
conclusion can be stated once. The output must be JSON only:

```json
{
  "schema_version": "canonical_reclean_chunk_notes_v1",
  "record_id": "...",
  "coordinate": {"message_index": 2, "segment_type": "thought", "segment_index": 0},
  "chunk_index": 0,
  "unique_content": "..."
}
```
