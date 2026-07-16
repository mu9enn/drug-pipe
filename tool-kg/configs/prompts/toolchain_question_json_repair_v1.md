The previous model output is not valid JSON or does not match the required schema.

Read `raw_output.txt` and `output_schema.json`. Convert the previous output into exactly one valid JSON object with this shape:

```json
{
  "status": "success",
  "public_question_text": "...",
  "question_payload": {
    "task": "...",
    "inputs": {},
    "expected_output": "..."
  },
  "rationale": "..."
}
```

Do not change the scientific meaning. Do not add tools, identifiers, scientific values, or unsupported facts. Return JSON only.
Do not write `output.json` or any file. Do not include tool IDs, tool/product names, a blueprint, or explicit tool order in the public question or payload. Do not use sequencing words such as first, then, next, finally, afterwards, or subsequently.
Do not add requests for the user to provide missing information. If the original task is not self-contained, return `reject`.
