# Output example

```json
{
  "schema_version": "react_context_summary_v1",
  "source_context_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "events": [
    {
      "source_message_indices": [2, 3],
      "rationale": "Validate the recorded molecular input before downstream analysis.",
      "tool_calls": [
        {"source_message_index": 2, "tool_name": "is_valid_smiles", "arguments": {"smiles_list": ["CCO"]}}
      ],
      "observations": [
        {"source_message_index": 3, "tool_name": "is_valid_smiles", "status": "success", "artifacts": [], "paths": [], "ids": [], "error": null, "result_summary": "The recorded input was valid."}
      ]
    }
  ],
  "unresolved_state": []
}
```

Do not include a proposed next action, current gold response, or facts absent from the indexed source messages.
