from __future__ import annotations

import json
from pathlib import Path

from drug_agent.scripts.build_toolrl_length_probes import build_length_probes


def test_length_probes_are_unique_and_ordered(tmp_path: Path):
    source = tmp_path / "view.jsonl"
    rows = []
    for index, tokens in enumerate((1, 3, 5, 7, 9)):
        for copy_index in range(2):
            rows.append(
                {
                    "prompt": [],
                    "metadata": {
                        "source_id": f"id-{index}",
                        "assistant_index": index,
                        "decision_type": "tool_call",
                        "prompt_tokens_final": tokens,
                        "sampling_copy_index": copy_index,
                    },
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = build_length_probes(source, tmp_path / "probes", candidates_per_tier=4)
    assert manifest["unique_decisions"] == 5
    assert manifest["schema_version"] == "toolrl_length_probe_candidates_v2"
    assert manifest["probes"]["shortest"]["min_prompt_tokens"] == 1
    assert manifest["probes"]["shortest"]["max_prompt_tokens"] == 7
    assert manifest["probes"]["p50"]["min_prompt_tokens"] == 1
    assert manifest["probes"]["p50"]["max_prompt_tokens"] == 7
    assert manifest["probes"]["p95"]["min_prompt_tokens"] == 3
    assert manifest["probes"]["near_limit"]["max_prompt_tokens"] == 9
    assert all(sum(1 for _ in open(item["path"], encoding="utf-8")) == 4 for item in manifest["probes"].values())
