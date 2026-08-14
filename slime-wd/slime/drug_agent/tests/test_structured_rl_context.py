from __future__ import annotations

import hashlib
import json

from drug_agent.scripts.compact_rl_context import compact_prompt_with_audit


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(str(message.get("content") or "") for message in messages)
        if kwargs.get("tokenize"):
            return self.encode(text)
        return text

    def encode(self, text, **kwargs):
        return str(text).split()


class _SizedTokens:
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __bool__(self):
        return self.length > 0

    def __getitem__(self, index):
        return 0


class _MillionTokenObservationTokenizer(_Tokenizer):
    marker = "MILLION_TOKEN_PAYLOAD_MARKER"

    def encode(self, text, **kwargs):
        value = str(text)
        if self.marker in value:
            return _SizedTokens(1_000_000)
        return _SizedTokens(max(1, len(value.split())))


def test_structured_compaction_preserves_prefix_recent_suffix_and_artifact():
    giant = "base64 " + "payload " * 5000
    prompt = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "original task"},
        {
            "role": "assistant",
            "content": '<thought>make artifact</thought><tool_call>{"tool_name":"Write","arguments":{"file_path":"artifact.pdb","content":'
            + json.dumps(giant)
            + "}}</tool_call>",
        },
        {
            "role": "user",
            "content": '<observation tool_name="Write">'
            + json.dumps({"ok": True, "status": "success", "content": {"output_file": "artifact.pdb", "blob": giant}})
            + "</observation>",
        },
        {
            "role": "assistant",
            "content": '<thought>inspect result</thought><tool_call>{"tool_name":"Read","arguments":{"file_path":"artifact.pdb"}}</tool_call>',
        },
        {"role": "user", "content": '<observation tool_name="Read">{"ok":true,"status":"success"}</observation>'},
    ]
    compacted, audit = compact_prompt_with_audit(_Tokenizer(), prompt, 260, summary_max_tokens=160)
    rendered = json.dumps(compacted, ensure_ascii=False)
    assert audit["compacted"] is True
    assert audit["output_tokens"] <= 260
    assert compacted[:2] == prompt[:2]
    assert compacted[-2:] == prompt[-2:]
    assert "artifact.pdb" in rendered
    assert hashlib.sha256(giant.encode()).hexdigest() in rendered
    assert giant not in rendered


def test_structured_compaction_does_not_include_current_target():
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"x"}}</tool_call>'},
        {"role": "user", "content": '<observation tool_name="Read">{"content":"' + "large " * 2000 + '"}</observation>'},
    ]
    compacted, audit = compact_prompt_with_audit(_Tokenizer(), prompt, 120, summary_max_tokens=60)
    assert audit["output_tokens"] <= 120
    assert "SECRET_CURRENT_GOLD" not in json.dumps(compacted)


def test_structured_compaction_handles_million_token_observation_without_materializing_tokens():
    tokenizer = _MillionTokenObservationTokenizer()
    payload = "x" * 400 + tokenizer.marker + "y" * 400
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"huge.json"}}</tool_call>'},
        {
            "role": "user",
            "content": '<observation tool_name="Read">'
            + json.dumps({"ok": True, "content": {"base64": payload}})
            + "</observation>",
        },
        {"role": "assistant", "content": '<tool_call>{"tool_name":"Write","arguments":{"file_path":"artifact.json","content":"ok"}}</tool_call>'},
        {"role": "user", "content": '<observation tool_name="Write">{"ok":true,"artifact":"artifact.json"}</observation>'},
    ]
    compacted, audit = compact_prompt_with_audit(tokenizer, prompt, 200, summary_max_tokens=120)
    rendered = json.dumps(compacted, ensure_ascii=False)
    assert audit["original_tokens"] == 1_000_000
    assert audit["output_tokens"] <= 200
    assert tokenizer.marker not in rendered
    assert hashlib.sha256(payload.encode()).hexdigest() in rendered
    assert "artifact.json" in rendered


def test_microcompact_clears_only_old_large_observations_and_preserves_pairing():
    giant = "payload " * 6000
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for index in range(6):
        prompt.extend(
            [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"artifact-'
                    + str(index)
                    + '.json"}}</tool_call>',
                },
                {
                    "role": "user",
                    "content": '<observation tool_name="Read">'
                    + json.dumps(
                        {
                            "ok": True,
                            "status": "success",
                            "content": {"output_path": f"artifact-{index}.json", "blob": giant if index < 2 else "small"},
                        }
                    )
                    + "</observation>",
                },
            ]
        )
    compacted, audit = compact_prompt_with_audit(_Tokenizer(), prompt, 1000, summary_max_tokens=200)
    rendered = json.dumps(compacted, ensure_ascii=False)
    assert audit["strategy"] == "microcompact_only"
    assert audit["removed_messages"] == 0
    assert audit["microcompact"]["compacted_observation_blocks"] == 2
    assert len(compacted) == len(prompt)
    assert rendered.count("<tool_call>") == 6
    assert rendered.count("<observation tool_name") == 6
    assert "artifact-0.json" in rendered and "artifact-5.json" in rendered
    assert hashlib.sha256(giant.encode()).hexdigest() in rendered
    assert giant not in rendered


def test_llm_summary_is_called_once_with_history_only_and_preserves_recent_suffix():
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for index in range(12):
        prompt.extend(
            [
                {"role": "assistant", "content": f'<thought>step {index}</thought><tool_call>{{"tool_name":"Read","arguments":{{"file_path":"{index}.json"}}}}</tool_call>'},
                {"role": "user", "content": '<observation tool_name="Read">' + json.dumps({"ok": True, "content": "value " * 100}) + "</observation>"},
            ]
        )
    calls = []

    def summarize(omitted):
        calls.append(omitted)
        assert all("SECRET_CURRENT_GOLD" not in json.dumps(message) for message in omitted)
        assert all("_source_message_index" in message for message in omitted)
        return {
            "schema_version": "react_context_summary_v1",
            "source_context_sha256": "a" * 64,
            "events": [],
            "unresolved_state": [],
        }, {"cache_hit": False, "attempts": 1}

    compacted, audit = compact_prompt_with_audit(
        _Tokenizer(), prompt, 420, summary_max_tokens=100, semantic_summarizer=summarize
    )
    assert len(calls) == 1
    assert audit["strategy"] == "microcompact_then_llm_summary"
    assert audit["output_tokens"] <= 420
    assert compacted[:2] == prompt[:2]
    assert compacted[-2:] == prompt[-2:]
