"""Prepare lossless full-text inputs for the official NeMo semantic workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from drug_agent.toolrl.v8_dataset import sha256_file, stable_json


def encoding_text(row):
    # JSON preserves role/field boundaries and list order. Training-only masks
    # and audit metadata are not part of the embedding document.
    def message(value):
        return {key: val for key, val in value.items() if key != "step_loss_mask"}
    return stable_json({
        "history_before_current_decision": [message(m) for m in row["prompt"]],
        "available_tool_schemas": row["tools"],
        "current_teacher_response": message(row["label"]["target_assistant"]),
    })


def _init_tokenizer(path):
    global _tokenizer
    _tokenizer = Tokenizer.from_file(path)
    _tokenizer.no_truncation()
    _tokenizer.no_padding()


def _encode_row(line):
    row = json.loads(line)
    text = encoding_text(row)
    return row, text, len(_tokenizer.encode(text, add_special_tokens=True).ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    # Load the model's exact saved Rust tokenizer; no model-family conversion.
    seen = set()
    counts = Counter()
    scopes = Counter()
    previous_order = (-1, -1)
    text_digest = hashlib.sha256()
    def batches(source):
        with ProcessPoolExecutor(max_workers=6, initializer=_init_tokenizer,
                                 initargs=(str(Path(args.model) / "tokenizer.json"),)) as pool:
            while lines := list(islice(source, 64)):
                yield from pool.map(_encode_row, lines)
    with args.input.open() as source, (root / "encoding_audit.jsonl").open("w") as audit, (root / "eligible_full_text.jsonl").open("w") as eligible:
        for row, text, n in batches(source):
            ident = row["id"]
            assert ident and ident not in seen, ident
            seen.add(ident)
            metadata = row["metadata"]
            order = (metadata["trajectory_index"], metadata["decision_ordinal"])
            assert order > previous_order, ident
            previous_order = order
            disposition = "eligible" if n <= 32768 else "protected_over_embedding_limit"
            # Reuse the actual comparison boundary, never the old description
            # truncation guard or the old clustering/representative selector.
            scope = json.loads(metadata["comparison_scope"])
            scope.pop("lossy_content_guard", None)
            calls = row["label"].get("target_tool_calls", [])
            scope["skill_names"] = [c["arguments"].get("name", "") for c in calls if c["name"] == "skill"]
            scope = stable_json(scope)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            text_digest.update(stable_json([ident, text_hash]).encode())
            record = {"id": ident, "tokens": n, "text_sha256": text_hash,
                      "training_content_sha256": hashlib.sha256(stable_json({k:row[k] for k in ("prompt","tools","label")}).encode()).hexdigest(),
                      "disposition": disposition, "scope": scope}
            audit.write(json.dumps(record, ensure_ascii=False) + "\n")
            if disposition == "eligible":
                eligible.write(json.dumps({"id": ident, "text": text, "scope": scope, "tokens": n}, ensure_ascii=False) + "\n")
                scopes[scope] += 1
            counts[disposition] += 1
            if len(seen) % 500 == 0:
                audit.flush()
                print(len(seen), dict(counts), flush=True)
    manifest = {"input": str(args.input.resolve()), "input_sha256": sha256_file(args.input),
                "records": len(seen), "counts": dict(counts), "eligible_scope_sizes": dict(scopes),
                "encoding_model": args.model, "embedding_limit": 32768,
                "text_serialization": "full_fields_json_v1", "text_manifest_sha256": text_digest.hexdigest(),
                "truncation": False, "max_chars": None, "pretokenize": False,
                "tokenizer_files": {p.name:sha256_file(p) for p in Path(args.model).glob("*token*.json")}}
    (root / "preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
