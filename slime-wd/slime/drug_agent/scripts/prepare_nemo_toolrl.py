"""Prepare task/current-preserving recent-history copies for NeMo comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from drug_agent.toolrl.v8_dataset import sha256_file, stable_json
from drug_agent.toolrl.selector_window import window_document, VERSION
from drug_agent.toolrl.nemo_scope import comparison_scope, VERSION as SCOPE_VERSION
from drug_agent.toolrl.nemo_config import MODEL_PATH


def encoding_text(row):
    # JSON preserves role/field boundaries and list order. Training-only masks
    # and audit metadata are not part of the embedding document.
    def message(value):
        return {key: val for key, val in value.items() if key != "step_loss_mask"}
    by_name = {tool["function"]["name"]:tool for tool in row["tools"]}
    if len(by_name) != len(row["tools"]):
        raise ValueError("Duplicate tool names in catalog")
    current_definitions = [by_name[call["function"]["name"]]
                           for call in row["label"]["target_assistant"].get("tool_calls", [])]
    return stable_json({
        "history_before_current_decision": [message(m) for m in row["prompt"]],
        "current_call_tool_definitions": current_definitions,
        "current_teacher_response": message(row["label"]["target_assistant"]),
    })


def _init_tokenizer(path):
    global _tokenizer
    _tokenizer = Tokenizer.from_file(path)
    _tokenizer.no_truncation()
    _tokenizer.no_padding()


def _encode_row(line):
    row = json.loads(line)
    full_text = encoding_text(row)
    before_tokens = len(_tokenizer.encode(full_text, add_special_tokens=True).ids)
    text, n, window = window_document(json.loads(full_text),
        lambda value: len(_tokenizer.encode(value, add_special_tokens=True).ids))
    return row, text, n, window, before_tokens


def prompt_contains_catalog(row, serialized):
    """Check exact JSON and native <tools> blocks, not mere tool-name mentions."""
    decoder = json.JSONDecoder()
    for message in row["prompt"]:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if serialized in content:
            return True
        for block in re.findall(r"<tools>(.*?)</tools>", content, re.S):
            definitions = []
            remainder = block.strip()
            try:
                while remainder:
                    value, end = decoder.raw_decode(remainder)
                    definitions.extend(value if isinstance(value, list) else [value])
                    remainder = remainder[end:].strip()
            except ValueError:
                continue
            if stable_json(definitions) == serialized:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    if (root / "preparation_manifest.json").exists():
        raise ValueError("Preparation already complete; use a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    (root / "tool_catalogs").mkdir(exist_ok=True)
    (root / "system_messages").mkdir(exist_ok=True)
    # Load the model's exact saved Rust tokenizer; no model-family conversion.
    seen = set()
    counts = Counter()
    scopes = Counter()
    catalog_counts = Counter()
    catalog_cache = {}
    system_counts = Counter()
    token_lengths = {"before":[], "after":[]}
    position_counts = Counter()
    previous_order = (-1, -1)
    text_digest = hashlib.sha256()
    def batches(source):
        with ProcessPoolExecutor(max_workers=6, initializer=_init_tokenizer,
                                 initargs=(str(Path(args.model) / "tokenizer.json"),)) as pool:
            while lines := list(islice(source, 64)):
                yield from pool.map(_encode_row, lines)
    with args.input.open() as source, (root / "encoding_audit.jsonl").open("w") as audit, (root / "eligible_text.jsonl").open("w") as eligible:
        for row, text, n, window, before_tokens in batches(source):
            ident = row["id"]
            assert ident and ident not in seen, ident
            seen.add(ident)
            metadata = row["metadata"]
            system = stable_json([{k:v for k,v in m.items() if k != "step_loss_mask"}
                                  for m in row["prompt"] if m["role"] == "system"])
            system_version = hashlib.sha256(system.encode()).hexdigest()
            if system_version not in system_counts:
                if system_counts:
                    raise ValueError("Mixed system message versions: split the input into separate runs")
                (root / "system_messages" / f"{system_version}.json").write_text(system + "\n")
            system_counts[system_version] += 1
            catalog = stable_json(row["tools"])
            if catalog not in catalog_cache:
                if catalog_cache:
                    raise ValueError("Mixed full tool catalog versions: split the input into separate runs")
                version = hashlib.sha256(catalog.encode()).hexdigest()
                catalog_cache[catalog] = version
                (root / "tool_catalogs" / f"{version}.json").write_text(catalog + "\n")
            version = catalog_cache[catalog]
            catalog_counts[version] += 1
            if prompt_contains_catalog(row, catalog):
                counts["prompt_contains_complete_catalog"] += 1
            order = (metadata["trajectory_index"], metadata["decision_ordinal"])
            assert order > previous_order, ident
            previous_order = order
            disposition = "protected_window_exception" if window["protected_reason"] else "eligible"
            scope = comparison_scope(row)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            text_digest.update(stable_json([ident, text_hash]).encode())
            record = {"id": ident, "tokens": n, "text_sha256": text_hash,
                      "training_content_sha256": hashlib.sha256(stable_json({k:row[k] for k in ("prompt","tools","label")}).encode()).hexdigest(),
                      "disposition": disposition, "scope": scope, "window": window}
            record["before_tokens"] = before_tokens
            token_lengths["before"].append(before_tokens)
            token_lengths["after"].append(n)
            fraction = metadata["decision_ordinal"] / max(1, metadata["trajectory_decision_count"] - 1)
            position = "first_third" if fraction < 1/3 else "middle_third" if fraction < 2/3 else "last_third"
            position_counts[f"{position}/{disposition}"] += 1
            audit.write(json.dumps(record, ensure_ascii=False) + "\n")
            if disposition == "eligible":
                eligible.write(json.dumps({"id": ident, "text": text, "scope": scope, "tokens": n}, ensure_ascii=False) + "\n")
                scopes[scope] += 1
            counts[disposition] += 1
            if len(seen) % 500 == 0:
                audit.flush()
                print(len(seen), dict(counts), flush=True)
    from drug_agent.toolrl.v8_dataset import percentiles
    manifest = {"input": str(args.input.resolve()), "input_sha256": sha256_file(args.input),
                "records": len(seen), "counts": dict(counts), "eligible_scope_sizes": dict(scopes),
                "encoding_model": args.model, "embedding_limit": 32768,
                "text_serialization": VERSION, "text_manifest_sha256": text_digest.hexdigest(),
                "comparison_boundary_version": SCOPE_VERSION,
                "public_catalog_versions":dict(catalog_counts), "trajectory_position_counts":dict(position_counts),
                "public_system_versions":dict(system_counts),
                "token_distributions":{k:{**percentiles(v), "min":min(v)} for k,v in token_lengths.items()},
                "encoder_truncation": False, "history_window_is_lossy": True,
                "tool_results_in_encoding":False, "system_messages_in_encoding":False,
                "max_chars": None, "pretokenize": False,
                "tokenizer_files": {p.name:sha256_file(p) for p in Path(args.model).glob("*token*.json")}}
    (root / "preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**{k:v for k,v in manifest.items() if k != "eligible_scope_sizes"},
                      "comparison_scopes":len(scopes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
