from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "drug_agent_qwen35_toolrl_decision_v8"
DESCRIPTION_VERSION = "toolrl_v8_selection_description_v1"
EMBEDDING_INSTRUCTION = (
    "Group drug-agent decisions by equivalent tool-use behavior and decision context, "
    "not merely by task topic. Preserve differences in tools, skills, parameters, "
    "constraints, and prior success or failure."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _strip_training_fields(message: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(message)
    out.pop("step_loss_mask", None)
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _assistant_messages(messages: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return [(index, message) for index, message in enumerate(messages) if message.get("role") == "assistant"]


def _semantic_decisions(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in record.get("events", []) if event.get("type") == "assistant_decision"]


def _semantic_call(call: dict[str, Any]) -> dict[str, Any]:
    return {"name": str(call.get("name") or ""), "arguments": copy.deepcopy(call.get("arguments") or {})}


def _qwen_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Qwen tool-call arguments must be an object")
    return {"name": str(function.get("name") or ""), "arguments": copy.deepcopy(arguments)}


def _parse_final(value: Any) -> Any:
    if not isinstance(value, str):
        return copy.deepcopy(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _outcome_from_history(prompt: list[dict[str, Any]]) -> str:
    observations = [message for message in prompt if message.get("role") == "tool"]
    if not observations:
        return "none"
    latest = observations[-1]
    content = str(latest.get("content") or "").lower()
    if any(token in content for token in ('"is_error":true', '"ok":false', '"status":"error"', '"status":"failed"', '"status":"timeout"', "traceback")):
        return "error"
    if any(token in content for token in ('"is_error":false', '"ok":true', '"status":"success"', '"status":"completed"', "<skill_content")):
        return "success"
    return "unknown"


def _bounded_text(value: Any, limit: int) -> tuple[str, bool]:
    text = value if isinstance(value, str) else stable_json(value)
    if len(text) <= limit:
        return text, False
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n…[length={len(text)},sha256={digest}]…\n{tail}", True


def _call_scope(calls: list[dict[str, Any]], task_type: str, outcome: str) -> dict[str, Any]:
    names = [str(call.get("name") or "") for call in calls]
    scope: dict[str, Any] = {
        "decision_type": "tool_call",
        "task_type": task_type,
        "ordered_tool_names": names,
        "call_count": len(calls),
        "prior_outcome": outcome,
    }
    if names and all(name == "skill" for name in names):
        scope["skill_names"] = [str((call.get("arguments") or {}).get("name") or "") for call in calls]
    if names and all(name.lower() == "read" for name in names):
        paths = [str((call.get("arguments") or {}).get("file_path") or "") for call in calls]
        scope["read_kinds"] = ["skill" if path.endswith("SKILL.md") else "ordinary" for path in paths]
        scope["skill_paths"] = [path for path in paths if path.endswith("SKILL.md")]
    return scope


def build_selection_description(
    *,
    user_task: str,
    prompt: list[dict[str, Any]],
    target: dict[str, Any],
    task_type: str,
) -> tuple[str, str, bool]:
    calls = [_qwen_call(call) for call in (target.get("tool_calls") or [])]
    decision_type = "tool_call" if calls else "final_answer"
    outcome = _outcome_from_history(prompt)
    scope = (
        _call_scope(calls, task_type, outcome)
        if calls
        else {"decision_type": "final_answer", "task_type": task_type, "prior_outcome": outcome}
    )
    lossy = False
    task_text, cut = _bounded_text(user_task, 1600)
    lossy |= cut
    reasoning_text, cut = _bounded_text(target.get("reasoning_content") or "", 3200)
    lossy |= cut
    action_value: Any = calls if calls else target.get("content") or ""
    action_text, cut = _bounded_text(action_value, 6000)
    lossy |= cut

    current_names = {call["name"] for call in calls}
    related: list[dict[str, Any]] = []
    tools_in_history = [message for message in prompt if message.get("role") == "tool"]
    for message in reversed(tools_in_history):
        if current_names and str(message.get("name") or "") not in current_names and related:
            continue
        related.append(message)
        if len(related) == 2:
            break
    related.reverse()
    result_parts: list[str] = []
    for message in related:
        excerpt, cut = _bounded_text(message.get("content") or "", 1600)
        lossy |= cut
        result_parts.append(f"{message.get('name') or 'tool'}: {excerpt}")

    description = "\n".join(
        [
            f"Task: {task_text}",
            f"Decision type: {decision_type}",
            f"Prior result status: {outcome}",
            f"Current action: {action_text}",
            f"Teacher rationale: {reasoning_text}",
            "Prior related results: " + ("\n".join(result_parts) if result_parts else ""),
        ]
    )
    # If long arguments or complex shell text were shortened, do not use the
    # shortened embedding to downsample. These decisions remain valid inputs.
    unsafe = lossy and any(
        len(stable_json(call.get("arguments") or {})) > 6000
        or (call.get("name") == "Bash" and len(str((call.get("arguments") or {}).get("command") or "")) > 500)
        for call in calls
    )
    if unsafe:
        # Hash the complete structured action, not the bounded description.
        # Different values can share the same retained head/tail (large protein
        # payloads with a ligand in the middle are a real example), in which
        # case hashing ``action_text`` would defeat this safety boundary.
        scope["lossy_content_guard"] = hashlib.sha256(
            stable_json(action_value).encode()
        ).hexdigest()[:20]
    return description, stable_json(scope), unsafe


def materialize_decisions(semantic_path: Path, qwen_sft_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semantic_rows = _load_jsonl(semantic_path)
    qwen_rows = _load_jsonl(qwen_sft_path)
    qwen_by_id = {str(row.get("id") or ""): row for row in qwen_rows}
    if len(qwen_by_id) != len(qwen_rows):
        raise ValueError("Qwen SFT view contains missing or duplicate ids")

    output: list[dict[str, Any]] = []
    tool_histogram: Counter[str] = Counter()
    role_histogram: Counter[str] = Counter()
    for trajectory_index, semantic in enumerate(semantic_rows):
        source_id = str(semantic.get("id") or "")
        qwen = qwen_by_id.pop(source_id, None)
        if qwen is None:
            raise ValueError(f"missing Qwen SFT adapter row for {source_id}")
        messages = qwen.get("messages") if isinstance(qwen.get("messages"), list) else []
        qwen_assistants = _assistant_messages(messages)
        decisions = _semantic_decisions(semantic)
        if len(qwen_assistants) != len(decisions):
            raise ValueError(f"assistant decision count mismatch for {source_id}")
        tools = copy.deepcopy(qwen.get("tools") or [])
        for ordinal, ((message_index, assistant), decision) in enumerate(zip(qwen_assistants, decisions, strict=True)):
            semantic_calls = [_semantic_call(call) for call in (decision.get("tool_calls") or [])]
            semantic_calls_for_audit = [
                {"name": call["name"].casefold(), "arguments": call["arguments"]}
                for call in semantic_calls
            ]
            qwen_calls = [_qwen_call(call) for call in (assistant.get("tool_calls") or [])]
            raw_qwen_calls = assistant.get("tool_calls") or []
            for call_index, (source_call, public_call, raw_public_call) in enumerate(
                zip(semantic_calls_for_audit, qwen_calls, raw_qwen_calls, strict=True)
            ):
                source_id_value = str((decision.get("tool_calls") or [])[call_index].get("source_tool_use_id") or "")
                if source_id_value != str(raw_public_call.get("id") or ""):
                    raise ValueError(f"tool call id/order mismatch for {source_id} decision {ordinal}")
                if stable_json(source_call["arguments"]) != stable_json(public_call["arguments"]):
                    raise ValueError(f"tool call argument mismatch for {source_id} decision {ordinal}")
                public_name = str(public_call["name"])
                if not public_name.startswith("mcp__") and source_call["name"] != public_name.casefold():
                    raise ValueError(f"local tool name mismatch for {source_id} decision {ordinal}")
            semantic_final = decision.get("final_answer")
            qwen_final = assistant.get("content") or ""
            if not semantic_calls and str(semantic_final or "") != str(qwen_final):
                raise ValueError(f"final answer mismatch for {source_id} decision {ordinal}")
            if not semantic_calls and semantic_final is None:
                raise ValueError(f"empty target decision for {source_id} decision {ordinal}")

            prompt = [_strip_training_fields(message) for message in messages[:message_index]]
            target = _strip_training_fields(assistant)
            description, comparison_scope, guarded = build_selection_description(
                user_task=str(semantic.get("user_task") or ""),
                prompt=prompt,
                target=target,
                task_type=str((semantic.get("metadata") or {}).get("task_type") or "unknown"),
            )
            decision_type = "tool_call" if qwen_calls else "final_answer"
            role_histogram[decision_type] += 1
            tool_histogram.update(call["name"] for call in qwen_calls)
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": f"{source_id}::decision_{ordinal:04d}",
                    "prompt": prompt,
                    "tools": tools,
                    "label": {
                        "decision_type": decision_type,
                        "target_assistant": target,
                        "target_reasoning": str(decision.get("reasoning") or ""),
                        "target_tool_calls": qwen_calls,
                        "target_final_answer": _parse_final(semantic_final) if decision_type == "final_answer" else None,
                    },
                    "metadata": {
                        "source_id": source_id,
                        "source_message_id": decision.get("source_message_id"),
                        "trajectory_index": trajectory_index,
                        "decision_ordinal": ordinal,
                        "trajectory_decision_count": len(decisions),
                        "decision_type": decision_type,
                        "decision_role": "tool_step" if qwen_calls else "final",
                        "task_type": str((semantic.get("metadata") or {}).get("task_type") or "unknown"),
                        "tool_names": [call["name"] for call in qwen_calls],
                        "tool_call_count": len(qwen_calls),
                        "selection_description": description,
                        "selection_description_version": DESCRIPTION_VERSION,
                        "comparison_scope": comparison_scope,
                        "lossy_content_guard": guarded,
                    },
                }
            )
    if qwen_by_id:
        raise ValueError(f"Qwen SFT view has ids absent from semantic source: {sorted(qwen_by_id)[:5]}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description_version": DESCRIPTION_VERSION,
        "semantic_source": {"path": str(semantic_path.resolve()), "sha256": sha256_file(semantic_path), "records": len(semantic_rows)},
        "qwen_adapter_source": {"path": str(qwen_sft_path.resolve()), "sha256": sha256_file(qwen_sft_path), "records": len(qwen_rows)},
        "decision_records": len(output),
        "trajectory_records": len(semantic_rows),
        "decision_type_counts": dict(role_histogram),
        "tool_call_counts": dict(tool_histogram),
        "lossy_content_guard_records": sum(bool(row["metadata"]["lossy_content_guard"]) for row in output),
        "contract": {
            "one_raw_assistant_response_per_decision": True,
            "parallel_calls_preserve_order_and_multiplicity": True,
            "prompt_excludes_current_target_and_future_observations": True,
            "selection_does_not_rewrite_prompt_or_target": True,
        },
    }
    return output, manifest


def embedding_text(record: dict[str, Any]) -> str:
    description = str((record.get("metadata") or {}).get("selection_description") or "")
    return f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery: {description}"


def quota_for_cluster(size: int) -> int:
    return min(size, max(8, math.ceil(math.sqrt(size))))


def _choose_representatives(indices: list[int], vectors: np.ndarray, records: list[dict[str, Any]], count: int) -> list[int]:
    if count >= len(indices):
        return list(indices)
    matrix = vectors[indices]
    centroid = matrix.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    center_scores = matrix @ centroid
    first_local = min(range(len(indices)), key=lambda i: (-float(center_scores[i]), str(records[indices[i]]["id"])))
    chosen_local = [first_local]
    chosen_sources = {str(records[indices[first_local]]["metadata"]["source_id"])}
    while len(chosen_local) < count:
        best: tuple[float, int, str, int] | None = None
        for local_index, global_index in enumerate(indices):
            if local_index in chosen_local:
                continue
            max_similarity = max(float(matrix[local_index] @ matrix[selected]) for selected in chosen_local)
            source = str(records[global_index]["metadata"]["source_id"])
            candidate = (max_similarity, 1 if source in chosen_sources else 0, str(records[global_index]["id"]), local_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        chosen_local.append(best[3])
        chosen_sources.add(str(records[indices[best[3]]]["metadata"]["source_id"]))
    return [indices[index] for index in chosen_local]


def cluster_and_select(
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    *,
    distance_threshold: float,
    total_budget: int | None = None,
    min_final_records: int = 0,
) -> tuple[list[int], dict[str, Any], list[dict[str, Any]]]:
    from sklearn.cluster import AgglomerativeClustering

    if vectors.shape[0] != len(records):
        raise ValueError("embedding row count does not match decision records")
    if not 0 < distance_threshold < 2:
        raise ValueError("distance_threshold must be between 0 and 2")
    if min_final_records < 0:
        raise ValueError("min_final_records must be non-negative")
    scopes: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        scopes[str(record["metadata"]["comparison_scope"])].append(index)

    clusters: list[list[int]] = []
    for scope in sorted(scopes):
        indices = scopes[scope]
        if len(indices) == 1:
            clusters.append(indices)
            continue
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=distance_threshold,
        ).fit_predict(vectors[indices])
        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in zip(indices, labels, strict=True):
            by_label[int(label)].append(index)
        clusters.extend(by_label[label] for label in sorted(by_label))

    default_quotas = [quota_for_cluster(len(cluster)) for cluster in clusters]
    quotas = list(default_quotas)
    minimum = len(clusters)
    default_total = sum(default_quotas)
    if total_budget is not None:
        if total_budget < minimum:
            raise ValueError(f"budget {total_budget} cannot cover {minimum} homogeneous clusters")
        if total_budget < default_total:
            quotas = [1] * len(clusters)
            remaining = total_budget - minimum
            candidates: list[tuple[float, int, int]] = []
            for cluster_index, (cluster, cap) in enumerate(zip(clusters, default_quotas, strict=True)):
                for slot in range(1, cap):
                    priority = len(cluster) / (slot + 1)
                    candidates.append((-priority, cluster_index, slot))
            allocated_slots: set[tuple[int, int]] = set()
            final_clusters = {
                cluster_index
                for cluster_index, cluster in enumerate(clusters)
                if str(records[cluster[0]]["metadata"].get("decision_type") or "")
                == "final_answer"
            }
            current_final = len(final_clusters)
            final_needed = max(0, min_final_records - current_final)
            # The ordinary sqrt quota is the default diversity policy, but a
            # strict global budget can otherwise collapse terminal supervision
            # to one row per final-answer cluster.  Build a final-only reserve
            # up to the real cluster size; it is consumed solely to satisfy the
            # explicit minimum and is not available to general allocation.
            final_candidates = sorted(
                (
                    (-len(clusters[cluster_index]) / (slot + 1), cluster_index, slot)
                    for cluster_index in final_clusters
                    for slot in range(1, len(clusters[cluster_index]))
                )
            )
            if final_needed > len(final_candidates) or final_needed > remaining:
                raise ValueError(
                    f"budget {total_budget} cannot retain {min_final_records} final decisions "
                    f"while covering all {minimum} homogeneous clusters"
                )
            for _, cluster_index, slot in final_candidates[:final_needed]:
                quotas[cluster_index] += 1
                allocated_slots.add((cluster_index, slot))
            remaining -= final_needed
            for _, cluster_index, slot in (
                candidate
                for candidate in sorted(candidates)
                if (candidate[1], candidate[2]) not in allocated_slots
            ):
                if remaining == 0:
                    break
                quotas[cluster_index] += 1
                remaining -= 1

    selected: set[int] = set()
    assignments: list[dict[str, Any]] = []
    for cluster_id, (cluster, quota) in enumerate(zip(clusters, quotas, strict=True)):
        representatives = _choose_representatives(cluster, vectors, records, quota)
        selected.update(representatives)
        rep_matrix = vectors[representatives]
        for index in cluster:
            similarities = rep_matrix @ vectors[index]
            best_local = int(np.argmax(similarities))
            assignments.append(
                {
                    "decision_id": records[index]["id"],
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster),
                    "cluster_keep_count": quota,
                    "selected": index in selected,
                    "representative_id": records[representatives[best_local]]["id"],
                    "representative_similarity": min(1.0, max(-1.0, float(similarities[best_local]))),
                    "reason": "selected_homogeneous_cluster_representative" if index in selected else "representative_downsampling_not_selected",
                }
            )
    selected_ordered = sorted(selected)
    report = {
        "distance_threshold": distance_threshold,
        "comparison_scopes": len(scopes),
        "homogeneous_clusters": len(clusters),
        "input_records": len(records),
        "selected_records": len(selected_ordered),
        "not_selected_records": len(records) - len(selected_ordered),
        "default_quota_total": default_total,
        "requested_total_budget": total_budget,
        "minimum_final_records": min_final_records,
        "cluster_size_histogram": dict(Counter(str(len(cluster)) for cluster in clusters)),
        "largest_clusters": sorted(
            ({"cluster_id": i, "size": len(cluster), "keep": quotas[i]} for i, cluster in enumerate(clusters)),
            key=lambda item: (-item["size"], item["cluster_id"]),
        )[:50],
    }
    return selected_ordered, report, assignments


def percentiles(values: Iterable[int]) -> dict[str, int]:
    array = np.asarray(list(values), dtype=np.int64)
    if not len(array):
        return {key: 0 for key in ("p50", "p90", "p95", "p99", "max")}
    return {
        "p50": int(np.percentile(array, 50, method="higher")),
        "p90": int(np.percentile(array, 90, method="higher")),
        "p95": int(np.percentile(array, 95, method="higher")),
        "p99": int(np.percentile(array, 99, method="higher")),
        "max": int(array.max()),
    }
