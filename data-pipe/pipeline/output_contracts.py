from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


CONTRACT_VERSION = "molbench_answer_contract_v4"


@dataclass(frozen=True)
class TaskConstraints:
    task_type: str
    candidates: tuple[str, ...] = ()
    exact_count: int | None = None
    complete_ranking: bool = False


def task_constraints(task: str, task_type: str) -> TaskConstraints:
    """Read public task requirements only; ground truth is never an input."""
    if task_type == "ac":
        values = re.findall(r"Molecule [AB]:\s*([^\n]+)", task)
        if len(values) != 2:
            raise ValueError("AC task must contain two explicit candidates")
        return TaskConstraints(task_type, tuple(v.strip() for v in values), 1)
    if task_type == "pf":
        block = re.search(r"SMILES:\s*\n(.*?)(?:Constraints:|Output format:|Selection requirement:)", task, re.S)
        if block is None:
            raise ValueError("PF task must contain a SMILES candidate block")
        values = tuple(line.strip() for line in block[1].splitlines() if line.strip())
        if not values:
            raise ValueError("PF candidate block is empty")
        single = bool(re.search(r"Print ONLY the selected SMILES|exactly one candidate|Find the molecule sharing the MOST", task, re.I))
        return TaskConstraints(task_type, values, 1 if single else None)
    if task_type == "vs":
        payload = strict_json_loads(task)
        values = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v.strip() for v in values):
            raise ValueError("VS task must contain candidate strings")
        return TaskConstraints(task_type, tuple(values), len(values), True)
    if task_type not in {"kg", "e2e", "mo-opt", "mo-edit"}:
        raise ValueError(f"unsupported task type {task_type!r}")
    return TaskConstraints(task_type)


def strict_json_loads(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

ANSWER_KEYS = {
    "ac": "answer_smiles",
    "pf": "selected_smiles",
    "vs": "ranked_smiles",
    "kg": "result",
    "e2e": "result",
    "mo-opt": "Final Target Molecule",
    "mo-edit": "output",
}

LEGACY_CONTRACTS_V1 = {
    "ac": (
        "Return exactly one JSON object and nothing else. It must contain exactly "
        'these fields: {"answer_smiles":"<one exact candidate SMILES>","evidence":[]}.'
    ),
    "pf": (
        "Return exactly one JSON object and nothing else. It must contain exactly "
        'these fields: {"selected_smiles":["<each exact passing candidate SMILES>"],"evidence":[]}.'
    ),
    "vs": (
        "Return exactly one JSON object and nothing else. It must contain exactly "
        'these fields: {"ranked_smiles":["<all candidate SMILES in rank order>"],"evidence":[]}.'
    ),
    "kg": (
        "Return exactly one JSON object and nothing else. It must contain exactly "
        'these fields: {"result":<the requested result or deliverables>,"evidence":[]}.'
    ),
    "e2e": (
        "Return exactly one JSON object and nothing else. It must contain exactly "
        'these fields: {"result":<the requested result or deliverables>,"evidence":[]}.'
    ),
}

CONTRACTS = dict(LEGACY_CONTRACTS_V1)
CONTRACTS.update({task: 'Return exactly one JSON object and nothing else. It must contain exactly this field: '
                  + json.dumps({ANSWER_KEYS[task]: "<modified molecule SMILES>"}) + "."
                  for task in ("mo-opt", "mo-edit")})


def normalize_task_prompt(task: str, task_type: str) -> str:
    """Return the exact task text shown to the teacher and retained for SFT."""
    task_type = task_type.strip().lower()
    if task_type not in CONTRACTS:
        raise ValueError(f"unsupported task type {task_type!r}")
    contract = CONTRACTS[task_type]
    text = task.strip()

    if task_type in {"mo-opt", "mo-edit"}:
        if text.endswith(contract):
            return text
        text, count = re.subn(
            r'Your response must be directly parsable JSON format:\s*\{[^{}]*\}',
            '', text,
        )
        if count != 1:
            raise ValueError("MO question must have one upstream output format block")
        return text.rstrip() + "\n\nOutput format:\n" + contract

    if task_type == "vs":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("VS task is not a JSON object") from exc
        if not isinstance(payload, dict):
            raise ValueError("VS task is not a JSON object")
        candidates = payload.get("candidates")
        suffix = (f" The ranked_smiles list must contain all {len(candidates)} candidate strings exactly once, in the requested order."
                  if isinstance(candidates, list) else "")
        payload["output_format"] = contract + suffix
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    single = task_type == "pf" and bool(re.search(
        r"Print ONLY the selected SMILES|exactly one candidate|Find the molecule sharing the MOST", text, re.I
    ))
    if single and "The selected_smiles list must contain exactly one candidate." not in text:
        text = text.replace(contract, "").rstrip()
        text = re.sub(r"\s*Output format:\s*$", "", text)
    if text.endswith(contract):
        return text

    legacy_contract = LEGACY_CONTRACTS_V1[task_type]
    legacy_suffix = "\n\nOutput format:\n" + legacy_contract
    if text.endswith(legacy_suffix):
        text = text[: -len(legacy_suffix)].rstrip()

    if task_type == "ac":
        text, count = re.subn(
            r"\s*Only output the corresponding SMILES\.\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if count > 1:
            raise ValueError("AC task has multiple terminal output instructions")
    elif task_type == "pf":
        text, empty_line_count = re.subn(
            r"(?im)^[ \t]*If none satisfy, output an empty line\.[ \t]*\n?",
            "",
            text,
        )
        if empty_line_count > 1:
            raise ValueError("PF task has multiple empty-line output instructions")
        text, count = re.subn(
            r"\s*(?:Output format:\s*)?Print(?: each satisfying| ONLY the selected) SMILES"
            r"(?: on its own line,)? and nothing else\.\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if count > 1:
            raise ValueError("PF task has multiple terminal output instructions")

    if single:
        text += "\n\nSelection requirement:\nThe selected_smiles list must contain exactly one candidate."
    return text.rstrip() + "\n\nOutput format:\n" + contract


def normalize_final_answer(final_answer: str, task_type: str, *, constraints: TaskConstraints | None = None) -> str:
    """Validate the teacher's final answer and serialize its exact two-field contract."""
    task_type = task_type.strip().lower()
    answer_key = ANSWER_KEYS.get(task_type)
    if answer_key is None:
        raise ValueError(f"unsupported task type {task_type!r}")
    try:
        payload = strict_json_loads(final_answer)
    except json.JSONDecodeError as exc:
        raise ValueError("terminal answer is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("terminal answer is not a JSON object")
    required = {answer_key} if task_type in {"mo-opt", "mo-edit"} else {answer_key, "evidence"}
    if set(payload) != required:
        raise ValueError(
            f"terminal answer fields must be exactly {sorted(required)!r}"
        )

    answer = payload[answer_key]
    if task_type in {"ac", "mo-opt", "mo-edit"} and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError(f"{answer_key} must be a non-empty string")
    if task_type in {"pf", "vs"} and (
        not isinstance(answer, list)
        or not all(isinstance(value, str) and value.strip() for value in answer)
    ):
        raise ValueError(f"{answer_key} must be a list of non-empty strings")
    if task_type in {"kg", "e2e"} and answer is None:
        raise ValueError("result must not be null")
    if "evidence" in required and not isinstance(payload["evidence"], list):
        raise ValueError("evidence must be a list")

    if constraints is not None:
        if constraints.task_type != task_type:
            raise ValueError("task constraint type mismatch")
        if task_type in {"ac", "pf", "vs"}:
            values = [answer] if task_type == "ac" else answer
            if any(value not in constraints.candidates for value in values):
                raise ValueError("answer contains a non-candidate value")
            if constraints.exact_count is not None and len(values) != constraints.exact_count:
                raise ValueError(f"answer requires exactly {constraints.exact_count} candidate values")
            if constraints.complete_ranking and Counter(values) != Counter(constraints.candidates):
                raise ValueError("ranking must be a complete candidate permutation")

    return json.dumps(
        {key: payload[key] for key in (answer_key, "evidence") if key in required},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def parsed_answer_values(final_answer: str, task_type: str) -> list[str]:
    payload = json.loads(normalize_final_answer(final_answer, task_type))
    value: Any = payload[ANSWER_KEYS[task_type]]
    if task_type == "ac":
        return [value]
    if task_type in {"pf", "vs"}:
        return list(value)
    return [json.dumps(value, ensure_ascii=False, separators=(",", ":"))]
