from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.constants import DATA_ROOT, SLIME_DRUG_RUNS_ROOT
from drug_agent.protocol.parse_policy import (
    ROLLOUT_MODE_TRAIN_STRICT,
    parse_action_with_policy,
    resolve_rollout_controls,
)
from drug_agent.tools.tool_success import evaluate_tool_success
from drug_agent.utils import ensure_dir, write_json


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _audit_parse_modes() -> dict[str, Any]:
    cases = {
        "A_pure_tool_call": '{"type":"tool_call","tool_name":"dummy","arguments":{}}',
        "B_prose_plus_json": 'Answer:\\n{"type":"tool_call","tool_name":"dummy","arguments":{}}',
        "C_markdown_fenced_json": '```json\\n{"type":"tool_call","tool_name":"dummy","arguments":{}}\\n```',
        "D_think_plus_json": '</think>\\n{"type":"tool_call","tool_name":"dummy","arguments":{}}',
        "E_missing_required_arg": '{"type":"tool_call","tool_name":"dummy","arguments":{}}',
        "F_wrong_type_arg": '{"type":"tool_call","tool_name":"dummy","arguments":{"n":"oops"}}',
        "I_valid_final_answer": (
            '{"type":"final_answer","answer":{"summary":"ok","evidence":[],"result":{"ranked_molecules":[]},"ranked_molecules":[]}}'
        ),
        "J_malformed_final_answer": '{"type":"final_answer","answer":{"summary":1,"evidence":[],"result":{}}}',
    }
    expected = {
        "A_pure_tool_call": (True, True),
        "B_prose_plus_json": (False, True),
        "C_markdown_fenced_json": (False, True),
        "D_think_plus_json": (False, True),
        "E_missing_required_arg": (True, True),
        "F_wrong_type_arg": (True, True),
        "I_valid_final_answer": (True, True),
        "J_malformed_final_answer": (False, False),
    }

    rows = []
    mismatches = []
    for name, text in cases.items():
        strict, strict_recovery, strict_normalized, strict_source = parse_action_with_policy(
            text,
            parse_recovery_enabled=False,
        )
        permissive, permissive_recovery, permissive_normalized, permissive_source = parse_action_with_policy(
            text,
            parse_recovery_enabled=True,
        )
        row = {
            "case": name,
            "strict_ok": bool(strict.ok),
            "strict_source": strict_source,
            "strict_error_type": strict.error_type,
            "strict_normalized_preview": strict_normalized[:200],
            "permissive_ok": bool(permissive.ok),
            "permissive_source": permissive_source,
            "permissive_recovered": bool(
                isinstance(permissive_recovery, dict) and permissive_recovery.get("recovered") is True
            ),
            "permissive_error_type": permissive.error_type,
            "permissive_normalized_preview": permissive_normalized[:200],
        }
        rows.append(row)

        expected_strict, expected_permissive = expected[name]
        if row["strict_ok"] != expected_strict or row["permissive_ok"] != expected_permissive:
            mismatches.append(
                {
                    "case": name,
                    "expected": {"strict_ok": expected_strict, "permissive_ok": expected_permissive},
                    "actual": {"strict_ok": row["strict_ok"], "permissive_ok": row["permissive_ok"]},
                }
            )

    return {
        "ok": len(mismatches) == 0,
        "cases": rows,
        "mismatches": mismatches,
    }


def _audit_tool_success_semantics() -> dict[str, Any]:
    cases = [
        {
            "name": "transport_error",
            "kwargs": {
                "transport_ok": False,
                "tool_schema_valid": True,
                "parsed_payload": None,
                "raw_payload": None,
            },
            "expect_semantic_success": False,
        },
        {
            "name": "schema_validation_error",
            "kwargs": {
                "transport_ok": True,
                "tool_schema_valid": False,
                "parsed_payload": None,
                "raw_payload": None,
            },
            "expect_semantic_success": False,
        },
        {
            "name": "raw_is_error_true",
            "kwargs": {
                "transport_ok": True,
                "tool_schema_valid": True,
                "parsed_payload": {"status": "success"},
                "raw_payload": {"isError": True},
            },
            "expect_semantic_success": False,
        },
        {
            "name": "parsed_status_error",
            "kwargs": {
                "transport_ok": True,
                "tool_schema_valid": True,
                "parsed_payload": {"status": "error"},
                "raw_payload": {"isError": False},
            },
            "expect_semantic_success": False,
        },
        {
            "name": "structured_status_error",
            "kwargs": {
                "transport_ok": True,
                "tool_schema_valid": True,
                "parsed_payload": {"structuredContent": {"status": "error"}},
                "raw_payload": {"isError": False},
            },
            "expect_semantic_success": False,
        },
        {
            "name": "explicit_success",
            "kwargs": {
                "transport_ok": True,
                "tool_schema_valid": True,
                "parsed_payload": {"status": "success", "result": {"value": 1}},
                "raw_payload": {"isError": False},
            },
            "expect_semantic_success": True,
        },
        {
            "name": "semantic_unknown",
            "kwargs": {
                "transport_ok": True,
                "tool_schema_valid": True,
                "parsed_payload": {"value": 1},
                "raw_payload": {"isError": False},
            },
            "expect_semantic_success": False,
        },
    ]

    rows = []
    mismatches = []
    for case in cases:
        out = evaluate_tool_success(unknown_as_failure=True, **case["kwargs"])
        row = {
            "name": case["name"],
            "output": out,
            "expect_semantic_success": case["expect_semantic_success"],
        }
        rows.append(row)
        if bool(out.get("tool_semantic_success")) != bool(case["expect_semantic_success"]):
            mismatches.append(
                {
                    "name": case["name"],
                    "expected": case["expect_semantic_success"],
                    "actual": out.get("tool_semantic_success"),
                }
            )

    return {
        "ok": len(mismatches) == 0,
        "cases": rows,
        "mismatches": mismatches,
    }


async def _reward_probe_async() -> dict[str, Any]:
    from slime.utils.types import Sample

    from drug_agent.rollout.reward_func import reward_func

    sample = Sample(
        prompt="dummy",
        label={"ground_truth": "XYZ", "expected": {"answer_hit_pass": True}},
        metadata={
            "env_kwargs": {"max_steps": 6},
            "drug_agent_trace": {
                "rollout_mode": ROLLOUT_MODE_TRAIN_STRICT,
                "parse_recovery_enabled": False,
                "done_reason": "final_answer",
                "num_steps": 3,
                "num_invalid": 1,
                "num_parse_recovery": 0,
                "strict_valid_count": 2,
                "recovered_valid_count": 0,
                "num_tool_success": 1,
                "num_tool_error": 1,
                "num_tool_schema_error": 1,
                "num_tool_execution_success": 1,
                "num_tool_semantic_error": 1,
                "num_tool_semantic_unknown": 0,
                "num_transport_error": 0,
                "final_answer": {
                    "summary": "contains XYZ",
                    "evidence": [],
                    "result": {"ranked_molecules": []},
                    "ranked_molecules": [],
                },
                "actions": [
                    {"parsed": {"ok": True}},
                    {"parsed": {"ok": False}, "parse_recovery": None},
                    {"parsed": {"ok": True}},
                ],
                "observations": [
                    {
                        "type": "tool_result",
                        "ok": True,
                        "transport_ok": True,
                        "tool_schema_valid": True,
                        "tool_execution_success": True,
                        "tool_semantic_success": True,
                        "semantic_unknown": False,
                    },
                    {
                        "type": "tool_result",
                        "ok": False,
                        "transport_ok": False,
                        "tool_schema_valid": False,
                        "tool_execution_success": False,
                        "tool_semantic_success": False,
                        "semantic_unknown": False,
                        "error": {"type": "ToolValidationError", "message": "missing arg"},
                    },
                ],
            },
        },
    )
    out = await reward_func(SimpleNamespace(), sample)
    return out


def _audit_reward_semantics() -> dict[str, Any]:
    try:
        out = asyncio.run(_reward_probe_async())
    except Exception as exc:
        return {
            "ok": False,
            "needs_manual_review": True,
            "reason": f"reward_probe_dependency_or_runtime_error: {type(exc).__name__}: {exc}",
        }
    required_components = {
        "format_reward",
        "action_valid_reward",
        "tool_schema_reward",
        "tool_execution_reward",
        "tool_semantic_reward",
        "parse_recovery_penalty",
        "progress_reward",
        "final_reward",
        "efficiency_reward",
    }
    required_diag = {
        "num_invalid",
        "num_parse_recovery",
        "strict_valid_count",
        "recovered_valid_count",
        "num_tool_schema_error",
        "num_tool_execution_success",
        "num_tool_semantic_error",
        "num_transport_error",
        "num_final_answer",
        "num_max_steps",
    }
    components = out.get("components") if isinstance(out.get("components"), dict) else {}
    diagnostics = out.get("diagnostics") if isinstance(out.get("diagnostics"), dict) else {}

    missing_components = sorted(k for k in required_components if k not in components)
    missing_diag = sorted(k for k in required_diag if k not in diagnostics)

    return {
        "ok": len(missing_components) == 0 and len(missing_diag) == 0,
        "output": out,
        "missing_components": missing_components,
        "missing_diagnostics": missing_diag,
    }


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _audit_sft_strict(input_path: Path | None, tokenizer_path: Path | None) -> dict[str, Any]:
    if input_path is None:
        return {
            "ok": False,
            "needs_manual_review": True,
            "reason": "sft_input_not_found",
        }

    cmd = [
        sys.executable,
        "drug_agent/data/validate_sft_messages.py",
        "--input",
        str(input_path),
        "--protocol",
        "auto",
    ]
    if tokenizer_path is not None and tokenizer_path.exists():
        cmd.extend(["--tokenizer", str(tokenizer_path)])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    parsed = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except Exception:
            parsed = {"raw_stdout": stdout}
    else:
        parsed = {"raw_stdout": ""}

    out = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": cmd,
        "summary": parsed,
        "stderr": proc.stderr.strip(),
    }
    if isinstance(parsed, dict):
        previews = parsed.get("apply_chat_template_failed_preview")
        if isinstance(previews, list) and previews:
            reason = previews[0].get("reason")
            if isinstance(reason, str) and "No module named 'transformers'" in reason:
                out["needs_manual_review"] = True
                out["reason"] = "transformers_not_installed_in_current_python_env"
        if parsed.get("deprecated_legacy_protocol"):
            out["needs_manual_review"] = True
            out["reason"] = "legacy_action_json_sft_detected"
    return out


def _audit_runtime_defaults(repo_root: Path) -> dict[str, Any]:
    saved_mode = os.environ.get("DRUG_AGENT_ROLLOUT_MODE")
    saved_recovery = os.environ.get("DRUG_AGENT_ALLOW_PARSE_RECOVERY")
    try:
        if "DRUG_AGENT_ROLLOUT_MODE" in os.environ:
            del os.environ["DRUG_AGENT_ROLLOUT_MODE"]
        if "DRUG_AGENT_ALLOW_PARSE_RECOVERY" in os.environ:
            del os.environ["DRUG_AGENT_ALLOW_PARSE_RECOVERY"]
        controls = resolve_rollout_controls()
    finally:
        if saved_mode is None:
            os.environ.pop("DRUG_AGENT_ROLLOUT_MODE", None)
        else:
            os.environ["DRUG_AGENT_ROLLOUT_MODE"] = saved_mode
        if saved_recovery is None:
            os.environ.pop("DRUG_AGENT_ALLOW_PARSE_RECOVERY", None)
        else:
            os.environ["DRUG_AGENT_ALLOW_PARSE_RECOVERY"] = saved_recovery

    ppo_text = (repo_root / "drug_agent/scripts/run_qwen3_5_0_8b_drug_ppo_smoke.sh").read_text(encoding="utf-8")
    grpo_text = (repo_root / "drug_agent/scripts/run_qwen3_5_0_8b_drug_grpo_smoke.sh").read_text(encoding="utf-8")
    sft_text = (repo_root / "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh").read_text(encoding="utf-8")

    ppo_has_runtime = (
        "DRUG_AGENT_ROLLOUT_MODE" in ppo_text
        and "train_strict" in ppo_text
        and "DRUG_AGENT_ALLOW_PARSE_RECOVERY" in ppo_text
        and ":-0" in ppo_text
    )
    grpo_has_runtime = (
        "DRUG_AGENT_ROLLOUT_MODE" in grpo_text
        and "train_strict" in grpo_text
        and "DRUG_AGENT_ALLOW_PARSE_RECOVERY" in grpo_text
        and ":-0" in grpo_text
    )
    sft_unconditional_debug = "--debug-train-only" in sft_text and 'SFT_DEBUG_TRAIN_ONLY:-0' not in sft_text

    return {
        "ok": (
            controls.get("rollout_mode") == ROLLOUT_MODE_TRAIN_STRICT
            and controls.get("parse_recovery_enabled") is False
            and ppo_has_runtime
            and grpo_has_runtime
            and not sft_unconditional_debug
        ),
        "default_rollout_controls": controls,
        "ppo_runtime_env_strict": ppo_has_runtime,
        "grpo_runtime_env_strict": grpo_has_runtime,
        "sft_unconditional_debug_train_only": sft_unconditional_debug,
    }


def _audit_debug_isolation(repo_root: Path) -> dict[str, Any]:
    offenders = []
    for path in (repo_root / "drug_agent").rglob("*.py"):
        rel = path.relative_to(repo_root)
        if "tools_debug" in rel.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "tools_debug" in text:
            offenders.append(str(rel))
    return {
        "ok": len(offenders) == 0,
        "offenders": offenders,
    }


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Drug Agent Training Compliance Audit",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- repo_root: `{report['repo_root']}`",
        f"- overall_ok: `{report['overall_ok']}`",
        "",
        "## Findings",
    ]
    if report.get("deprecated_audit_helper"):
        lines.insert(5, "- authority: `deprecated helper`")
        lines.insert(6, "- note: informational only; use validate_sft_messages.py for protocol validation")
    for item in report["findings"]:
        lines.append(f"- [{item['severity']}] {item['id']}: {item['message']}")
    lines.append("")
    lines.append("## Checks")
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value.get('ok')}`")
    if report["needs_manual_review"]:
        lines.append("")
        lines.append("## Needs Manual Review")
        for item in report["needs_manual_review"]:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit drug_agent formal-training compliance without launching training.")
    parser.add_argument("--repo-root", type=str, default=str(Path.cwd()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sft-input", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = SLIME_DRUG_RUNS_ROOT / f"training_compliance_audit_{_now_tag()}"

    default_sft_paths: list[Path] = []
    if args.sft_input:
        default_sft_paths.append(Path(args.sft_input))
    default_sft_paths.extend(
        [
            DATA_ROOT / "mcp_sft_all.train.jsonl",
        ]
    )
    if os.environ.get("DRUG_AGENT_DATA_ROOT"):
        default_sft_paths.append(Path(os.environ["DRUG_AGENT_DATA_ROOT"]) / "sft/mixed.jsonl")
    sft_input = _find_first_existing(default_sft_paths)

    tokenizer = None
    if args.tokenizer:
        tokenizer = Path(args.tokenizer)
    else:
        default_tokenizers = [
            DATA_ROOT / "Qwen3.5-0.8B",
            DATA_ROOT / "Qwen3.5-27B",
        ]
        tokenizer = _find_first_existing(default_tokenizers)

    checks = {
        "parse_modes": _audit_parse_modes(),
        "tool_success_semantics": _audit_tool_success_semantics(),
        "reward_semantics": _audit_reward_semantics(),
        "sft_strict_validation": _audit_sft_strict(sft_input, tokenizer),
        "runtime_defaults": _audit_runtime_defaults(repo_root),
        "debug_isolation": _audit_debug_isolation(repo_root),
    }

    findings: list[dict[str, Any]] = []
    needs_manual_review: list[str] = []

    if not checks["parse_modes"]["ok"]:
        findings.append(
            {
                "id": "parse_mode_behavior",
                "severity": "critical",
                "message": "parse mode behavior mismatch with strict/permissive expectations",
            }
        )
    if not checks["tool_success_semantics"]["ok"]:
        findings.append(
            {
                "id": "tool_success_semantics",
                "severity": "critical",
                "message": "tool success semantics do not match training compliance expectations",
            }
        )
    reward_check = checks["reward_semantics"]
    if reward_check.get("needs_manual_review"):
        needs_manual_review.append(str(reward_check.get("reason")))
    elif not reward_check["ok"]:
        findings.append(
            {
                "id": "reward_semantics",
                "severity": "critical",
                "message": "reward output missing required component/diagnostic fields",
            }
        )
    if not checks["runtime_defaults"]["ok"]:
        findings.append(
            {
                "id": "runtime_defaults",
                "severity": "medium",
                "message": "runtime defaults are not fully strict by default",
            }
        )
    if not checks["debug_isolation"]["ok"]:
        findings.append(
            {
                "id": "debug_isolation",
                "severity": "medium",
                "message": "training path contains debug-only module references",
            }
        )

    sft_check = checks["sft_strict_validation"]
    sft_summary = sft_check.get("summary") if isinstance(sft_check.get("summary"), dict) else {}
    if sft_check.get("needs_manual_review"):
        needs_manual_review.append(str(sft_check.get("reason") or "sft_strict_validation_needs_manual_review"))
    elif isinstance(sft_summary, dict) and sft_summary.get("protocol_warning") == "mixed_protocols_detected":
        needs_manual_review.append("legacy_or_mixed_sft_protocol_detected")
    elif isinstance(sft_summary, dict) and sft_summary.get("chat_template_import_failed"):
        needs_manual_review.append("chat_template_dependency_unavailable")
    elif not sft_check["ok"]:
        findings.append(
            {
                "id": "sft_strict_validation",
                "severity": "medium",
                "message": "SFT strict validation failed or tokenizer/template check failed",
            }
        )

    try:
        ensure_dir(out_dir)
    except Exception:
        out_dir = ensure_dir(repo_root / "outputs" / "training_compliance_audit_fallback" / _now_tag())

    overall_ok = len(findings) == 0
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "output_dir": str(out_dir),
        "overall_ok": overall_ok,
        "deprecated_audit_helper": True,
        "checks": checks,
        "findings": findings,
        "needs_manual_review": needs_manual_review,
    }
    json_path = out_dir / "audit_report.json"
    md_path = out_dir / "audit_report.md"
    write_json(json_path, report)
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": overall_ok,
                "output_dir": str(out_dir),
                "audit_report_json": str(json_path),
                "audit_report_md": str(md_path),
                "deprecated_audit_helper": True,
                "findings": findings,
                "needs_manual_review": needs_manual_review,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
