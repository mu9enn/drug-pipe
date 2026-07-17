from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL_SCRIPTS = {
    "sft": ROOT / "scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh",
    "sft_4b_smoke": ROOT / "scripts/run_qwen3_5_4b_drug_sft_smoke.sh",
    "sft_4b_full": ROOT / "scripts/run_qwen3_5_4b_drug_sft_full.sh",
    "toolrl": ROOT / "toolrl/scripts/run_toolrl_grpo.sh",
    "toolrl_smoke": ROOT / "toolrl/scripts/run_toolrl_grpo_smoke.sh",
    "toolrl_learn": ROOT / "toolrl/scripts/run_toolrl_grpo_learn.sh",
    "gad_stage2_negatives": ROOT / "gad/scripts/generate_stage2_negatives.sh",
    "gad_stage2_warmup": ROOT / "gad/scripts/run_stage2_discriminator_warmup.sh",
    "gad_discriminator_service": ROOT / "gad/scripts/serve_discriminator.sh",
    "gad_stage3": ROOT / "gad/scripts/run_stage3_gad_grpo.sh",
    "gad_stage3_smoke": ROOT / "gad/scripts/run_stage3_gad_grpo_smoke.sh",
    "gad_stage3_full": ROOT / "gad/scripts/run_stage3_gad_grpo_full.sh",
}
FORMAL_HOOKS = {
    "shared_decision_extractor": ROOT / "decision_extractor.py",
    "sft_materialize": ROOT / "data/materialize_sft_jsonl.py",
    "sft_validate": ROOT / "data/validate_sft_messages.py",
    "toolrl_converter": ROOT / "toolrl/convert_react_to_toolrl_steps.py",
    "toolrl_parser": ROOT / "toolrl/parse_tool_calls.py",
    "toolrl_normalization": ROOT / "toolrl/normalization.py",
    "toolrl_reward": ROOT / "toolrl/molclaw_reward.py",
    "gad_converter": ROOT / "gad/data.py",
    "gad_discriminator": ROOT / "gad/discriminator.py",
    "gad_reward": ROOT / "gad/reward.py",
    "gad_negative_cache": ROOT / "gad/negative_cache.py",
    "gad_warmup": ROOT / "gad/train_discriminator.py",
    "gad_service": ROOT / "gad/service.py",
    "gad_trajectory_logger": ROOT / "gad/trajectory_logger.py",
}
LEGACY_ONLINE_TRAINING = (
    ROOT / "scripts/run_qwen3_5_0_8b_drug_grpo_smoke.sh",
    ROOT / "scripts/run_qwen3_5_0_8b_drug_grpo_learn.sh",
    ROOT / "scripts/run_qwen3_5_0_8b_drug_ppo_smoke.sh",
)
FORBIDDEN_TRAINING_REFERENCES = (
    "drug_agent.rollout.generate_with_drug_agent.generate",
    "--custom-generate-function-path",
    "MCPToolExecutor",
    "drug_agent.tools.mcp_client",
    "drug_agent.tools.tool_executor",
    "MOLCLAW_SCP_SERVER_URL",
    "MOLCLAW_SCP_API_KEY",
    "EnvManager",
    "VectorEnv",
    "subprocess",
    "os.system",
    "requests.",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _delegates_to_guarded_entry(text: str) -> bool:
    guarded_entries = (
        "run_qwen3_5_0_8b_drug_sft_smoke.sh",
        "run_toolrl_grpo.sh",
        "run_stage3_gad_grpo.sh",
    )
    return any(entry in text for entry in guarded_entries)


def audit() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scripts: dict[str, Any] = {}
    for name, path in FORMAL_SCRIPTS.items():
        text = _read(path)
        hits = [token for token in FORBIDDEN_TRAINING_REFERENCES if token in text]
        direct_guard = "offline_training_env.sh" in text
        guarded_delegate = _delegates_to_guarded_entry(text)
        offline_guard = direct_guard or guarded_delegate
        guard_before_launch = (
            direct_guard and text.index("offline_training_env.sh") < text.find("ray start")
            if "ray start" in text
            else offline_guard
        )
        scripts[name] = {
            "path": str(path),
            "offline_guard": offline_guard,
            "direct_guard": direct_guard,
            "guarded_delegate": guarded_delegate,
            "guard_before_launch": guard_before_launch,
            "forbidden_hits": hits,
        }
        if hits or not guard_before_launch:
            findings.append({"severity": "critical", "type": "formal_script_not_offline", "name": name, "hits": hits})

    hooks: dict[str, Any] = {}
    for name, path in FORMAL_HOOKS.items():
        text = _read(path)
        hits = [token for token in FORBIDDEN_TRAINING_REFERENCES if token in text]
        hooks[name] = {"path": str(path), "forbidden_hits": hits}
        if hits:
            findings.append({"severity": "critical", "type": "formal_hook_tool_dependency", "name": name, "hits": hits})

    legacy = {}
    for path in LEGACY_ONLINE_TRAINING:
        text = _read(path)
        disabled = "reject_legacy_online_training.sh" in text
        disabled_before_launch = disabled and text.index("reject_legacy_online_training.sh") < text.index("ray start")
        legacy[path.name] = {"path": str(path), "disabled_before_launch": disabled_before_launch}
        if not disabled_before_launch:
            findings.append({"severity": "critical", "type": "legacy_online_training_enabled", "path": str(path)})

    gad_stage3 = _read(FORMAL_SCRIPTS["gad_stage3"])
    gad_stage2 = _read(FORMAL_SCRIPTS["gad_stage2_negatives"])
    sft = _read(FORMAL_SCRIPTS["sft"])
    toolrl = _read(FORMAL_SCRIPTS["toolrl"])
    training_contract = {
        "sft_native_teacher_forcing": "slime.rollout.sft_rollout.generate_rollout" in sft
        and "--loss-type sft_loss" in sft,
        "toolrl_native_single_response_rollout": "slime.rollout.sglang_rollout.generate_rollout" in toolrl
        and "--custom-generate-function-path" not in toolrl,
        "toolrl_offline_rule_reward": "drug_agent.toolrl.molclaw_reward.reward_func" in toolrl,
        "toolrl_hooks_not_environment_overridable": "ROLLOUT_FUNCTION_PATH=${" not in toolrl
        and "CUSTOM_RM_PATH=${" not in toolrl,
    }
    if not all(training_contract.values()):
        findings.append({"severity": "critical", "type": "offline_training_contract_broken", "details": training_contract})
    gad_contract = {
        "stage3_current_student_generation": "--custom-generate-function-path" not in gad_stage3
        and "slime.rollout.generate_with_drug_agent" not in gad_stage3,
        "stage3_group_discriminator_reward": "--group-rm" in gad_stage3
        and "drug_agent.gad.reward.reward_func" in gad_stage3,
        "stage2_current_student_generation": "--custom-generate-function-path" not in gad_stage2,
        "stage2_no_tool_reward": "drug_agent.gad.negative_cache.zero_reward" in gad_stage2,
        "discriminator_score_and_update": "/score-and-update" in _read(ROOT / "gad/reward.py"),
    }
    if not all(gad_contract.values()):
        findings.append({"severity": "critical", "type": "gad_on_policy_contract_broken", "details": gad_contract})

    network_targets = {
        "formal_training_allowed": [
            "local Ray dashboard/job submission",
            "SGLang model generation/router",
            "GAD discriminator /health and /score-and-update",
            "checkpoint/logging/metrics infrastructure",
        ],
        "formal_training_forbidden": ["MolClaw/MCP", "sandbox/tool executor", "agent environment"],
    }
    return {
        "ok": not findings,
        "policy": "offline fixed states + current-policy next-action sampling; actions are never executed during training",
        "formal_scripts": scripts,
        "formal_hooks": hooks,
        "legacy_online_training": legacy,
        "offline_training_contract": training_contract,
        "gad_on_policy_contract": gad_contract,
        "network_targets": network_targets,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit zero-tool-environment interaction in formal Drug Agent training")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = audit()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
