from pathlib import Path

import pytest

from drug_agent.scripts.check_rl_training_gate import check_log


def test_v4_launcher_pins_decision_aware_algorithm_and_excludes_gad():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_qwen3_5_9b_v4_sft_toolrl_official.sh").read_text(encoding="utf-8")
    for expected in (
        "ADVANTAGE_ESTIMATOR=grpo",
        "ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16",
        "TOOLRL_REWARD_MODE=decision_aware",
        "NORMALIZE_ADVANTAGES=0",
        "USE_KL_LOSS=0",
        "ROLLOUT_MAX_RESPONSE_LEN=16384",
        "TOOLRL_NUM_ROLLOUT=1259",
        "ROLLOUT_MAX_PROMPT_LEN=245760",
        "ROLLOUT_MAX_CONTEXT_LEN=262144",
        "materialize_decision_aware_toolrl_view.py",
        "build_toolrl_length_probes.py",
        "for length_gate in shortest p50 p95 near_limit",
        "TOOLRL_GATE_CANDIDATES=4",
        "TOOLRL_GATE_MIN_NONZERO_GROUP_RATIO=0.25",
        'NUM_ROLLOUT="$TOOLRL_GATE_CANDIDATES" ROLLOUT_BATCH_SIZE=1',
        "CUSTOM_ROLLOUT_LOG_FUNCTION_PATH=drug_agent.toolrl.metrics.augment_rollout_metrics",
        "TOOLRL_DISTRIBUTED_TIMEOUT_MINUTES=60",
        "TOOLRL_SAVE_INTERVAL=100",
    ):
        assert expected in text
    assert "reinforce_plus_plus" not in text
    assert "TOOLRL_REWARD_MODE=molclaw" not in text
    assert "run_stage3_gad" not in text


def test_rl_gate_ignores_sglang_config_fields_but_rejects_runtime_nccl(tmp_path):
    successful = """
ServerArgs(enable_nccl_nvls=False, watchdog_timeout=300, error_log=None)
rollout 0: {'rollout/raw_reward': 2.25, 'rollout/truncated': 0.25}
step 0: {'train/loss': 1e-8, 'train/grad_norm': 0.7}
"""
    log = tmp_path / "gate.log"
    log.write_text(successful, encoding="utf-8")
    assert check_log(log, 1)["gate"] == "PASS"

    log.write_text(successful + "\nNCCL WARN communicator timed out\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime failure"):
        check_log(log, 1)


def test_rl_gate_requires_a_minimum_fraction_of_reward_variance_groups(tmp_path):
    log = tmp_path / "variance.log"
    log.write_text(
        "\n".join(
            f"rollout {index}: {{'rollout/raw_reward': -0.5, 'rollout/truncated': 0.0, "
            f"'decision_role/tool_step/nonzero_std_group_ratio': {ratio}}}\n"
            f"step {index}: {{'train/loss': {0.1 if ratio else 0.0}, 'train/grad_norm': {ratio}}}"
            for index, ratio in enumerate((0.0, 0.0, 1.0, 0.0))
        ),
        encoding="utf-8",
    )
    assert check_log(log, 4, minimum_nonzero_group_ratio=0.25)["gate"] == "PASS"
    with pytest.raises(ValueError, match="nonzero reward-group ratio"):
        check_log(log, 4, minimum_nonzero_group_ratio=0.26)


def test_rl_gate_can_accept_finite_zero_variance_probe_without_weakening_default(tmp_path):
    log = tmp_path / "zero_variance.log"
    log.write_text(
        "rollout 0: {'rollout/raw_reward': -0.4, 'rollout/truncated': 0.0}\n"
        "step 0: {'train/loss': 0.0, 'train/grad_norm': 0.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="all-zero gradients"):
        check_log(log, 1)
    assert check_log(log, 1, allow_all_zero_gradients=True)["gate"] == "PASS"
