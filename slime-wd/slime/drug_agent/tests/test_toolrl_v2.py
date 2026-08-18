from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drug_agent.toolrl.policy_boundary import audit_all_groups, policy_boundary_filter


class _Sample:
    def __init__(self, reward: float, reward_stage: str | None = None):
        self.reward = (
            {"score": reward, "diagnostics": {"reward_stage": reward_stage}}
            if reward_stage is not None else reward
        )
        self.metadata = {}
        self.group_index = 0
        self.index = 0

    def get_reward_value(self, args):
        return self.reward["score"] if isinstance(self.reward, dict) else self.reward


def test_policy_boundary_filter_uses_current_group_variance():
    args = SimpleNamespace(use_kl_loss=False)
    assert bool(policy_boundary_filter(args, [_Sample(-0.4), _Sample(0.9), _Sample(-0.4), _Sample(0.9)]).keep)
    mastered = policy_boundary_filter(args, [_Sample(1.0) for _ in range(4)])
    too_hard = policy_boundary_filter(args, [_Sample(-0.5) for _ in range(4)])
    assert not bool(mastered.keep) and mastered.reason == "mastered_all_correct"
    assert not bool(too_hard.keep) and too_hard.reason == "too_hard_all_wrong"


def test_policy_boundary_has_no_selective_format_recovery_kl_exception():
    samples = [_Sample(-0.5, "invalid_react_tool_envelope") for _ in range(4)]
    without_kl = policy_boundary_filter(SimpleNamespace(use_kl_loss=False), samples)
    with_kl = policy_boundary_filter(SimpleNamespace(use_kl_loss=True), samples)
    assert not bool(without_kl.keep) and without_kl.reason == "too_hard_all_wrong"
    assert not bool(with_kl.keep) and with_kl.reason == "too_hard_all_wrong"


def test_policy_boundary_ablation_does_not_keep_stable_wrong_tool_group_with_kl():
    samples = [_Sample(-0.4, "wrong_tool") for _ in range(4)]
    result = policy_boundary_filter(SimpleNamespace(use_kl_loss=True), samples)
    assert not bool(result.keep) and result.reason == "too_hard_all_wrong"


def test_group_audit_accepts_bound_get_samples_callback(tmp_path, monkeypatch):
    class _Source:
        sample_offset = 1

        def __len__(self):
            return 3

        def get_samples(self, count):
            raise AssertionError("audit must not fetch more data")

    output = tmp_path / "audit.jsonl"
    monkeypatch.setenv("TOOLRL_LEARNABILITY_LOG", str(output))
    samples = [_Sample(-0.4) for _ in range(4)]
    audit_all_groups(
        SimpleNamespace(),
        [samples],
        _Source().get_samples,
        accepted_groups=[samples],
        rollout_id=0,
    )
    row = output.read_text().strip()
    assert '"dataset_cursor":0' in row
    assert '"data_source_cursor_after_batch":1' in row


def test_v6_launcher_uses_fixed_epoch_and_symmetric_clip():
    script = (Path(__file__).parents[1] / "scripts" / "run_qwen3_5_9b_v6_turn_sft_toolrl.sh").read_text()
    assert "TOOLRL_ENABLE_DYNAMIC_FILTER=0" in script
    assert "TOOLRL_REQUIRE_EXACT_EPOCH=1" in script
    assert 'TOOLRL_NUM_ROLLOUT="$(( BASELINE_RECORDS / 4 ))"' in script
    assert 'TOOLRL_NUM_ROLLOUT="$(( PRODUCTION_RECORDS / 4 ))"' in script
    assert "EPS_CLIP=0.2 EPS_CLIP_HIGH=0.2" in script
    assert "TOOLRL_REWARD_MODE=toolrl_official_8cee13e" in script
    assert "TOOLRL_KL_COEF=0.001" in script
    assert "compute_official_8cee13e_advantages" in script


def test_v2_launcher_uses_sft_directly_and_online_selector():
    script = (Path(__file__).parents[1] / "scripts" / "run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh").read_text()
    assert "select_toolrl_decisions.py" in script
    assert "policy_boundary.policy_boundary_filter" in script
    assert "N_SAMPLES_PER_PROMPT=4" in script
    assert "DYNAMIC_SAMPLING_STRICT_MAX_DROPS=1" in script
    assert "COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=1" in script
    assert 'TOOLRL_REF_LOAD="${TOOLRL_REF_LOAD:-$BASE_SFT_DIR}"' in script
    assert 'USE_KL_LOSS="$TOOLRL_USE_KL_LOSS"' in script
    assert 'KL_COEF="$TOOLRL_KL_COEF"' in script
    assert "EXPECTED_TOOLRL_NUM_ROLLOUT" in script
    assert "toolrl_gate_near_limit_rbs4_offload" in script
    assert 'NUM_ROLLOUT="$TOOLRL_LONG_BATCH_GATE_UPDATES" ROLLOUT_BATCH_SIZE=4' in script
    assert "N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 SAVE_INTERVAL=1" in script
    assert 'TOOLRL_SAVE_INTERVAL="${TOOLRL_SAVE_INTERVAL:-25}"' in script
    assert 'TOOLRL_CHECKPOINT_KEEP_LAST="${TOOLRL_CHECKPOINT_KEEP_LAST:-4}"' in script
    assert 'TOOLRL_RETAIN_GATE_CHECKPOINTS="${TOOLRL_RETAIN_GATE_CHECKPOINTS:-0}"' in script
    assert 'rm -rf -- "$GATE_ROOT/near_limit_rbs4_offload"' in script
    assert 'TOOLRL_LOAD="$TOOLRL_DIR"' in script
    assert "TOOLRL_RESUME_FLAG=1" in script
    assert "materialize_decision_aware_toolrl_view.py" not in script
    assert "intermediate-budget" not in script
    assert 'LOAD="$BASE_SFT_DIR"' in script
    assert "build_plan_view.py" not in script
    assert "PLAN_SFT" not in script
    assert "plan_sft" not in script

    canonical = (Path(__file__).parents[1] / "scripts" / "run_qwen3_5_9b_v4_sft_toolrl_v2.sh").read_text()
    assert "run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh" in canonical


def test_v4_mol_serial_launcher_drains_sft_runtime_before_toolrl_exec():
    script = (Path(__file__).parents[1] / "scripts" / "run_qwen3_5_9b_v4_mol_sft_toolrl_v2.sh").read_text()
    drain = script.index("drain_previous_stage_runtime\n")
    toolrl_start = script.index('START toolrl_v2_pipeline')
    toolrl_exec = script.index("exec env", toolrl_start)

    assert drain < toolrl_start < toolrl_exec
    assert "guard_ray_restart.sh" in script
    assert "ray stop --force" in script
    assert "--query-compute-apps=pid" in script
    assert "SFT GPU processes did not exit within 180 seconds" in script
    assert "ALLOW_BUSY_GPUS=1" not in script


def test_v5_launcher_has_explicit_full_and_mol_contracts():
    script = (Path(__file__).parents[1] / "scripts" / "run_qwen3_5_9b_v5_sft_toolrl.sh").read_text()
    assert 'V5_VARIANT="${V5_VARIANT:-full}"' in script
    assert "live_tool_catalog_v5-sftnrl" in script
    assert "live_tool_catalog_v5-mol-sftnrl" in script
    assert "EXPECTED_CANONICAL_RECORDS=605" in script
    assert "EXPECTED_CANONICAL_RECORDS=365" in script
    assert "run_qwen3_5_9b_v4_mol_sft_toolrl_v2.sh" in script
