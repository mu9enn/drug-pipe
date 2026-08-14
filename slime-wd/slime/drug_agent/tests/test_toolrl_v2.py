from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drug_agent.toolrl.policy_boundary import policy_boundary_filter


class _Sample:
    def __init__(self, reward: float):
        self.reward = reward

    def get_reward_value(self, args):
        return self.reward


def test_policy_boundary_filter_uses_current_group_variance():
    args = SimpleNamespace()
    assert bool(policy_boundary_filter(args, [_Sample(-0.4), _Sample(0.9), _Sample(-0.4), _Sample(0.9)]).keep)
    mastered = policy_boundary_filter(args, [_Sample(1.0) for _ in range(4)])
    too_hard = policy_boundary_filter(args, [_Sample(-0.5) for _ in range(4)])
    assert not bool(mastered.keep) and mastered.reason == "mastered_all_correct"
    assert not bool(too_hard.keep) and too_hard.reason == "too_hard_all_wrong"


def test_v2_launcher_uses_sft_directly_and_online_selector():
    script = (Path(__file__).parents[1] / "scripts" / "run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh").read_text()
    assert "select_toolrl_decisions.py" in script
    assert "policy_boundary.policy_boundary_filter" in script
    assert "N_SAMPLES_PER_PROMPT=4" in script
    assert "DYNAMIC_SAMPLING_STRICT_MAX_DROPS=1" in script
    assert "COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=1" in script
    assert "toolrl_gate_near_limit_rbs4_offload" in script
    assert 'NUM_ROLLOUT="$TOOLRL_LONG_BATCH_GATE_UPDATES" ROLLOUT_BATCH_SIZE=4' in script
    assert "N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 SAVE_INTERVAL=1" in script
    assert 'TOOLRL_SAVE_INTERVAL="${TOOLRL_SAVE_INTERVAL:-25}"' in script
    assert 'TOOLRL_CHECKPOINT_KEEP_LAST="${TOOLRL_CHECKPOINT_KEEP_LAST:-4}"' in script
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
