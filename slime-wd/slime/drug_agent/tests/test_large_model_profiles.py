from __future__ import annotations

import os
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class LargeModelProfileTest(unittest.TestCase):
    def run_validator(self, profile: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "MODEL_PROFILE": profile,
                "VALIDATE_LARGE_PROFILE_DATA": "0",
                **overrides,
            }
        )
        return subprocess.run(
            ["bash", "drug_agent/scripts/validate_qwen3_large_profile.sh"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )

    def test_all_large_profiles_are_statically_consistent(self) -> None:
        for profile in (
            "qwen35-27b-4xh200",
            "qwen36-35b-4xh200",
            "qwen35-122b-8xh200",
        ):
            with self.subTest(profile=profile):
                result = self.run_validator(profile)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("PASS: large model profile is statically consistent", result.stdout)

    def test_invalid_expert_grid_is_rejected(self) -> None:
        result = self.run_validator("qwen36-35b-4xh200", EXPERT_MODEL_PARALLEL_SIZE="3")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ETP*EP*PP", result.stdout)

    def test_fp32_accumulation_must_match_main_gradient_dtype(self) -> None:
        result = self.run_validator("qwen35-27b-4xh200", MAIN_GRADS_DTYPE="bf16")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("FP32 accumulation flag", result.stdout)

    def test_cpu_offload_does_not_promise_bf16_adam_moments(self) -> None:
        result = self.run_validator(
            "qwen35-122b-8xh200",
            QWEN122_OPTIMIZER_CPU_OFFLOAD="1",
            OFFLOAD_OPTIMIZER_STATES="0",
            FP8_PARAM_GATHER="0",
            EXP_AVG_DTYPE="bf16",
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("CPUAdam path", result.stdout)

    def test_profiles_avoid_context_parallel_optimizer_replication(self) -> None:
        expected = {
            "qwen35-27b-4xh200": '"pp": 2',
            "qwen36-35b-4xh200": '"pp": 2',
            "qwen35-122b-8xh200": '"pp": 4',
        }
        for profile, marker in expected.items():
            with self.subTest(profile=profile):
                result = self.run_validator(profile)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn('"cp": 1', result.stdout)
                self.assertIn(marker, result.stdout)

    def test_online_gad_has_a_method_specific_host_memory_gate(self) -> None:
        result = self.run_validator("qwen35-27b-4xh200")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('"gad_minimum_host_memory_gib": 600', result.stdout)
        probe = (ROOT / "drug_agent/scripts/run_qwen3_large_probe.sh").read_text()
        self.assertIn('worker_preflight "$GAD_MIN_HOST_MEMORY_GIB"', probe)

    def test_122b_uses_fp8_fused_adam_state_offload_profile(self) -> None:
        result = self.run_validator("qwen35-122b-8xh200")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('"fp8_param_gather": true', result.stdout)
        self.assertIn('"fp8_recipe": "delayed"', result.stdout)
        self.assertIn('"state_offload": true', result.stdout)
        self.assertIn('"main_grad": "bf16"', result.stdout)
        self.assertIn('"exp_avg": "fp8"', result.stdout)
        launcher = (ROOT / "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh").read_text()
        self.assertIn("--offload-optimizer-states", launcher)
        self.assertIn("--fp8-param-gather", launcher)

    def test_122b_rejects_blockwise_fp8_with_fp16_main_params(self) -> None:
        result = self.run_validator("qwen35-122b-8xh200", FP8_RECIPE="blockwise")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("BlockwiseQTensor", result.stdout)

    def test_slime_training_loop_handles_optimizer_state_offload(self) -> None:
        model_py = (ROOT / "slime/backends/megatron_utils/model.py").read_text()
        self.assertIn("reload_offloaded_states", model_py)
        self.assertIn("release_offloaded_gpu_states", model_py)
        patch_py = (ROOT / "slime/backends/megatron_utils/fp8_optimizer_state_offload.py").read_text()
        self.assertIn('raw.dtype == torch.uint8', patch_py)
        self.assertIn('OptimizerStateOffloader._reload_states = _reload_states', patch_py)
        self.assertIn('split_fp8_optimizer_param_groups(optimizer)', model_py)
        self.assertIn('_DEFAULT_MAX_GROUP_NUMEL = 64 * 1024 * 1024', patch_py)
        self.assertIn('_slime_fp8_fragmentation_patch', patch_py)
        self.assertIn('_step_with_streamed_moment_offload', patch_py)
        self.assertIn('pin_memory=False', patch_py)
        self.assertIn('use_pin_memory = False', patch_py)
        self.assertIn('bind_fp8_optimizer_state_offload(optimizer)', model_py)
        self.assertIn('OFFLOAD_OPTIMIZER_MASTER_WEIGHTS', model_py)

    def test_launchers_use_independent_expert_grid_and_physical_node_size(self) -> None:
        launchers = (
            "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh",
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        )
        for relative in launchers:
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("EXPERT_MODEL_SIZE", text)
                self.assertIn("--num-gpus-per-node", text)
                self.assertNotIn("EXPERT_DOMAIN_SIZE", text)

    def test_sft_chunks_full_vocab_log_probs_for_long_contexts(self) -> None:
        text = (ROOT / "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh").read_text()
        self.assertIn('LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}', text)
        self.assertIn('--log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE"', text)
        self.assertIn("--recompute-loss-function", text)

    def test_probe_driver_cannot_start_a_full_training_run(self) -> None:
        text = (ROOT / "drug_agent/scripts/run_qwen3_large_probe.sh").read_text()
        self.assertIn("sft-one-step", text)
        self.assertIn("toolrl-one-group", text)
        self.assertIn("gad-one-group", text)
        self.assertNotIn("full-epoch)", text)
        self.assertNotIn("full-training)", text)
        self.assertIn('DISABLE_CHECKPOINT_SAVE="${PROBE_DISABLE_CHECKPOINT_SAVE:-1}"', text)

    def test_colocated_rl_can_keep_large_actor_resident(self) -> None:
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("COLOCATE_OFFLOAD_TRAIN", text)
                self.assertIn("--no-offload-train", text)
                self.assertIn("--offload-rollout", text)

    def test_rl_launchers_propagate_low_memory_122b_optimizer_flags(self) -> None:
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("--offload-optimizer-states", text)
                self.assertIn("--fp8-param-gather", text)

    def test_large_launchers_guard_destructive_ray_restart(self) -> None:
        guard = (ROOT / "drug_agent/scripts/guard_ray_restart.sh").read_text()
        self.assertIn("DRUG_AGENT_ALLOW_RAY_RESTART_WITH_ACTIVE_JOBS", guard)
        self.assertIn('str(job.status) in {"RUNNING", "PENDING"}', guard)
        for relative in (
            "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh",
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("guard_ray_restart.sh", text)


if __name__ == "__main__":
    unittest.main()
