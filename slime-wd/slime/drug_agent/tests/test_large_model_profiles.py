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
            "qwen35-27b-8xh200",
            "qwen36-35b-4xh200",
            "qwen36-35b-8xh200",
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
            "qwen35-27b-8xh200": '"pp": 2',
            "qwen36-35b-4xh200": '"pp": 2',
            "qwen36-35b-8xh200": '"pp": 2',
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

    def test_122b_streamed_optimizer_uses_measured_expandable_allocator(self) -> None:
        result = self.run_validator("qwen35-122b-8xh200")
        self.assertEqual(result.returncode, 0, result.stdout)
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        self.assertIn(
            "QWEN122_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True",
            profile,
        )

    def test_122b_colocated_online_methods_fail_closed(self) -> None:
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        self.assertIn(
            "COLOCATED_TOOLRL_SUPPORTED=${COLOCATED_TOOLRL_SUPPORTED:-0}", profile
        )
        self.assertIn("COLOCATED_GAD_SUPPORTED=${COLOCATED_GAD_SUPPORTED:-0}", profile)
        probe = (ROOT / "drug_agent/scripts/run_qwen3_large_probe.sh").read_text()
        self.assertIn("require_colocated_online_support ToolRL", probe)
        self.assertIn("require_colocated_online_support GAD-negative-generation", probe)
        self.assertIn('ALLOW_UNSUPPORTED_COLOCATED_RL:-0', probe)

    def test_122b_uses_official_fp8_source_for_actor_and_rollout(self) -> None:
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        toolrl = (ROOT / "drug_agent/toolrl/scripts/run_toolrl_grpo.sh").read_text()
        gad = (ROOT / "drug_agent/gad/scripts/run_stage3_gad_grpo.sh").read_text()
        self.assertIn("Qwen3.5-122B-A10B-FP8", profile)
        self.assertIn("Qwen3.5-122B-A10B-FP8_torch_dist", profile)
        self.assertIn("ROLLOUT_HF_CHECKPOINT=${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}", profile)
        self.assertNotIn("Qwen3.5-122B-A10B_torch_dist", profile)
        self.assertIn("SGLANG_DISABLE_CUSTOM_ALL_REDUCE", profile)
        self.assertIn("TRAIN_MEMORY_MARGIN_BYTES=${TRAIN_MEMORY_MARGIN_BYTES:-536870912}", profile)
        self.assertIn("SLIME_FP8_OPTIMIZER_MAX_GROUP_NUMEL", profile)
        self.assertIn('--train-memory-margin-bytes "$TRAIN_MEMORY_MARGIN_BYTES"', toolrl)
        self.assertIn('--train-memory-margin-bytes "$TRAIN_MEMORY_MARGIN_BYTES"', gad)
        self.assertIn("SGLANG_KV_CACHE_DTYPE", profile)
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("ROLLOUT_HF_CHECKPOINT", text)
                self.assertIn("--sglang-kv-cache-dtype", text)

    def test_8gpu_profiles_expose_distributed_optimizer_overlap(self) -> None:
        for profile in ("qwen35-27b-8xh200", "qwen36-35b-8xh200"):
            with self.subTest(profile=profile):
                result = self.run_validator(profile)
                self.assertEqual(result.returncode, 0, result.stdout)
        profile_text = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        self.assertIn("OVERLAP_GRAD_REDUCE", profile_text)
        self.assertIn("OVERLAP_PARAM_GATHER", profile_text)

    def test_8gpu_profiles_balance_the_heavier_loss_pipeline_rank(self) -> None:
        profile_text = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        self.assertIn("NUM_LAYERS_IN_FIRST_PIPELINE_STAGE", profile_text)
        self.assertIn("NUM_LAYERS_IN_LAST_PIPELINE_STAGE", profile_text)
        for relative in (
            "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh",
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("--decoder-first-pipeline-num-layers", text)
                self.assertIn("--decoder-last-pipeline-num-layers", text)

    def test_122b_long_sft_balances_lm_head_stage_and_bounds_logits_chunk(self) -> None:
        result = self.run_validator("qwen35-122b-8xh200")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('"first_stage_layers": 12', result.stdout)
        self.assertIn('"last_stage_layers": 12', result.stdout)
        self.assertIn('"pipeline_layout": "Et*12|t*12|t*12|t*12L"', result.stdout)
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        self.assertIn("RECOMPUTE_VOCAB_LOG_PROBS", profile)
        self.assertIn("QWEN122_LOG_PROBS_CHUNK_SIZE:-512", profile)
        loss = (ROOT / "slime/backends/megatron_utils/loss.py").read_text()
        self.assertIn('getattr(args, "recompute_vocab_log_probs", False)', loss)
        qwen_spec = (ROOT / "slime_plugins/models/qwen3_5.py").read_text()
        self.assertNotIn('assert config.pipeline_model_parallel_layout is None', qwen_spec)
        model = (ROOT / "slime/backends/megatron_utils/model.py").read_text()
        self.assertIn('forward_kwargs["fp32_output"] = False', model)
        ppo = (ROOT / "slime/utils/ppo_utils.py").read_text()
        self.assertIn("class _RecomputedVocabParallelLogProbs", ppo)

    def test_formal_27b_serial_run_uses_long_sequence_topology_in_every_actor_stage(self) -> None:
        serial = (
            ROOT / "drug_agent/scripts/run_qwen3_large_training_serial.sh"
        ).read_text()
        self.assertIn("FORMAL_PP=4", serial)
        self.assertIn("FORMAL_FIRST=20", serial)
        self.assertIn("FORMAL_LAST=12", serial)
        self.assertIn('"RECOMPUTE_VOCAB_LOG_PROBS=1"', serial)
        self.assertIn('"OVERLAP_CPU_OPTIMIZER_D2H_H2D=0"', serial)
        self.assertIn('"TOOLRL_OPTIMIZER_CPU_OFFLOAD=0"', serial)
        self.assertIn('"EXP_AVG_DTYPE=bf16"', serial)
        self.assertEqual(serial.count("USE_PRECISION_AWARE_OPTIMIZER=0"), 2)
        self.assertIn("SGLANG_MEM_FRACTION_STATIC=0.12", serial)
        self.assertIn('"DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS=8"', serial)
        self.assertEqual(serial.count('env "${FORMAL_PARALLEL_ENV[@]}"'), 4)
        self.assertEqual(serial.count("USE_ROLLOUT_LOGPROBS=1"), 2)
        # ToolRL uses REINFORCE++ across independent prompts, so only the
        # grouped GAD branch needs the non-zero reward-std filter.
        self.assertEqual(serial.count("check_reward_nonzero_std"), 1)
        self.assertIn('toolrl) require_path "$TOOLRL_DIR/latest_checkpointed_iteration.txt"', serial)
        self.assertIn('"MAX_TOKENS_PER_GPU=6144"', serial)

    def test_formal_serial_run_supports_gad_only_v2_with_complete_negative_cache_gate(self) -> None:
        serial = (
            ROOT / "drug_agent/scripts/run_qwen3_large_training_serial.sh"
        ).read_text()
        self.assertIn("RUN_GAD_ONLY=${RUN_GAD_ONLY:-0}", serial)
        self.assertIn("SFT_DIR=${SFT_CHECKPOINT_DIR:-$RUN_ROOT/sft}", serial)
        self.assertIn("EXPECTED_SFT_RECORDS=${EXPECTED_SFT_RECORDS:-364}", serial)
        self.assertIn("EXPECTED_TOOLRL_RECORDS=${EXPECTED_TOOLRL_RECORDS:-3182}", serial)
        self.assertIn("EXPECTED_GAD_RECORDS=${EXPECTED_GAD_RECORDS:-3147}", serial)
        self.assertIn("NEGATIVE_ROWS == GAD_COUNT", serial)
        self.assertIn("validate_negative_cache", serial)
        self.assertIn("collections.Counter(actual) != collections.Counter(expected)", serial)
        self.assertIn("FORMAL_GAD_MAX_PROMPT_LEN=${FORMAL_GAD_MAX_PROMPT_LEN:-98304}", serial)
        self.assertIn("FORMAL_GAD_MAX_CONTEXT_LEN=${FORMAL_GAD_MAX_CONTEXT_LEN:-102400}", serial)

    def test_122b_external_rollout_disables_unsafe_overlap_schedule(self) -> None:
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        server = (
            ROOT / "drug_agent/scripts/serve_qwen122_fp8_external_rollout.sh"
        ).read_text()
        self.assertIn("SGLANG_DISABLE_OVERLAP_SCHEDULE", profile)
        self.assertIn("--disable-overlap-schedule", server)
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("SGLANG_DISABLE_OVERLAP_SCHEDULE", text)
                self.assertIn("--sglang-disable-overlap-schedule", text)

    def test_online_rl_launchers_support_behavior_logprobs_and_dynamic_filtering(self) -> None:
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("DYNAMIC_SAMPLING_FILTER_PATH", text)
                self.assertIn("--dynamic-sampling-filter-path", text)
                self.assertIn("USE_ROLLOUT_LOGPROBS", text)
                self.assertIn("--use-rollout-logprobs", text)
                self.assertIn("--dynamic-sampling-max-dropped-groups", text)

    def test_gad_launcher_exposes_measured_grouped_rollout_router_policy(self) -> None:
        launcher = (ROOT / "drug_agent/gad/scripts/run_stage3_gad_grpo.sh").read_text()
        self.assertIn("ROUTER_POLICY=${ROUTER_POLICY:-}", launcher)
        self.assertIn('SGLANG_EXTRA_ARGS+=(--router-policy "$ROUTER_POLICY")', launcher)

    def test_online_rl_launchers_expose_length_aware_generation(self) -> None:
        arguments = (ROOT / "slime/utils/arguments.py").read_text()
        self.assertIn('"--rollout-long-response-len"', arguments)
        self.assertIn('"--rollout-long-task-types"', arguments)
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("CUSTOM_GENERATE_FUNCTION_PATH", text)
                self.assertIn("--custom-generate-function-path", text)
                self.assertIn("--rollout-long-response-len", text)
                self.assertIn("--rollout-long-task-types", text)

    def test_9b_v2_serial_is_size_independent_and_uses_8k_16k_tiers(self) -> None:
        serial = (
            ROOT / "drug_agent/scripts/run_qwen3_5_9b_sft_toolrl_gad_serial.sh"
        ).read_text()
        self.assertIn("live_tool_catalog_v2", serial)
        self.assertNotIn("Expected 373 canonical records", serial)
        self.assertNotIn("NUM_ROLLOUT=757", serial)
        self.assertIn("TOOLRL_NUM_ROLLOUT=$((TOOLRL_EPOCHS * TOOLRL_COUNT / 8))", serial)
        self.assertIn("GAD_NUM_ROLLOUT=$((GAD_EPOCHS * GAD_COUNT))", serial)
        self.assertIn("ROLLOUT_MAX_RESPONSE_LEN=8192", serial)
        self.assertIn("ROLLOUT_LONG_RESPONSE_LEN=16384", serial)
        self.assertIn("ROLLOUT_MAX_CONTEXT_LEN=131072", serial)
        self.assertIn("ROLLOUT_LONG_TASK_TYPES='vs pf'", serial)
        self.assertIn("PIPELINE_MODEL_PARALLEL_SIZE=1", serial)
        self.assertIn("GLOBAL_BATCH_SIZE=2 MAX_TOKENS_PER_GPU=16384", serial)
        self.assertIn("ADVANTAGE_ESTIMATOR=reinforce_plus_plus", serial)
        self.assertIn("ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=8", serial)
        self.assertIn("ROUTER_POLICY=round_robin", serial)

    def test_toolrl_launcher_exposes_policy_stability_controls(self) -> None:
        launcher = (ROOT / "drug_agent/toolrl/scripts/run_toolrl_grpo.sh").read_text()
        serial = (ROOT / "drug_agent/scripts/run_qwen3_large_training_serial.sh").read_text()
        self.assertIn("LR_WARMUP_FRACTION", launcher)
        self.assertIn("--lr-warmup-fraction", launcher)
        self.assertIn("USE_KL_LOSS", launcher)
        self.assertIn("--use-kl-loss", launcher)
        self.assertIn("FORMAL_TOOLRL_LR=${FORMAL_TOOLRL_LR:-1e-8}", serial)
        self.assertIn("FORMAL_USE_KL_LOSS=${FORMAL_USE_KL_LOSS:-0}", serial)
        self.assertIn("LR_WARMUP_FRACTION=0.05", serial)
        self.assertIn("ROLLOUT_TEMPERATURE=0.7", serial)
        self.assertEqual(serial.count("ROLLOUT_MAX_PROMPT_LEN=65536"), 1)
        self.assertEqual(serial.count("ROLLOUT_MAX_CONTEXT_LEN=69632"), 1)
        self.assertEqual(serial.count('ROLLOUT_MAX_PROMPT_LEN="$FORMAL_GAD_MAX_PROMPT_LEN"'), 2)
        self.assertEqual(serial.count('ROLLOUT_MAX_CONTEXT_LEN="$FORMAL_GAD_MAX_CONTEXT_LEN"'), 2)

    def test_resident_colocated_rl_preserves_expandable_allocator(self) -> None:
        for relative in (
            "drug_agent/toolrl/scripts/run_toolrl_grpo.sh",
            "drug_agent/gad/scripts/generate_stage2_negatives.sh",
            "drug_agent/gad/scripts/run_stage3_gad_grpo.sh",
        ):
            with self.subTest(launcher=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("USES_COLOCATED_MEMORY_SAVER", text)
                self.assertIn("COLOCATE_OFFLOAD_ROLLOUT", text)
                self.assertIn("COLOCATE_OFFLOAD_TRAIN", text)
        rollout = (ROOT / "slime/ray/rollout.py").read_text()
        self.assertIn('env_vars["PYTORCH_CUDA_ALLOC_CONF"] = ""', rollout)
        self.assertIn('env_vars["PYTORCH_ALLOC_CONF"] = ""', rollout)
        external = (
            ROOT / "drug_agent/scripts/serve_qwen122_fp8_external_rollout.sh"
        ).read_text()
        self.assertIn("unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF", external)

    def test_8gpu_sft_can_use_precision_aware_moments_without_changing_cpuadam_rl(self) -> None:
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        launcher = (ROOT / "drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh").read_text()
        self.assertIn("SFT_EXP_AVG_DTYPE", profile)
        self.assertIn("SFT_EXP_AVG_SQ_DTYPE", profile)
        self.assertIn("SFT_EXP_AVG_DTYPE", launcher)
        self.assertIn("SFT_EXP_AVG_SQ_DTYPE", launcher)

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
        self.assertIn('SLIME_FP8_OPTIMIZER_MAX_GROUP_NUMEL', patch_py)
        self.assertIn('64 * 1024 * 1024', patch_py)
        self.assertIn('_slime_fp8_fragmentation_patch', patch_py)
        self.assertIn('fp8_state = quantizer.make_empty(param.shape)', patch_py)
        self.assertIn('_storage_tensor(fp8_state).zero_()', patch_py)
        self.assertIn('_initialize_all_state_without_fp32_master_transient', patch_py)
        self.assertIn('param.dequantize(dtype=master_dtype).detach()', patch_py)
        self.assertIn('FusedAdam.initialize_state =', patch_py)
        self.assertIn('_apply_scale_with_reallocated_destination', patch_py)
        self.assertIn('_slime_low_peak_unscale_active', patch_py)
        self.assertIn('unscaled_state.untyped_storage().resize_(0)', patch_py)
        self.assertIn('torch.cuda.empty_cache()', model_py)
        self.assertIn('_step_with_streamed_moment_offload', patch_py)
        self.assertIn('pin_memory=False', patch_py)
        self.assertIn('use_pin_memory = False', patch_py)
        self.assertIn('bind_fp8_optimizer_state_offload(optimizer)', model_py)
        self.assertIn('OFFLOAD_OPTIMIZER_MASTER_WEIGHTS', model_py)
        self.assertIn('_drop_optimizer_cpu_state_before_weights_only_save', model_py)
        self.assertIn('SLIME_DROP_OPTIMIZER_STATE_BEFORE_WEIGHTS_ONLY_SAVE', model_py)
        serial = (ROOT / "drug_agent/scripts/run_qwen3_large_training_serial.sh").read_text()
        self.assertIn('SLIME_DROP_OPTIMIZER_STATE_BEFORE_WEIGHTS_ONLY_SAVE', serial)

    def test_pinned_hybrid_optimizer_waits_for_h2d_parameter_copies(self) -> None:
        model = (ROOT / "slime/backends/megatron_utils/model.py").read_text()
        patch = (
            ROOT / "slime/backends/megatron_utils/hybrid_optimizer_stream_patch.py"
        ).read_text()
        self.assertIn("install_hybrid_optimizer_h2d_wait_patch()", model)
        self.assertIn("self._h2d_stream.record_event().wait(current_stream)", patch)
        self.assertNotIn("self._d2h_stream.record_event().wait(current_stream)", patch)

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
                self.assertIn("COLOCATE_OFFLOAD_ROLLOUT", text)
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

    def test_fp8_converter_expands_qwen35_fused_experts(self) -> None:
        converter = (ROOT / "tools/convert_hf_to_fp8.py").read_text()
        self.assertIn("quantize_qwen35_fused_experts", converter)
        self.assertIn('gate_up_suffix = ".experts.gate_up_proj"', converter)
        self.assertIn('down_suffix = ".experts.down_proj"', converter)
        self.assertIn('f"{prefix}.experts.{expert_index}.{projection}.weight"', converter)
        self.assertIn('f"{output_key}_scale_inv"', converter)
        self.assertIn("v.numel() * v.element_size()", converter)
        self.assertIn("block_max.clamp(min=1e-12)", converter)
        self.assertIn('choices=["float32", "bfloat16"]', converter)

    def test_qwen35_bridge_imports_official_fp8_weights_safely(self) -> None:
        bridge = (ROOT / "slime_plugins/mbridge/qwen3_5.py").read_text()
        self.assertIn("Qwen35DequantFP8SafeTensorIO", bridge)
        self.assertIn('f"cuda:{torch.cuda.current_device()}"', bridge)
        self.assertIn('quant_method == "fp8"', bridge)
        self.assertIn("_weight_name_mapping_fp8_expert", bridge)
        self.assertIn('f"{hf_prefix}.gate_proj.weight"', bridge)
        self.assertIn('f"{hf_prefix}.up_proj.weight"', bridge)
        self.assertIn('f"{hf_prefix}.down_proj.weight"', bridge)

    def test_122b_moments_only_offload_applies_to_colocated_rl(self) -> None:
        profile = (ROOT / "drug_agent/scripts/qwen3_large_profile.sh").read_text()
        self.assertIn(
            "OFFLOAD_OPTIMIZER_MASTER_WEIGHTS=${OFFLOAD_OPTIMIZER_MASTER_WEIGHTS:-0}",
            profile,
        )

    def test_122b_lora_rl_uses_measured_blockwise_standard_ppo_profile(self) -> None:
        launcher = (ROOT / "drug_agent/scripts/run_qwen35_122b_lora_rl_serial.sh").read_text()
        self.assertIn("QWEN122_LORA_FP8_RECIPE:-blockwise", launcher)
        self.assertIn("QWEN122_MAX_TOKENS_PER_GPU:-10240", launcher)
        self.assertIn("USE_ROLLOUT_LOGPROBS=0", launcher)

    def test_lora_reference_policy_disables_adapter_without_full_checkpoint_backup(self) -> None:
        actor = (ROOT / "slime/backends/megatron_utils/actor.py").read_text()
        self.assertIn("_lora_reference_via_disabled_adapter", actor)
        self.assertIn("with peft.disable_adapter(self.model):", actor)
        self.assertIn('store_prefix="ref_"', actor)

    def test_gad_discriminator_has_a_cpu_capacity_fallback(self) -> None:
        service = (ROOT / "drug_agent/gad/service.py").read_text()
        launcher = (ROOT / "drug_agent/gad/scripts/serve_discriminator.sh").read_text()
        self.assertIn('parser.add_argument(\n        "--device"', service)
        self.assertIn("device=args.device", service)
        self.assertIn('GAD_DISCRIMINATOR_DEVICE:-cuda', launcher)

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
