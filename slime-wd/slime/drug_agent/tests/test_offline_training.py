import os
from pathlib import Path
import unittest
from unittest.mock import patch

from drug_agent.offline_guard import assert_offline_training_environment, assert_tool_environment_allowed
from drug_agent.tools_debug.audit_offline_training import audit


class TestOfflineTrainingBoundary(unittest.TestCase):
    def test_formal_training_static_audit(self):
        report = audit()
        self.assertTrue(report["ok"], report["findings"])
        self.assertTrue(all(report["gad_on_policy_contract"].values()))

    def test_tool_environment_fails_closed_without_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                assert_tool_environment_allowed("test MCP access")

    def test_executor_fails_before_creating_runtime(self):
        from drug_agent.tools.tool_executor import MCPToolExecutor

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                MCPToolExecutor(connect_on_init=False)

    def test_offline_training_overrides_online_opt_in(self):
        with patch.dict(
            os.environ,
            {"DRUG_AGENT_TRAINING_OFFLINE": "1", "DRUG_AGENT_ALLOW_TOOL_ENV": "1"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                assert_tool_environment_allowed("test MCP access")

    def test_offline_training_has_no_molclaw_credentials(self):
        with patch.dict(
            os.environ,
            {"DRUG_AGENT_TRAINING_OFFLINE": "1", "DRUG_AGENT_ALLOW_TOOL_ENV": "0"},
            clear=True,
        ):
            assert_offline_training_environment()

    def test_formal_grpo_and_gad_checkpoint_guards_are_explicit(self):
        root = Path(__file__).resolve().parents[1]
        toolrl = (root / "toolrl/scripts/run_toolrl_grpo.sh").read_text(encoding="utf-8")
        self.assertIn("N_SAMPLES_PER_PROMPT:-2", toolrl)
        self.assertIn("requires N_SAMPLES_PER_PROMPT >= 2", toolrl)
        gad = (root / "gad/scripts/run_stage3_gad_grpo.sh").read_text(encoding="utf-8")
        for name in ("STUDENT_WARMUP_LOAD", "DISCRIMINATOR_WARMUP_LOAD", "GAD_WARMUP_MANIFEST"):
            self.assertIn(f"${{{name}:?", gad)
        self.assertNotIn("STUDENT_LOAD=${STUDENT_LOAD:-$REF_LOAD}", gad)


if __name__ == "__main__":
    unittest.main()
