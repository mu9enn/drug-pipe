import os
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


if __name__ == "__main__":
    unittest.main()
