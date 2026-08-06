from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.claude_agent.run_claude import _validate_kg_worker_capacity
from pipeline.kg.tool_admission import (
    expected_tools_from_task_spec,
    first_admissible_index,
    load_tool_limits,
    serial_tool_claims,
)


class ToolAdmissionTest(unittest.TestCase):
    def test_kg_worker_capacity_requires_matching_gate_for_three_or_four(self) -> None:
        _validate_kg_worker_capacity(2)
        with self.assertRaisesRegex(ValueError, "must be <= 4"):
            _validate_kg_worker_capacity(5)
        with patch.dict(
            "os.environ",
            {
                "CLAUDE_GATE_MAX_CONCURRENCY": "4",
                "CLAUDE_GATE_DATA_PIPE_MAX_CONCURRENCY": "2",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "exceeds Claude gate"):
                _validate_kg_worker_capacity(3)
        with patch.dict(
            "os.environ",
            {
                "CLAUDE_GATE_MAX_CONCURRENCY": "4",
                "CLAUDE_GATE_DATA_PIPE_MAX_CONCURRENCY": "4",
            },
            clear=False,
        ):
            _validate_kg_worker_capacity(4)

    def test_policy_contains_all_current_molclaw_tools(self) -> None:
        limits = load_tool_limits()
        self.assertEqual(len(limits), 81)
        self.assertEqual(sum(limit == 4 for limit in limits.values()), 44)
        self.assertEqual(sum(limit == 30 for limit in limits.values()), 37)
        promoted = {
            "residue_mapper",
            "equiscore_pocket",
            "get_pepinvent_info",
            "pulchura_rebuild",
            "fpocket_toolkit",
        }
        self.assertTrue(all(limits[tool] == 30 for tool in promoted))
        self.assertEqual(serial_tool_claims(tuple(promoted), limits), frozenset())

    def test_expected_tools_use_toolchain_then_trajectory_fallback(self) -> None:
        self.assertEqual(
            expected_tools_from_task_spec(
                {"toolchain": {"tools": ["foldx_tool", "is_valid_smiles", "foldx_tool"]}}
            ),
            ("foldx_tool", "is_valid_smiles"),
        )
        self.assertEqual(
            expected_tools_from_task_spec(
                {
                    "expected_trajectory": {
                        "execution_plan": {"tool_order": ["pred_binding_affinity_boltz2"]}
                    }
                }
            ),
            ("pred_binding_affinity_boltz2",),
        )

    def test_only_limit_four_tools_become_serial_claims(self) -> None:
        limits = load_tool_limits()
        claims = serial_tool_claims(
            (
                "foldx_tool",
                "is_valid_smiles",
                "server_file_to_base64",
                "fpocket_toolkit",
                "residue_mapper",
            ),
            limits,
        )
        self.assertEqual(claims, frozenset({"foldx_tool"}))
        with self.assertRaisesRegex(ValueError, "unregistered"):
            serial_tool_claims(("new_unknown_tool",), limits)

    def test_admission_blocks_same_compute_tool_but_not_light_or_different_tools(self) -> None:
        active = [frozenset({"foldx_tool"})]
        pending = [
            frozenset({"foldx_tool"}),
            frozenset({"pred_binding_affinity_boltz2"}),
            frozenset(),
        ]
        self.assertEqual(first_admissible_index(pending, active), 1)
        self.assertIsNone(
            first_admissible_index([frozenset({"foldx_tool"})], active)
        )
        self.assertEqual(first_admissible_index([frozenset()], active), 0)


if __name__ == "__main__":
    unittest.main()
