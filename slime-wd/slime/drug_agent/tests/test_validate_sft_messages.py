import unittest

from drug_agent.data.validate_sft_messages import audit_react_actions


class ValidateSftMessagesTest(unittest.TestCase):
    def test_parallel_tool_results_may_arrive_out_of_submission_order(self) -> None:
        messages = [
            {"role": "user", "content": "test parallel tools"},
            {
                "role": "assistant",
                "content": (
                    '<thought>Run both.</thought>'
                    '<tool_call>{"tool_name":"tool_a","arguments":{}}</tool_call>'
                    '<tool_call>{"tool_name":"tool_b","arguments":{}}</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    '<observation tool_name="tool_b">{"ok":true}</observation>'
                    '<observation tool_name="tool_a">{"ok":true}</observation>'
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<final_answer>{"task_type":"kg","result":"done",'
                    '"evidence":[]}</final_answer>'
                ),
            },
        ]

        counts, issues = audit_react_actions(messages)

        self.assertEqual(issues, [])
        self.assertEqual(counts["react_json_parse_failed"], 0)
        self.assertEqual(counts["orphan_tool_calls"], 0)
        self.assertEqual(counts["out_of_order_tool_results"], 1)

    def test_unmatched_tool_result_is_still_rejected(self) -> None:
        messages = [
            {"role": "user", "content": "test mismatch"},
            {
                "role": "assistant",
                "content": '<tool_call>{"tool_name":"tool_a","arguments":{}}</tool_call>',
            },
            {
                "role": "user",
                "content": '<observation tool_name="tool_b">{"ok":true}</observation>',
            },
        ]

        counts, issues = audit_react_actions(messages)

        self.assertGreater(counts["react_json_parse_failed"], 0)
        self.assertEqual(counts["orphan_tool_calls"], 1)
        self.assertTrue(
            any(
                "expected tool_a, got tool_b" in issue["strict_error_message"]
                for issue in issues
            )
        )


if __name__ == "__main__":
    unittest.main()
