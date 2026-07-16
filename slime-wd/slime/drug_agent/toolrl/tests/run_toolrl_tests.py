from __future__ import annotations

from pathlib import Path
import sys
import tempfile

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from drug_agent.toolrl.tests.test_converter import (  # noqa: E402
    test_converter_builds_step_level_samples,
    test_converter_skips_final_answer_only_and_malformed,
)
from drug_agent.toolrl.tests.test_parse_tool_calls import (  # noqa: E402
    test_parse_tool_calls_filters_non_molclaw,
    test_parse_tool_calls_rejects_malformed_json,
    test_parse_tool_calls_single_and_multi_call,
)
from drug_agent.toolrl.tests.test_reward import (  # noqa: E402
    test_reward_bool_number_smiles_and_artifact_matching,
    test_reward_hyphen_tool_name_no_longer_matches_underscore,
    test_reward_missing_and_extra_params_penalized,
    test_reward_order_insensitive_multi_tool_calls,
    test_reward_parameter_alias_no_longer_matches,
    test_reward_perfect_match_near_one,
)
from drug_agent.toolrl.validate_toolrl_offline_data import validate_toolrl_offline_data  # noqa: E402


def main() -> int:
    for fn in [
        test_parse_tool_calls_single_and_multi_call,
        test_parse_tool_calls_filters_non_molclaw,
        test_parse_tool_calls_rejects_malformed_json,
        test_reward_perfect_match_near_one,
        test_reward_order_insensitive_multi_tool_calls,
        test_reward_hyphen_tool_name_no_longer_matches_underscore,
        test_reward_parameter_alias_no_longer_matches,
        test_reward_missing_and_extra_params_penalized,
        test_reward_bool_number_smiles_and_artifact_matching,
    ]:
        fn()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_converter_builds_step_level_samples(tmp)
        input_dir = tmp / "input"
        output_path = tmp / "toolrl.jsonl"
        report = validate_toolrl_offline_data(output_path)
        assert report["ok"] is True
        assert report["valid_rows"] == 2
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_converter_skips_final_answer_only_and_malformed(tmp)

    print("ToolRL tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
