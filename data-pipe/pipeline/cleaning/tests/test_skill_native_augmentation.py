from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.cleaning.skill_native_augmentation import (
    TOOL_TO_SKILL,
    augment_record,
    normalize_l1_paths,
    normalize_skill_references,
)


def write_skill_set(root: Path) -> None:
    for skill in set(TOOL_TO_SKILL.values()):
        path = root / skill
        path.mkdir()
        (path / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: Test.\n---\n", encoding="utf-8"
        )


class L1ReadAugmentationTest(unittest.TestCase):
    def test_normalizes_legacy_l1_paths_in_nested_content(self) -> None:
        value = {"reasoning": "read .claude/skills/L1_tools/x/SKILL.md", "args": ["ls skills/L1_tools/"]}
        self.assertEqual(
            normalize_l1_paths(value),
            {"reasoning": "read .agents/skills/x/SKILL.md", "args": ["ls .agents/skills/"]},
        )

    def test_normalizes_known_aliases_and_neutralizes_old_report_names(self) -> None:
        value = {
            "reasoning": "Load skill molclaw-mol-drug-chemistry, not molclaw-drug-discovery-methodology.",
            "log": ["server mcp__molclaw-scp__fix_pdb uses molclaw-pdbfixer"],
        }
        self.assertEqual(
            normalize_skill_references(value),
            {
                "reasoning": "Load skill molclaw-drug-likeness, not drug discovery methodology.",
                "log": ["server mcp__molclaw-scp__fix_pdb uses molclaw-fix-pdb"],
            },
        )

    def test_inserts_one_skill_before_first_tool_and_reuses_shared_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill_set(root)
            skill = root / "molclaw-mol-similarity"
            (skill / "SKILL.md").write_text("---\nname: molclaw-mol-similarity\ndescription: Similarity.\n---\n", encoding="utf-8")
            record = {
                "schema_version": "drug_agent_semantic_trajectory_v1",
                "id": "sample",
                "user_task": "compare",
                "events": [
                    {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "High-level plan:\ncompare", "tool_calls": [
                        {"name": "calculate_morgan_fingerprint_similarity", "arguments": {}, "source_tool_use_id": "c1"},
                    ], "final_answer": None},
                    {"type": "tool_observation", "name": "calculate_morgan_fingerprint_similarity", "source_tool_use_id": "c1", "status": "success", "is_error": False, "content": "ok"},
                    {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "fragments", "tool_calls": [
                        {"name": "calculate_common_fragments", "arguments": {}, "source_tool_use_id": "c2"},
                    ], "final_answer": None},
                    {"type": "tool_observation", "name": "calculate_common_fragments", "source_tool_use_id": "c2", "status": "success", "is_error": False, "content": "ok"},
                    {"type": "assistant_decision", "source_message_id": "d3", "reasoning": "done", "tool_calls": [], "final_answer": "answer"},
                ],
                "metadata": {},
            }
            result, stats = augment_record(record, root)
            calls = [call for event in result["events"] if event["type"] == "assistant_decision" for call in event["tool_calls"]]
            self.assertEqual(stats["inserted_skills"], 1)
            self.assertEqual(calls[0]["name"], "skill")
            self.assertEqual(calls[0]["arguments"]["name"], "molclaw-mol-similarity")
            self.assertEqual([call["name"] for call in calls[1:]], ["calculate_morgan_fingerprint_similarity", "calculate_common_fragments"])
            self.assertTrue(result["events"][0]["reasoning"].startswith("High-level plan:\n"))
            self.assertTrue(result["events"][2]["reasoning"].startswith("I have loaded the required L1 skill guidance"))
            self.assertEqual(sum("High-level plan:" in event.get("reasoning", "") for event in result["events"]), 1)
            self.assertIn("<available_skills>", result["metadata"]["skill_native_augmentation"]["catalog_message"])

    def test_normalizes_successful_existing_l1_read_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill_set(root)
            skill = root / "molclaw-fix-pdb"
            (skill / "SKILL.md").write_text("---\nname: molclaw-fix-pdb\ndescription: Repair.\n---\n", encoding="utf-8")
            record = {
                "schema_version": "drug_agent_semantic_trajectory_v1",
                "id": "sample",
                "user_task": "repair",
                "events": [
                    {"type": "assistant_decision", "source_message_id": "d0", "reasoning": "read", "tool_calls": [
                        {"name": "Read", "arguments": {"file_path": "skills/L1_tools/molclaw-fix-pdb/SKILL.md"}, "source_tool_use_id": "r1"},
                    ], "final_answer": None},
                    {"type": "tool_observation", "name": "Read", "source_tool_use_id": "r1", "status": "success", "is_error": False, "content": "old"},
                    {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "repair", "tool_calls": [
                        {"name": "fix_pdb", "arguments": {}, "source_tool_use_id": "c1"},
                    ], "final_answer": None},
                    {"type": "tool_observation", "name": "fix_pdb", "source_tool_use_id": "c1", "status": "success", "is_error": False, "content": "ok"},
                    {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "done", "tool_calls": [], "final_answer": "answer"},
                ],
                "metadata": {},
            }
            result, stats = augment_record(record, root)
            self.assertEqual(stats, {
                "inserted_skills": 0, "converted_l1_reads": 1,
                "removed_catalog_calls": 0, "removed_obsolete_skill_reads": 0,
                "removed_duplicate_skill_calls": 0,
                "canonicalized_skill_references": 0,
                "neutralized_unavailable_skill_references": 0,
            })
            self.assertEqual(result["events"][0]["tool_calls"][0], {
                "name": "skill", "arguments": {"name": "molclaw-fix-pdb"}, "source_tool_use_id": "r1"
            })
            self.assertIn('<skill_content name="molclaw-fix-pdb">', result["events"][1]["content"])

    def test_cleans_reasoning_tool_arguments_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill_set(root)
            record = {
                "schema_version": "drug_agent_semantic_trajectory_v1", "id": "sample", "user_task": "done",
                "events": [
                    {"type": "assistant_decision", "source_message_id": "d0",
                     "reasoning": "Use molclaw-mol-drug-chemistry and record skill molclaw-old-memo.",
                     "tool_calls": [{"name": "Write", "arguments": {
                         "file_path": "notes.txt", "content": "molclaw-pdbfixer / skill: molclaw-old-memo"
                     }, "source_tool_use_id": "w1"}], "final_answer": None},
                    {"type": "tool_observation", "name": "Write", "source_tool_use_id": "w1",
                     "status": "success", "is_error": False, "content": "saved molclaw-pdbfixer"},
                    {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "done",
                     "tool_calls": [], "final_answer": "answer"},
                ], "metadata": {},
            }
            result, stats = augment_record(record, root)
            self.assertIn("molclaw-drug-likeness", result["events"][0]["reasoning"])
            self.assertNotIn("molclaw-old-memo", str(result["events"]))
            self.assertIn("molclaw-fix-pdb", result["events"][0]["tool_calls"][0]["arguments"]["content"])
            self.assertEqual(stats["canonicalized_skill_references"], 3)
            self.assertEqual(stats["neutralized_unavailable_skill_references"], 2)

    def test_failed_skill_load_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill_set(root)
            call = {"name": "skill", "arguments": {"name": "unknown-skill"}, "source_tool_use_id": "bad"}
            observation = {"type": "tool_observation", "name": "skill", "source_tool_use_id": "bad",
                           "status": "error", "is_error": True, "content": "Unknown skill"}
            record = {"schema_version": "drug_agent_semantic_trajectory_v1", "user_task": "report", "id": "error-recovery", "metadata": {}, "events": [
                {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "Try the skill.",
                 "tool_calls": [call], "final_answer": None}, observation,
                {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "Report limitation.",
                 "tool_calls": [], "final_answer": "answer"}]}
            out, _ = augment_record(record, root)
            self.assertEqual(out["events"][0]["tool_calls"], [call])
            self.assertEqual(out["events"][1], observation)

    def test_removes_old_catalog_listing_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill_set(root)
            record = {
                "schema_version": "drug_agent_semantic_trajectory_v1", "id": "sample", "user_task": "done",
                "events": [
                    {"type": "assistant_decision", "source_message_id": "d0", "reasoning": "list", "tool_calls": [
                        {"name": "Bash", "arguments": {"command": "ls skills/L1_tools"}, "source_tool_use_id": "b1"}], "final_answer": None},
                    {"type": "tool_observation", "name": "Bash", "source_tool_use_id": "b1", "status": "success", "is_error": False, "content": "x"},
                    {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "check", "tool_calls": [
                        {"name": "is_valid_smiles", "arguments": {"smiles": "CCO"}, "source_tool_use_id": "c1"}], "final_answer": None},
                    {"type": "tool_observation", "name": "is_valid_smiles", "source_tool_use_id": "c1", "status": "success", "is_error": False, "content": "ok"},
                    {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "done", "tool_calls": [], "final_answer": "answer"},
                ], "metadata": {},
            }
            result, stats = augment_record(record, root)
            self.assertEqual(stats["removed_catalog_calls"], 1)
            self.assertFalse(any(event.get("source_message_id") == "d0" for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
