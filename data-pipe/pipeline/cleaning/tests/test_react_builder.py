from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))
from cleaning.invariants import validate_final_record  # noqa: E402
from cleaning.react_builder import reconstruct_react_messages  # noqa: E402
from cleaning.trace_parser import discover_rollout_samples  # noqa: E402


def assistant_event(*items: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(items)}}


def user_event(*items: dict) -> dict:
    return {"type": "user", "message": {"content": list(items)}}


class ReactConstructorTest(unittest.TestCase):
    def test_discovers_complete_session_without_parsed_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            row = results / "row0001_idx0"
            row.mkdir()
            (row / "complete_session.jsonl").write_text("{}\n", encoding="utf-8")
            samples = discover_rollout_samples(results)
            self.assertEqual([sample.sample_dir for sample in samples], [row])

    def test_preserves_reasoning_uses_bare_names_and_sanitizes_paths(self) -> None:
        events = [
            assistant_event({"type": "text", "text": "First inspect /home/user/work/input.pdb."}),
            assistant_event(
                {"type": "thinking", "thinking": "Now validate the structure."},
                {"type": "tool_use", "id": "w1", "name": "Read", "input": {"path": "/tmp/debug"}},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {"type": "tool_result", "tool_use_id": "w1", "content": "workspace chatter"},
                {"type": "tool_result", "tool_use_id": "c1", "content": {"status": "success", "valid": True}},
            ),
            assistant_event({"type": "thinking", "thinking": "The recorded result supports the conclusion."}),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer={"answer": "valid"},
            task="kg",
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn("First inspect", rendered)
        self.assertIn('"tool_name":"Read"', rendered)
        self.assertIn('"tool_name":"is_valid_smiles"', rendered)
        self.assertNotIn("mcp__molclaw-scp__", rendered)
        self.assertIn("workspace chatter", rendered)
        self.assertNotIn("/home/user/work", rendered)
        self.assertIn("<artifact:structure/input.pdb>", rendered)
        self.assertEqual(stats["molclaw_usage_count"], 1)
        self.assertEqual(stats["retained_local_tool_call_count"], 1)
        self.assertEqual(stats["raw_tool_name_map"]["mcp__molclaw-scp__is_valid_smiles"], "is_valid_smiles")

    def test_only_molclaw_mode_drops_local_call_and_paired_observation(self) -> None:
        events = [
            assistant_event(
                {"type": "thinking", "thinking": "Inspect the local input before validation."},
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "input.smi"}},
                {
                    "type": "tool_use",
                    "id": "m1",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {"type": "tool_result", "tool_use_id": "r1", "content": "CCO"},
                {
                    "type": "tool_result",
                    "tool_use_id": "m1",
                    "content": {"status": "success", "valid": True},
                },
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer={"answer": "valid"},
            only_molclaw_tool=True,
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertNotIn('"tool_name":"Read"', rendered)
        self.assertNotIn('tool_name="Read"', rendered)
        self.assertIn('"tool_name":"is_valid_smiles"', rendered)
        self.assertEqual(stats["molclaw_usage_count"], 1)
        self.assertEqual(stats["retained_local_tool_call_count"], 0)
        self.assertEqual(stats["dropped_observation_count"], 1)
        self.assertEqual(stats["orphan_observation_count"], 0)
        self.assertEqual(stats["dropped_tool_reason_hist"], {"only_molclaw_tool": 1})
        self.assertEqual(len(stats["only_molclaw_repair_hints"]), 1)

    def test_unpaired_calls_are_not_retained(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "missing",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
                {
                    "type": "tool_use",
                    "id": "paired",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCC"},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "paired",
                    "content": {"status": "success", "valid": True},
                }
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate the molecules",
            final_answer="CCC is valid.",
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertNotIn('"smiles":"CCO"', rendered)
        self.assertIn('"smiles":"CCC"', rendered)
        self.assertEqual(stats["molclaw_usage_count"], 1)
        self.assertEqual(stats["missing_observation_count"], 0)
        self.assertEqual(
            stats["dropped_tool_reason_hist"]["missing_paired_observation"],
            1,
        )

    def test_parallel_results_are_written_in_call_order_by_tool_use_id(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "first",
                    "name": "Read",
                    "input": {"file_path": "run_log.md"},
                },
                {
                    "type": "tool_use",
                    "id": "second",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "second",
                    "content": {"status": "success", "valid": True},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "first",
                    "content": "started",
                },
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer="valid",
        )
        assistant = next(
            message["content"]
            for message in messages
            if message["role"] == "assistant" and "<tool_call>" in message["content"]
        )
        observations = next(
            message["content"]
            for message in messages
            if "<observation " in message["content"]
        )
        self.assertLess(
            assistant.index('"tool_name":"Read"'),
            assistant.index('"tool_name":"is_valid_smiles"'),
        )
        self.assertLess(
            observations.index('tool_name="Read"'),
            observations.index('tool_name="is_valid_smiles"'),
        )
        self.assertEqual(stats["reordered_observation_count"], 2)

    def test_retains_only_l1_skill_and_safe_local_tools(self) -> None:
        l1_path = (
            "/home/teacher/project/.claude/skills/L1_tools/"
            "molclaw-pdbfixer/SKILL.md"
        )
        large_skill = "L1 tool instructions\n" + ("scientific details " * 1000)
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "l1",
                    "name": "Read",
                    "input": {"file_path": l1_path},
                },
                {
                    "type": "tool_use",
                    "id": "skill",
                    "name": "Skill",
                    "input": {"skill": "molclaw-pdbfixer"},
                },
                {
                    "type": "tool_use",
                    "id": "bash",
                    "name": "Bash",
                    "input": {"command": "wc -l run_log.md | head -1"},
                },
                {
                    "type": "tool_use",
                    "id": "write",
                    "name": "Write",
                    "input": {"file_path": "run_log.md", "content": "started"},
                },
                {
                    "type": "tool_use",
                    "id": "mol",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {"type": "tool_result", "tool_use_id": "l1", "content": large_skill},
                {"type": "tool_result", "tool_use_id": "skill", "content": "loaded"},
                {"type": "tool_result", "tool_use_id": "bash", "content": "1 run_log.md"},
                {"type": "tool_result", "tool_use_id": "write", "content": "written"},
                {"type": "tool_result", "tool_use_id": "mol", "content": {"status": "success"}},
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer="valid",
            max_observation_chars=100,
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn("skills/L1_tools/molclaw-pdbfixer/SKILL.md", rendered)
        self.assertIn("scientific details scientific details", rendered)
        self.assertIn('"tool_name":"Skill"', rendered)
        self.assertIn('"tool_name":"Bash"', rendered)
        self.assertIn('"tool_name":"Write"', rendered)
        self.assertEqual(stats["molclaw_usage_count"], 1)
        self.assertEqual(stats["retained_local_tool_call_count"], 4)

    def test_drops_l2_l3_teacher_tools_and_unsafe_bash_with_audit(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "l2",
                    "name": "Read",
                    "input": {"file_path": "/work/.claude/skills/L2_workflows/docking/SKILL.md"},
                },
                {
                    "type": "tool_use",
                    "id": "workflow",
                    "name": "Skill",
                    "input": {"skill": "molecular-docking-screening"},
                },
                {
                    "type": "tool_use",
                    "id": "danger",
                    "name": "Bash",
                    "input": {"command": "python make_report.py"},
                },
                {"type": "tool_use", "id": "web", "name": "WebSearch", "input": {"query": "target"}},
                {"type": "tool_use", "id": "task", "name": "Task", "input": {"prompt": "delegate"}},
                {
                    "type": "tool_use",
                    "id": "mol",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                *[
                    {"type": "tool_result", "tool_use_id": call_id, "content": "removed"}
                    for call_id in ("l2", "workflow", "danger", "web", "task")
                ],
                {"type": "tool_result", "tool_use_id": "mol", "content": {"status": "success"}},
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer="valid",
        )
        rendered = json.dumps(messages, ensure_ascii=False)
        for name in ("Read", "Skill", "Bash", "WebSearch", "Task"):
            self.assertNotIn(f'"tool_name":"{name}"', rendered)
        self.assertEqual(stats["molclaw_usage_count"], 1)
        self.assertEqual(stats["dropped_observation_count"], 5)
        self.assertEqual(stats["dropped_tool_reason_hist"]["unsupported_teacher_tool"], 2)
        self.assertEqual(stats["dropped_tool_reason_hist"]["non_l1_skill"], 1)
        self.assertEqual(stats["dropped_tool_reason_hist"]["unsafe_bash_command"], 1)

    def test_drops_teacher_sidecars_and_non_l1_skill_catalog_access(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "question",
                    "name": "Read",
                    "input": {"file_path": "question.json"},
                },
                {
                    "type": "tool_use",
                    "id": "catalog",
                    "name": "Bash",
                    "input": {"command": "ls -la .claude/skills/"},
                },
                {
                    "type": "tool_use",
                    "id": "log",
                    "name": "Write",
                    "input": {"file_path": "run_log.md", "content": "started"},
                },
                {
                    "type": "tool_use",
                    "id": "mol",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "question",
                    "content": {"answer": "hidden benchmark label"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "catalog",
                    "content": "L1_tools\nL2_workflows\nL3_methodology",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "log",
                    "content": "written",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "mol",
                    "content": {"status": "success", "valid": True},
                },
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer="valid",
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertNotIn("hidden benchmark label", rendered)
        self.assertNotIn("L2_workflows", rendered)
        self.assertNotIn('"tool_name":"Read"', rendered)
        self.assertNotIn('"tool_name":"Bash"', rendered)
        self.assertIn('"tool_name":"Write"', rendered)
        self.assertEqual(
            stats["dropped_tool_reason_hist"]["teacher_runtime_sidecar_access"],
            1,
        )
        self.assertEqual(
            stats["dropped_tool_reason_hist"]["non_l1_skill_catalog_access"],
            1,
        )

    def test_compacts_as_json_and_records_error_status_conflict(self) -> None:
        events = [
            assistant_event(
                {"type": "tool_use", "id": "c1", "name": "mcp__molclaw-scp__fpocket_toolkit", "input": {}}
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "is_error": False,
                    "content": {
                        "status": "error",
                        "message": "pocket calculation failed",
                        "rows": [{"value": index} for index in range(200)],
                    },
                }
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Find a pocket",
            final_answer="No pocket was produced.",
            task="e2e",
            max_observation_chars=200,
        )
        observation_text = next(message["content"] for message in messages if message["role"] == "user" and "<observation" in message["content"])
        payload_text = observation_text.split(">", 1)[1].rsplit("</observation>", 1)[0]
        payload = json.loads(payload_text)
        self.assertTrue(payload["is_error"])
        self.assertTrue(payload["content"]["compacted"])
        self.assertEqual(payload["compaction"]["method"], "structured_summary")
        self.assertEqual(len(stats["error_status_conflicts"]), 1)

    def test_final_answer_keeps_task_result_summary_and_evidence(self) -> None:
        events = [
            assistant_event(
                {"type": "tool_use", "id": "c1", "name": "mcp__molclaw-scp__dock", "input": {"smiles": "CCO"}}
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": {"status": "success", "score": -7.1, "output_path": "/tmp/dock/result.sdf"},
                }
            ),
            assistant_event({"type": "text", "text": "Docking completed with the recorded score."}),
        ]
        messages, _ = reconstruct_react_messages(
            events,
            question_text="Dock CCO",
            final_answer={"result": "completed"},
            task="e2e",
        )
        final_text = messages[-1]["content"]
        payload = json.loads(final_text.removeprefix("<final_answer>").removesuffix("</final_answer>"))
        self.assertEqual(payload["result"], {"result": "completed"})
        self.assertIn("Docking completed", payload["summary"])
        self.assertEqual(payload["evidence"][0]["key_values"]["score"], -7.1)
        self.assertIn("<artifact:", json.dumps(payload, ensure_ascii=False))

    def test_all_absolute_paths_share_mapping_and_final_uses_real_output_artifact(self) -> None:
        source_path = "/data/services/proteins/P08913_6K42.pdb"
        output_path = "/opt/molclaw/results/P08913_6K42_fixed.pdb"
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mcp__molclaw-scp__retrieve_protein_structure_by_gene_name",
                    "input": {"gene_name": "ADRA2A", "organism": "9606"},
                }
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": {"status": "success", "prot_structure_path": source_path},
                }
            ),
            assistant_event(
                {"type": "thinking", "thinking": f"Repair `{source_path}` with the recorded options."},
                {
                    "type": "tool_use",
                    "id": "c2",
                    "name": "mcp__molclaw-scp__fix_pdb",
                    "input": {"input_path": source_path, "add_hydrogens": True},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c2",
                    "content": {"status": "success", "output_file": output_path, "atom_count": 4495},
                }
            ),
            assistant_event(
                {"type": "text", "text": f"The repaired structure is available at {output_path}."}
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Retrieve and repair ADRA2A.",
            final_answer=[
                "Repaired file: step02_P08913_6K42_fixed.pdb. "
                "The result.md and run_log.md reports were written."
            ],
            task="kg",
        )
        rendered = json.dumps(messages, ensure_ascii=False)
        source_ref = "<artifact:structure/P08913_6K42.pdb>"
        output_ref = "<artifact:structure/P08913_6K42_fixed.pdb>"
        self.assertNotIn("/data/", rendered)
        self.assertNotIn("/opt/", rendered)
        self.assertGreaterEqual(rendered.count(source_ref), 3)
        self.assertEqual(stats["artifact_mappings"][source_path], source_ref)
        self.assertNotIn(f"{source_path}`", stats["artifact_mappings"])
        final_payload = json.loads(
            messages[-1]["content"].removeprefix("<final_answer>").removesuffix("</final_answer>")
        )
        self.assertEqual(final_payload["result"], output_ref)
        self.assertNotIn("step02_P08913_6K42_fixed.pdb", json.dumps(final_payload["result"]))
        self.assertNotIn("run_log.md", json.dumps(final_payload["result"]))

    def test_primary_artifact_comes_from_molclaw_not_later_local_write(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "mol",
                    "name": "mcp__molclaw-scp__fix_pdb",
                    "input": {"input_path": "/data/input.pdb"},
                }
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "mol",
                    "content": {"status": "success", "output_file": "/data/fixed.pdb"},
                }
            ),
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "write",
                    "name": "Write",
                    "input": {"file_path": "result.md", "content": "Delivered fixed.pdb"},
                }
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "write",
                    "content": {"status": "success", "output_file": "/workspace/result.md"},
                }
            ),
        ]
        messages, _ = reconstruct_react_messages(
            events,
            question_text="Repair the structure",
            final_answer="The repaired structure is fixed.pdb; details are in result.md.",
            task="kg",
        )
        payload = json.loads(
            messages[-1]["content"].removeprefix("<final_answer>").removesuffix("</final_answer>")
        )
        self.assertEqual(payload["result"], "<artifact:structure/fixed.pdb>")

    def test_vs_ranking_is_repaired_from_complete_same_context_quickvina_scores(self) -> None:
        common = {
            "pdb_file_path": "/data/receptor.pdb",
            "pocket_center_x": 1.0,
            "pocket_center_y": 2.0,
            "pocket_center_z": 3.0,
            "pocket_size_x": 25,
            "pocket_size_y": 25,
            "pocket_size_z": 25,
        }
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "mcp__molclaw-scp__molecule_docking_quickvina_fullprocess",
                    "input": {**common, "smiles": "AAA"},
                },
                {
                    "type": "tool_use",
                    "id": "b",
                    "name": "mcp__molclaw-scp__molecule_docking_quickvina_fullprocess",
                    "input": {**common, "smiles": "BBB"},
                },
                {
                    "type": "tool_use",
                    "id": "c",
                    "name": "mcp__molclaw-scp__molecule_docking_quickvina_fullprocess",
                    "input": {**common, "smiles": "CCC"},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "a",
                    "content": {"status": "success", "docking_affinity_value": -8.0},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "b",
                    "content": {"status": "success", "docking_affinity_value": -6.0},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "c",
                    "content": {"status": "success", "docking_affinity_value": -8.0},
                },
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Rank the candidates",
            final_answer={"ranked_smiles": ["BBB", "CCC", "AAA"], "selected_smiles": "BBB"},
            task="vs",
        )
        payload = json.loads(
            messages[-1]["content"].removeprefix("<final_answer>").removesuffix("</final_answer>")
        )
        self.assertEqual(payload["ranked_smiles"], ["CCC", "AAA", "BBB"])
        self.assertEqual(payload["selected_smiles"], "CCC")
        self.assertEqual(stats["resolved_final_answer"]["ranked_smiles"], ["CCC", "AAA", "BBB"])
        self.assertEqual(stats["vs_ranking_repair"]["status"], "repaired")

    def test_vs_ranking_repair_skips_ambiguous_context_or_higher_order_evidence(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "mcp__molclaw-scp__molecule_docking_quickvina_fullprocess",
                    "input": {"smiles": "AAA", "pocket_center_x": 1},
                },
                {
                    "type": "tool_use",
                    "id": "b",
                    "name": "mcp__molclaw-scp__molecule_docking_quickvina_fullprocess",
                    "input": {"smiles": "BBB", "pocket_center_x": 2},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "a",
                    "content": {"status": "success", "docking_affinity_value": -8.0},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "b",
                    "content": {"status": "success", "docking_affinity_value": -6.0},
                },
            ),
        ]
        _, stats = reconstruct_react_messages(
            events,
            question_text="Rank",
            final_answer=["BBB", "AAA"],
            task="vs",
        )
        self.assertEqual(stats["resolved_final_answer"], ["BBB", "AAA"])
        self.assertEqual(stats["vs_ranking_repair"]["status"], "skipped")
        self.assertEqual(
            stats["vs_ranking_repair"]["reason"],
            "ranking_uses_different_quickvina_contexts",
        )

    def test_path_sanitization_does_not_rewrite_smiles_stereochemistry(self) -> None:
        smiles = "N=C(N)N/C(=N\\C1CCCCC1)c1ccc(Cl)cc1"
        events = [
            assistant_event(
                {"type": "thinking", "thinking": f"Evaluate {smiles} before selecting it."},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": smiles},
                },
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": {"status": "success", "smiles": smiles, "valid": True},
                }
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text=f"Return the valid molecule {smiles}",
            final_answer=[smiles],
            task="pf",
        )
        rendered = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("<artifact:", rendered)
        self.assertGreaterEqual(rendered.count("N=C(N)N/C"), 4)
        self.assertGreaterEqual(rendered.count("C1CCCCC1"), 4)
        self.assertEqual(stats["artifact_mappings"], {})

    def test_invariant_validation_does_not_reconstruct_or_recompact(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mcp__molclaw-scp__fpocket_toolkit",
                    "input": {},
                }
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": {
                        "status": "success",
                        "score": 3.5,
                        "rows": [{"value": index} for index in range(200)],
                    },
                }
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Find a pocket",
            final_answer={"result": "Pocket 1"},
            task="e2e",
            max_observation_chars=200,
        )
        self.assertEqual(stats["compacted_observation_count"], 1)
        source_roles = [message["role"] for message in messages]
        self.assertEqual(
            sum(
                "<final_answer>" in message["content"]
                for message in messages
                if message["role"] == "assistant"
            ),
            1,
        )
        source_observation = next(
            message["content"] for message in messages if "<observation " in message["content"]
        )
        source_payload = json.loads(source_observation.split(">", 1)[1].split("</observation>", 1)[0])

        assistant_message = next(
            message for message in messages if message["role"] == "assistant" and "<tool_call>" in message["content"]
        )
        assistant_message["content"] = (
            "<thought>Review ../outputs/result.pdb.</thought>\n" + assistant_message["content"]
        )
        record = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "react_test",
            "messages": messages,
        }
        source_json = json.dumps(record, ensure_ascii=False, sort_keys=True)
        report = validate_final_record(record)
        cleaned_text = "\n".join(message["content"] for message in record["messages"])
        cleaned_observation = next(
            message["content"] for message in record["messages"] if "<observation " in message["content"]
        )
        cleaned_payload = json.loads(cleaned_observation.split(">", 1)[1].split("</observation>", 1)[0])

        self.assertEqual([message["role"] for message in record["messages"]], source_roles)
        self.assertEqual(
            sum(
                "<final_answer>" in message["content"]
                for message in record["messages"]
                if message["role"] == "assistant"
            ),
            1,
        )
        self.assertEqual(cleaned_payload, source_payload)
        self.assertIn("../outputs/result.pdb", cleaned_text)
        self.assertIn("message_2_unsanitized_path", report["errors"])
        self.assertEqual(json.dumps(record, ensure_ascii=False, sort_keys=True), source_json)
        self.assertEqual(report["counts"]["tool_calls"], 1)
        self.assertEqual(report["counts"]["observations"], 1)
        self.assertNotIn("final_status", report)


if __name__ == "__main__":
    unittest.main()
