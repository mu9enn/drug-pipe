from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_dsh_molbench as runner
from audit_protocol import audit
from install_eval_preset import PRESET_ID, install
from pipeline.output_contracts import CONTRACTS, normalize_task_prompt


def pf(row: int = 1) -> runner.Sample:
    instruction = (
        "Print each satisfying SMILES on its own line, and nothing else."
        if row <= 35 else
        "Print ONLY the selected SMILES and nothing else."
    )
    return runner.Sample(
        f"pf-{row}", "ms1", row,
        normalize_task_prompt(f"SMILES:\nCC\nCCC\nConstraints:\nselect CC\nOutput format:\n{instruction}", "pf"),
        "CC",
    )


def ac() -> runner.Sample:
    return runner.Sample(
        "ac", "ms2", 1,
        normalize_task_prompt("Molecule A: CC\nMolecule B: CCC\nOnly output the corresponding SMILES.", "ac"),
        "CC", molecule_a="CC", molecule_b="CCC",
    )


class StructuredV8ProjectionTest(unittest.TestCase):
    def test_pf_rows_1_to_35_accept_list_and_allow_empty(self) -> None:
        self.assertEqual(
            runner.project_prediction(pf(), '{"selected_smiles":["CC"],"evidence":[]}'),
            (["CC"], True, None),
        )
        self.assertEqual(
            runner.project_prediction(pf(), '{"selected_smiles":[],"evidence":[]}'),
            ([], True, None),
        )

    def test_pf_rows_36_to_50_require_exactly_one_list_value(self) -> None:
        sample = pf(36)
        self.assertEqual(
            runner.project_prediction(sample, '{"selected_smiles":["CC"],"evidence":[]}'),
            (["CC"], True, None),
        )
        for answer in (
            '{"selected_smiles":[],"evidence":[]}',
            '{"selected_smiles":["CC","CCC"],"evidence":[]}',
            '{"answer_smiles":"CC","evidence":[]}',
        ):
            with self.subTest(answer=answer):
                prediction, valid, _ = runner.project_prediction(sample, answer)
                self.assertEqual(prediction, [])
                self.assertFalse(valid)

    def test_ac_requires_one_exact_candidate_string(self) -> None:
        self.assertEqual(
            runner.project_prediction(ac(), '{"answer_smiles":"CC","evidence":[]}'),
            ("CC", True, None),
        )
        for answer in (
            '{"answer_smiles":["CC"],"evidence":[]}',
            '{"answer_smiles":" C C ","evidence":[]}',
            '{"answer_smiles":"CO","evidence":[]}',
        ):
            with self.subTest(answer=answer):
                self.assertFalse(runner.project_prediction(ac(), answer)[1])

    def test_nonconforming_surfaces_are_rejected_without_recovery(self) -> None:
        fence = chr(96) * 3 + 'json\n{"answer_smiles":"CC","evidence":[]}\n' + chr(96) * 3
        cases = (
            "CC",
            '<final_answer>{"answer_smiles":"CC","evidence":[]}</final_answer>',
            fence,
            'The answer is {"answer_smiles":"CC","evidence":[]}.',
            '<answer>{"answer_smiles":"CC","evidence":[]}</answer>',
            '{"task_type":"ac","answer_smiles":"CC","evidence":[]}',
            '{"answer_smiles":"CC","evidence":[],"summary":"x"}',
            '{"answer_smiles":"CC"}',
            '{"answer_smiles":"CC","evidence":"x"}',
        )
        for answer in cases:
            with self.subTest(answer=answer):
                prediction, valid, error = runner.project_prediction(ac(), answer)
                self.assertEqual(prediction, "")
                self.assertFalse(valid)
                self.assertIsNotNone(error)

    def test_pf_values_are_not_trimmed_or_deduplicated(self) -> None:
        sample = pf()
        prediction, valid, _ = runner.project_prediction(
            sample, '{"selected_smiles":["CC","CC"],"evidence":[]}',
        )
        self.assertEqual(prediction, ["CC", "CC"])
        self.assertTrue(valid)
        self.assertEqual(
            runner.project_prediction(sample, '{"selected_smiles":[" CC"],"evidence":[]}')[0],
            [],
        )


class PromptAndPresetTest(unittest.TestCase):
    def test_task_normalization_is_shared_and_idempotent(self) -> None:
        for sample, task_type in ((pf(), "pf"), (pf(36), "pf"), (ac(), "ac")):
            with self.subTest(task_id=sample.task_id):
                actual = runner.aligned_task_text(sample)
                self.assertTrue(actual.endswith(CONTRACTS[task_type]))
                if sample.source_row <= 35 or sample.suite != "ms1":
                    self.assertEqual(actual, normalize_task_prompt(sample.prompt, task_type))
        self.assertIn('"selected_smiles"', runner.aligned_task_text(pf(36)))
        self.assertNotIn('"answer_smiles"', runner.aligned_task_text(pf(36)))
        self.assertIn("exactly one candidate", runner.aligned_task_text(pf(36)))

    def test_eval_preset_has_complete_v8_system_and_only_required_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standard = root / "dsh/apps/cli/config/agent-presets/standard/agent.cordis.yml"
            standard.parent.mkdir(parents=True)
            standard.write_text("- id: persona\n", encoding="utf-8")
            system = root / "system.md"
            system.write_text("v8 system\n", encoding="utf-8")
            target = install(root / "dsh", root / "home", PRESET_ID, system)
            generated = (target / "agent.cordis.yml").read_text(encoding="utf-8")
            ids = [
                line.removeprefix("- id: ").strip()
                for line in generated.splitlines() if line.startswith("- id: ")
            ]
            self.assertEqual(ids, [
                "persona", "tool-bash", "tool-pwsh", "tool-fs",
                "tool-fs-search", "skill-filesystem", "tool-skill",
            ])
            self.assertIn("v8 system", generated)
            self.assertIn("complete: true", generated)
            self.assertIn("includeRuntimeContext: false", generated)
            for forbidden in ("tool-web", "tool-goal", "tool-todo", "delegation", "agent-instructions"):
                self.assertNotIn(forbidden, generated)

    def test_workspace_is_copied_verbatim_and_prefix_is_sent_with_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            skill = source / ".agents/skills/test-tool/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: test-tool\ndescription: test\n---\n", encoding="utf-8")
            (source / "prompt_prefix.md").write_text("policy\n", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            payload = runner.create_workspace_snapshot(run_dir, source)
            workspace = runner.prepare_workspace(run_dir, pf(), payload)
            self.assertTrue((workspace / ".agents/skills/test-tool/SKILL.md").is_file())
            self.assertFalse((workspace / "AGENTS.md").exists())
            self.assertFalse((workspace / "question.json").exists())
            self.assertTrue(runner.task_prompt(pf(), "policy").startswith("policy\n\n# Task"))
            manifest = json.loads((run_dir / "workspace_snapshot/manifest.json").read_text())
            self.assertEqual(manifest["adaptations"], [])


class RetryClassificationTest(unittest.TestCase):
    def test_only_transport_failures_are_retryable(self) -> None:
        timeout = {
            "status": "failed",
            "turn_reason": {"kind": "error", "error": {"code": "TIMEOUT", "message": "idle"}},
        }
        self.assertEqual(runner.failure_class(timeout), "retryable_infra")
        self.assertEqual(
            runner.failure_class({"status": "failed", "turn_reason": {"kind": "max-tokens"}}),
            "model_max_tokens",
        )
        self.assertEqual(
            runner.failure_class({"status": "failed", "error": "bad answer"}),
            "unclassified_failure",
        )
        self.assertFalse(runner.publishable_records([{'status': 'failed', 'protocol_verified': True,
                                                     'failure_class': 'unclassified_failure'}]))
        self.assertEqual(runner.failure_class({'status': 'failed', 'error': 'task exceeded 14400 seconds'}),
                         'model_or_protocol_failure')


class ScorerProjectionTest(unittest.TestCase):
    def test_pf_single_list_reaches_scorer_and_invalid_text_becomes_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = root / "molbench" / runner.PINNED_SCORER
            evaluator.parent.mkdir(parents=True)
            evaluator.write_bytes((runner.DEFAULT_MOLBENCH_ROOT / runner.PINNED_SCORER).read_bytes())
            run_dir = root / "run"
            good, bad = pf(36), pf(37)
            for sample, text in (
                (good, '{"selected_smiles":["CC"],"evidence":[]}'),
                (bad, "CC"),
            ):
                result = run_dir / "results" / sample.task_id
                result.mkdir(parents=True)
                (result / "record.json").write_text(json.dumps({
                    "status": "completed", "final_text": text,
                }), encoding="utf-8")
            runner.materialize_scores(run_dir, root / "molbench", [good, bad])
            rows = json.loads((run_dir / "preds/rdkit_bench/all.json").read_text())
            self.assertEqual(rows[0]["json_results"]["output"], "CC")
            self.assertEqual(rows[1]["json_results"]["output"], "")


class ProtocolAuditTest(unittest.TestCase):
    def test_startup_probe_snapshot_counts_tools_and_skills(self) -> None:
        tools = [{"name": name} for name in (
            ["bash", "read", "write", "edit", "grep", "glob", "skill"]
            + [f"mcp__molclaw-scp__tool_{index}" for index in range(81)]
        )]
        snapshot = runner.protocol_snapshot([
            {"type": "user/message", "data": {
                "source": {"kind": "skill-catalog", "entries": [
                    {"name": f"skill-{index}"} for index in range(68)
                ]},
            }},
            {"type": "request/header", "data": {"header": {
                "system": "v8", "config": {"maxTokens": 16384}, "tools": tools,
            }}},
        ])
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["molclaw_tool_count"], 81)
        self.assertEqual(snapshot["skill_catalog_count"], 68)
        self.assertEqual(set(snapshot["local_tools"]), runner.LOCAL_EVAL_TOOLS)
        self.assertFalse(runner.protocol_snapshot_matches(snapshot, "v8", 81, 68))  # Names alone cannot establish schema parity.
        self.assertFalse(runner.protocol_snapshot_matches(snapshot, "v8", 80, 68))

    def test_real_request_gate_checks_system_order_and_exact_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = root / "system.md"
            system.write_text("v8 system\n", encoding="utf-8")
            transcript = root / "run/results/task/diagnostic_transcript.json"
            transcript.parent.mkdir(parents=True)
            tools = [{"name": name} for name in (
                ["bash", "read", "write", "edit", "grep", "glob", "skill"]
                + [f"mcp__molclaw-scp__tool_{index}" for index in range(81)]
            )]
            transcript.write_text(json.dumps([
                {"type": "user/message", "data": {
                    "source": {"kind": "user"},
                    "content": [{"type": "text", "text": "task"}],
                }},
                {"type": "user/message", "data": {
                    "source": {"kind": "skill-catalog", "entries": [
                        {"name": f"skill-{index}"} for index in range(52)
                    ]},
                }},
                {"type": "request/header", "data": {"header": {
                    "system": "v8 system", "config": {"maxTokens": 16384}, "tools": tools,
                }}},
            ]), encoding="utf-8")
            (root / "run/run_manifest.json").write_text(json.dumps({
                "task_prompt_sha256": {
                    "task": hashlib.sha256(b"task").hexdigest(),
                },
            }), encoding="utf-8")
            result = audit(root / "run", system, 81, 52)
            self.assertTrue(result["passed"])
            self.assertEqual(result["tool_count"], 88)
            self.assertEqual(result["skill_catalog_count"], 52)

    def test_real_request_gate_rejects_missing_skill_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = root / "system.md"
            system.write_text("v8 system\n", encoding="utf-8")
            transcript = root / "run/results/task/diagnostic_transcript.json"
            transcript.parent.mkdir(parents=True)
            tools = [{"name": name} for name in (
                ["bash", "read", "write", "edit", "grep", "glob", "skill"]
                + [f"mcp__molclaw-scp__tool_{index}" for index in range(81)]
            )]
            transcript.write_text(json.dumps([
                {"type": "user/message", "data": {
                    "source": {"kind": "user"},
                    "content": [{"type": "text", "text": "task"}],
                }},
                {"type": "request/header", "data": {"header": {
                    "system": "v8 system", "config": {"maxTokens": 16384}, "tools": tools,
                }}},
            ]), encoding="utf-8")
            (root / "run/run_manifest.json").write_text(json.dumps({
                "task_prompt_sha256": {
                    "task": hashlib.sha256(b"task").hexdigest(),
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "skill catalog"):
                audit(root / "run", system, 81, 68)


if __name__ == "__main__":
    unittest.main()
