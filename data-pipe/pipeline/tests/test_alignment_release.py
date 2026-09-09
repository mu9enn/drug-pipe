import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from pipeline.output_contracts import normalize_final_answer, normalize_task_prompt, task_constraints
from pipeline.benchmark_release import ROOT, UPSTREAM_COMMIT, SPECS, heldout_groups, task_group, scientific_question_key
from pipeline.cleaning.release_aligned import repair_answer, unique_candidate_mapping
from pipeline.cleaning.skill_native_augmentation import skill_catalog, render_catalog, render_skill_result

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT/'slime-wd/dsh-molbench'))
import run_dsh_molbench as runner

class AlignmentReleaseTest(unittest.TestCase):
    def test_repeated_context_gate_updates_training_file_hashes(self):
        from pipeline.cleaning.gate_release import update_release_manifest
        from pipeline.benchmark_release import digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'training').mkdir()
            (root/'release_manifest.json').write_text('{"files":{"training/qwen35_sft_train.jsonl":"stale"}}')
            for name in ['context_gate_manifest.json', 'probes.jsonl', 'qwen35_sft_train.jsonl']:
                (root/'training'/name).write_text('new data')
            update_release_manifest(root, {'accepted_count': 1})
            manifest = json.loads((root/'release_manifest.json').read_text())
            for name, sha in manifest['files'].items():
                self.assertEqual(sha, digest((root/name).read_bytes()))

    def test_all_upstream_tasks_preserve_science(self):
        for suite,(folder,column,task,count) in SPECS.items():
            with (ROOT/'upstream'/UPSTREAM_COMMIT/'data'/folder/f'{folder}.csv').open(encoding='utf-8-sig',newline='') as f: original=list(csv.DictReader(f))
            samples=runner.load_samples(ROOT,{suite},0)
            self.assertEqual(len(samples),count)
            for row,sample in zip(original,samples):
                self.assertEqual(task_constraints(row[column],task),task_constraints(sample.prompt,task))
                self.assertEqual(normalize_task_prompt(sample.prompt,task),sample.prompt)
                self.assertEqual(sample.answer,row['answer'])
                self.assertEqual(scientific_question_key(row[column]), scientific_question_key(sample.prompt))
                if task=='vs':
                    before=json.loads(row[column]);after=json.loads(sample.prompt)
                    before.pop('output_format');after.pop('output_format');self.assertEqual(before,after)
        self.assertEqual(len(runner.load_samples(ROOT,{'ms1','ms2'},0)),87)

    def test_strict_json_and_candidates(self):
        q='For the target X, you are given\nMolecule A: CCO\nMolecule B: CCN'
        c=task_constraints(q,'ac')
        for bad in ['{"answer_smiles":"[CH]","evidence":[]}', '{"answer_smiles":"CCO","answer_smiles":"CCN","evidence":[]}', '{"answer_smiles":"CCO","evidence":[NaN]}']:
            with self.assertRaises(ValueError):normalize_final_answer(bad,'ac',constraints=c)
        self.assertEqual(json.loads(normalize_final_answer('{"answer_smiles":"CCO","evidence":[]}','ac',constraints=c))['answer_smiles'],'CCO')

    def test_reverse_question_is_same_group(self):
        q=runner.load_samples(ROOT,{'ms2'},0)[3].prompt
        reverse=q.replace('lower binding affinity (higher Ki)','higher binding affinity (lower Ki)')
        self.assertEqual(task_group(q,'ac'),task_group(reverse,'ac'))
        self.assertIn(task_group(reverse,'ac'),heldout_groups())

    def test_equivalent_candidate_mapping_preserves_stereochemistry_and_ambiguity(self):
        self.assertEqual(unique_candidate_mapping(['OCC'], ('CCO', 'CCN')), ['CCO'])
        with self.assertRaises(ValueError):
            unique_candidate_mapping(['OCC'], ('CCO', 'C(C)O'))
        with self.assertRaises(ValueError):
            unique_candidate_mapping(['C[C@@H](O)F'], ('C[C@H](O)F', 'CCO'))

    def test_filter_empty_single_count_and_incomplete_rank(self):
        question = 'SMILES:\nCCO\nCCN\nConstraints:\nSelect passing candidates.'
        empty = '{"selected_smiles":[],"evidence":[]}'
        normalize_final_answer(empty, 'pf', constraints=task_constraints(question, 'pf'))
        with self.assertRaises(ValueError):
            normalize_final_answer(empty, 'pf', constraints=task_constraints(question + '\nSelect exactly one candidate.', 'pf'))
        vs = task_constraints('{"candidates":["CCO","CCN"]}', 'vs')
        for values in [['CCO'], ['CCO', 'CCO'], ['CCO', '[CH]']]:
            with self.assertRaises(ValueError):
                normalize_final_answer(json.dumps({'ranked_smiles': values, 'evidence': []}), 'vs', constraints=vs)

    def test_cli_supports_ms3_and_last_tool_message_is_not_a_final(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['runner', '--suite', 'ms3']):
            self.assertEqual(runner.parse_args().suite, ['ms3'])
        events = [{'type': 'assistant/message', 'data': {'message': {'content': [
            {'type': 'text', 'text': '{"result":1,"evidence":[]}'},
            {'type': 'tool-call', 'name': 'probe'}]}}}]
        self.assertEqual(runner.summarize_events(events)['final_text'], '')

    def test_rankings_cannot_be_filled_or_deduplicated(self):
        q=json.dumps({'target_name':'X','candidates':['CCO','CCN']})
        for values in [['CCO'],['CCO','CCO'],['CCO','CO']]:
            old=json.dumps({'ranked_smiles':values,'evidence':[]})
            with self.assertRaises(ValueError):repair_answer(old,'vs',q,'Task completed.')

    def test_evidence_only_repair(self):
        q='For the target X, you are given\nMolecule A: CCO\nMolecule B: CCN'
        old='{"answer_smiles":"[CH]","evidence":[]}'
        fixed,method=repair_answer(old,'ac',q,'Selected **Molecule A: `CCO`** based on docking observations.')
        self.assertEqual(json.loads(fixed)['answer_smiles'],'CCO')
        for ambiguous in ['Compare `CCO` and `CCN`.', 'Expected answer is `CCO`.', 'I inspected `CCO` but cannot conclude.']:
            with self.assertRaises(ValueError):repair_answer(old,'ac',q,ambiguous)

    def test_recorded_catalog_and_skill_results_match_adapter(self):
        fixture=json.loads((PROJECT/'data-pipe/pipeline/cleaning/tests/fixtures/dsh_l1_protocol.json').read_text())
        skills=PROJECT/'workdir-skills/molclaw-l1-workspace/.agents/skills'
        self.assertEqual(render_catalog(skill_catalog(skills)),fixture['catalog']['content'][0]['text'])
        for result in fixture['skill_results']:
            name=re.search(r'<skill_content name="([^"]+)"',result)[1]
            normalized=re.sub(r'Base directory for this skill: .*?/\.agents/skills/', 'Base directory for this skill: .agents/skills/',result)
            self.assertEqual(render_skill_result(name,skills),normalized)

    def test_all_scorers_use_complete_denominator(self):
        samples=runner.load_samples(ROOT,{'ms1','ms2','ms3'},1)
        with tempfile.TemporaryDirectory() as tmp:
            summary=runner.materialize_scores(Path(tmp),ROOT,samples)
            self.assertEqual(summary['sample_count'],3)
            self.assertFalse(summary['publishable'])
            for metrics in summary['metrics'].values():self.assertEqual(metrics['n_samples'],1)
            self.assertEqual(summary['metrics']['molbench_vs_all']['avg_rank'],61)

    def test_ms3_mock_rollout_reaches_official_scorer(self):
        sample=runner.load_samples(ROOT,{'ms3'},1)[0]
        fixture=json.loads((PROJECT/'data-pipe/pipeline/cleaning/tests/fixtures/dsh_l1_protocol.json').read_text())
        answer=json.dumps({'ranked_smiles':list(task_constraints(sample.prompt,'vs').candidates),'evidence':[]})
        events=[{'type':'user/message','data':fixture['catalog']},
                {'type':'request/header','data':{'header':{'system':'test-system','tools':fixture['tools'],'config':{'maxTokens':16384}}}},
                {'type':'tool/call','data':{'turn':1,'name':'skill','callId':'c1'}},
                {'type':'tool/result','data':{'turn':1,'message':{'source':{'callId':'c1'},'content':[]}}},
                {'type':'assistant/message','data':{'turn':1,'message':{'content':[{'type':'text','text':answer}]}}},
                {'type':'turn/end','data':{'turn':1,'reason':{'kind':'completed'}}}]
        class FakeApi:
            def rpc(self,method,args,**kwargs):
                if method=='session.create':return {'sessionId':'mock'}
                if method=='session.selectModel':return {'selected':{'provider':'mock','model':'mock'}}
                return {}
            def history(self,session):return events
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            record=runner.run_sample(FakeApi(),path,sample,PROJECT/'workdir-skills/molclaw-l1-workspace',30,'mock','mock','mock','prefix',expected_system='test-system',expected_mcp_tools=81,expected_skill_count=52)
            self.assertTrue(record['valid_output'])
            self.assertTrue(record['protocol_verified'])
            self.assertTrue(runner.materialize_scores(path,ROOT,[sample])['publishable'])

    def test_successful_schema_mismatch_is_rejected(self):
        from pipeline.cleaning.tests.test_views import semantic, deployment_tools
        from pipeline.cleaning.sft_views import semantic_to_qwen35_sft
        row=semantic();tools=deployment_tools('tool')
        tool=next(t for t in tools.tools if t['name']=='tool')
        tool['input_schema']={'type':'object','required':['path'],'properties':{'path':{'type':'string'}}}
        with self.assertRaisesRegex(ValueError,'successful tool call incompatible'):
            semantic_to_qwen35_sft(row,deployment_tools=tools,system_prompt='test')
        row['events'][1]['is_error']=True
        semantic_to_qwen35_sft(row,deployment_tools=tools,system_prompt='test')
