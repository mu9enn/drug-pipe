import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import run_dsh_molbench as runner
from score_format_sensitivity import extract
from pipeline.output_contracts import normalize_task_prompt, task_constraints


class ExtendedSuitesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = runner.load_samples(runner.DEFAULT_MOLBENCH_ROOT, {'ms3', 'mo-opt', 'mo-edit'}, 0)

    def test_frozen_counts_and_idempotent_prompts(self):
        self.assertEqual(Counter(s.suite for s in self.samples), {'ms3': 25, 'mo-opt': 39, 'mo-edit': 39})
        for sample in self.samples:
            self.assertEqual(normalize_task_prompt(sample.prompt, runner.SUITE_TASKS[sample.suite]), sample.prompt)
        self.assertEqual(len({s.task_id for s in self.samples}), 103)

    def test_mo_requires_upstream_field_and_new_molecules_are_allowed(self):
        for suite, field in [('mo-opt', 'Final Target Molecule'), ('mo-edit', 'output')]:
            sample = next(s for s in self.samples if s.suite == suite)
            text = json.dumps({field: 'CCO'})
            self.assertEqual(runner.project_prediction(sample, text), ('CCO', True, None))
            for bad in [json.dumps({field: []}), json.dumps({field: 'CCO', 'extra': 1}),
                        '{"' + field + '":"CCO",}', '```json\n' + text + '\n```']:
                self.assertFalse(runner.project_prediction(sample, bad)[1])
            self.assertEqual(extract('Explanation\n```json\n' + text + '\n```', 'fence_only'), text)
            self.assertEqual(extract('Explanation\n' + text, 'unique_json_object'), text)
            self.assertIsNone(extract(text + '\n' + text, 'unique_json_object'))

    def test_ms3_accepts_partial_repeated_and_outside_candidates(self):
        sample = next(s for s in self.samples if s.suite == 'ms3')
        candidates = task_constraints(sample.prompt, 'vs').candidates
        self.assertTrue(runner.project_prediction(sample, json.dumps({'ranked_smiles': candidates, 'evidence': []}))[1])
        for ranking in [[], candidates[:3], [candidates[0], candidates[0], 'outside'], candidates[:-1]]:
            self.assertTrue(runner.project_prediction(sample, json.dumps({'ranked_smiles': ranking, 'evidence': []}))[1])
        self.assertFalse(runner.project_prediction(sample, '{"ranked_smiles":"not a list"}')[1])
        self.assertFalse(runner.project_prediction(sample, '{"ranked_smiles":[]}')[1])
        self.assertFalse(runner.project_prediction(sample, '{"ranked_smiles":[],"evidence":{},"extra":1}')[1])

    def test_ms3_scores_original_top3_without_filtering(self):
        sample = next(s for s in self.samples if s.suite == 'ms3')
        hit = json.loads(sample.answer)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ranking, expected in [(['outside', hit, hit], 2 / 3),
                                      (['outside'] * 3 + [hit], 0.0),
                                      ([hit], 1 / 3)]:
                runner.write_json(root / 'results' / sample.task_id / 'record.json',
                                  {'status': 'completed', 'final_text': json.dumps({'ranked_smiles': ranking, 'evidence': []})})
                summary = runner.materialize_scores(root, runner.DEFAULT_MOLBENCH_ROOT, [sample])
                metric = next(iter(summary['metrics'].values()))
                self.assertAlmostEqual(metric['top3_hit_rate'], expected)

    def test_upstream_scoring_full_denominators_and_known_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sample in self.samples:
                if sample.suite == 'ms3':
                    candidates = list(task_constraints(sample.prompt, 'vs').candidates)
                    answer = json.loads(sample.answer)
                    # Frozen answer uses a top3 list; remaining candidates are arbitrary for this scorer fixture.
                    top = answer[:3]
                    text = json.dumps({'ranked_smiles': top + [s for s in candidates if s not in top], 'evidence': []})
                elif sample.suite == 'mo-edit':
                    text = json.dumps({'output': sample.metadata['reference']})
                else:
                    text = json.dumps({'Final Target Molecule': sample.source_molecule})
                runner.write_json(root / 'results' / sample.task_id / 'record.json',
                                  {'status': 'completed', 'final_text': text})
            summary = runner.materialize_scores(root, runner.DEFAULT_MOLBENCH_ROOT, self.samples)
            self.assertEqual(summary['sample_count'], 103)
            self.assertEqual(summary['valid_output_count'], 103)
            for key, value in summary['metrics'].items():
                if key.startswith('molbench-mo-opt'):
                    self.assertEqual(value['improvement']['mean'], 0)
            missing = runner.materialize_scores(root / 'missing', runner.DEFAULT_MOLBENCH_ROOT, self.samples)
            self.assertEqual(missing['sample_count'], 103)
            self.assertEqual(missing['valid_output_count'], 0)
            self.assertFalse(missing['publishable'])


if __name__ == '__main__':
    unittest.main()
