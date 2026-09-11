import json
import unittest
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import run_dsh_molbench as runner


def call(cid, args, result, *, is_error=False):
    return [
        {'type': 'tool/call', 'data': {'callId': cid, 'name': 'mcp__test', 'arguments': args}},
        {'type': 'tool/result', 'seq': 2, 'data': {'message': {
            'source': {'callId': cid}, 'content': [{'isError': is_error, 'content': [{'text': result}]}]}}},
    ]


class InfrastructureRecoveryTest(unittest.TestCase):
    def test_same_call_recovery_and_changed_arguments(self):
        failed = call('a', '{"x":1,"y":2}', 'Error: fetch failed', is_error=True)
        recovered = call('b', '{"y":2,"x":1}', '{"status":"success"}')
        self.assertEqual(runner.unresolved_infrastructure(failed + recovered, {}), [])
        other = call('c', '{"x":2,"y":2}', '{"status":"success"}')
        self.assertEqual(len(runner.unresolved_infrastructure(failed + other, {})), 1)
        self.assertEqual(len(runner.unresolved_infrastructure(failed + recovered + failed, {})), 1)

    def test_business_error_with_false_outer_flag(self):
        events = call('a', '{}', json.dumps({'status': 'error', 'msg': 'CUDA out of memory'}))
        self.assertEqual(len(runner.unresolved_infrastructure(events, {})), 1)
        bad_input = call('b', '{}', '{"status":"error","msg":"input file not found"}')
        self.assertEqual(runner.unresolved_infrastructure(bad_input, {}), [])
        self.assertEqual(len(runner.unresolved_infrastructure(events + bad_input, {})), 1)

    def test_only_actual_errors_not_reasoning(self):
        events = [{'type': 'assistant/message', 'data': {'text': 'fetch failed CUDA out of memory'}}]
        self.assertEqual(runner.unresolved_infrastructure(events, {}), [])
        for record in [{'status': 'failed', 'turn_reason': {'kind': 'max-tokens'}},
                       {'status': 'completed', 'valid_output': False}]:
            record['failure_class'] = runner.failure_class(record)
            self.assertFalse(runner.retry_eligible(record))
        auth = {'status': 'failed', 'recovery_policy': 'infra_v1',
                'turn_reason': {'kind': 'error', 'error': {'code': 'SERVER', 'message': 'HTTP 401'}}}
        self.assertEqual(runner.unresolved_infrastructure([], auth), [])
        self.assertEqual(runner.failure_class(auth), 'unclassified_failure')
        record = {'status': 'completed', 'attempt': 1, 'unresolved_infra': [{'error': 'fetch failed'}]}
        record['failure_class'] = runner.failure_class(record)
        self.assertTrue(runner.retry_eligible(record))

    def test_bounded_retries_keep_last_not_best(self):
        records = iter([{'attempt': i, 'failure_class': 'retryable_infra', 'prediction': str(4-i)}
                        for i in range(1, 4)])
        archived, delays = [], []
        result = runner.recover_task(lambda: next(records), archived.append, delays.append)
        self.assertEqual(result['attempt'], 3)
        self.assertEqual(result['prediction'], '1')
        self.assertEqual([r['attempt'] for r in archived], [1, 2])
        self.assertEqual(delays, [60, 180])
        clean = iter([{'attempt': 2, 'failure_class': None, 'valid_output': False}])
        self.assertFalse(runner.recover_task(lambda: next(clean), archived.append, delays.append)['valid_output'])
        # Resume at attempt 3 does not gain a new budget.
        last = {'attempt': 3, 'failure_class': 'retryable_infra'}
        self.assertIs(runner.recover_task(lambda: last, self.fail, self.fail), last)


class WorkerRecoveryTest(unittest.TestCase):
    def test_worker_budget_survives_resume_and_requires_release(self):
        spec = importlib.util.spec_from_file_location('recovery_submit', Path(__file__).parent / 'pretrained_matrix/submit_recover.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'outputs').mkdir()
            submissions = []

            def submit(command, env):
                submissions.append(env['JOB_NAME'])
                infra = root / 'outputs' / env['INFRA_NAME']
                infra.mkdir()
                (infra / 'worker_restart_required').touch()
                return SimpleNamespace(returncode=1)

            with patch.object(module, 'ROOT', root), patch.dict(module.os.environ,
                    {'JOB_NAME': 'test', 'RUN_NAME': 'run', 'INFRA_NAME': 'infra'}), \
                    patch.object(module.subprocess, 'run', side_effect=submit), \
                    patch.object(module, 'released', return_value=True), patch.object(module.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                    module.main()
                self.assertEqual(submissions, ['test-w1', 'test-w2', 'test-w3'])
                with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                    module.main()
                self.assertEqual(len(submissions), 3)

    def test_format_scoring_keeps_manifest_subset(self):
        import score_format_sensitivity
        sample = runner.load_samples(runner.DEFAULT_MOLBENCH_ROOT, {'ms3'}, 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.write_json(root / 'run_manifest.json', {
                'suites': ['ms3'], 'limit_per_suite': 0, 'sample_ids': [sample.task_id],
            })
            runner.write_json(root / 'results' / sample.task_id / 'record.json', {
                'status': 'completed', 'protocol_verified': True,
                'final_text': json.dumps({'ranked_smiles': json.loads(sample.answer)[:3], 'evidence': []}),
            })
            result = score_format_sensitivity.score(root, runner.DEFAULT_MOLBENCH_ROOT)
            for policy in result.values():
                self.assertEqual(policy['sample_count'], 1)
                self.assertEqual(next(iter(policy['metrics'].values()))['n_samples'], 1)

    def test_unresolved_completed_answer_still_scores(self):
        sample = next(s for s in runner.load_samples(runner.DEFAULT_MOLBENCH_ROOT, {'ms3'}, 1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.write_json(root / 'results' / sample.task_id / 'record.json', {
                'status': 'completed', 'attempt': 3, 'protocol_verified': True,
                'failure_class': 'retryable_infra', 'retry_exhausted': True,
                'unresolved_infra': [{'error': 'fetch failed'}],
                'final_text': json.dumps({'ranked_smiles': json.loads(sample.answer)[:3], 'evidence': []}),
            })
            summary = runner.materialize_scores(root, runner.DEFAULT_MOLBENCH_ROOT, [sample])
            self.assertEqual(next(iter(summary['metrics'].values()))['hit_at_3'], 1)
            self.assertEqual(summary['retry_exhausted_count'], 1)
            self.assertFalse(summary['publishable'])


if __name__ == '__main__':
    unittest.main()
