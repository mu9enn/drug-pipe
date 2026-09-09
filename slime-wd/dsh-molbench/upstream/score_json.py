"""Score strict terminal answers against every selected frozen MolBench question.

Usage: python molbench/eval/score_json.py --suite ms1 --predictions finals.json --output results
finals.json maps molbench_ms1_001 etc. to raw final assistant strings. Missing IDs
remain in the denominator. Extra IDs are rejected. This entrypoint does not run agents.
"""
import argparse
import csv
import json
from pathlib import Path
from output_contracts import normalize_final_answer, strict_json_loads, task_constraints
from eval_runner import ACNetCuratedEval, MolbenchVsEval, RdkitBenchEval


def score(suite, predictions, output, data_root):
    column, task, count, evaluator, answer_key = {
        'ms1': ('prompt', 'pf', 50, RdkitBenchEval, 'selected_smiles'),
        'ms2': ('question', 'ac', 37, ACNetCuratedEval, 'answer_smiles'),
        'ms3': ('questions', 'vs', 25, MolbenchVsEval, 'ranked_smiles'),
    }[suite]
    folder = 'molbench-' + suite.replace('ms', 'ms-')
    with (data_root / folder / (folder + '.csv')).open(newline='', encoding='utf-8-sig') as f:
        questions = list(csv.DictReader(f))
    assert len(questions) == count
    ids = {f'molbench_{suite}_{i:03d}' for i in range(1, count + 1)}
    if not isinstance(predictions, dict) or set(predictions) - ids:
        raise ValueError('prediction keys must belong to the selected suite')
    records, audit = [], []
    for index, question in enumerate(questions, 1):
        task_id = f'molbench_{suite}_{index:03d}'
        constraints = task_constraints(question[column], task)
        text = predictions.get(task_id, '')
        value, format_valid, conforming, error = None, False, False, None
        try:
            normalize_final_answer(text, task)
            format_valid = True
            value = json.loads(normalize_final_answer(text, task, constraints=constraints))[answer_key]
            conforming = True
        except (ValueError, TypeError) as exc:
            error = str(exc)
        audit.append({'id': task_id, 'missing': task_id not in predictions,
                      'json_format_valid': format_valid, 'task_conforming': conforming, 'error': error})
        if suite == 'ms3':
            record = {'index': index, 'answer': json.loads(question['answer']),
                      'candidates': list(constraints.candidates), 'json_results': {'ranking': value or []}}
        else:
            projected = ('\n'.join(value) if suite == 'ms1' else value) if conforming else ''
            record = {'id': task_id, 'gt': question['answer'], 'json_results': {'output': projected}}
            if suite == 'ms2':
                record.update(s1=constraints.candidates[0], s2=constraints.candidates[1])
        records.append(record)
    preds = output / 'preds'
    preds.mkdir(parents=True, exist_ok=True)
    (preds / 'all.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    metrics = evaluator().run(str(preds), str(output), '')
    report = {'suite': suite, 'sample_count': count,
              'json_format_rate': sum(r['json_format_valid'] for r in audit) / count,
              'task_constraint_rate': sum(r['task_conforming'] for r in audit) / count,
              'missing_count': sum(r['missing'] for r in audit), 'metrics': metrics, 'records': audit}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['ms1', 'ms2', 'ms3'], required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score(args.suite, strict_json_loads(args.predictions.read_text()), args.output,
          Path(__file__).resolve().parents[1] / 'data')
