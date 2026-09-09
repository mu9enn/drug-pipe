"""Compare SFT deployment inputs with a real harness request and skill returns."""
import argparse
import json
import re
from pathlib import Path
from pipeline.benchmark_release import digest
from pipeline.output_contracts import normalize_task_prompt
from pipeline.cleaning.skill_native_augmentation import render_skill_result


def validate(transcript, sft, task_type):
    events = json.loads(transcript.read_text())
    header = next(e['data']['header'] for e in events if e['type'] == 'request/header')
    users = [e['data'] for e in events if e['type'] == 'user/message']
    catalog = next(e for e in users if e.get('source', {}).get('kind') == 'skill-catalog')
    question = next(e for e in users if e.get('source', {}).get('kind') == 'user')
    with sft.open() as f:
        row = json.loads(next(f))
    messages = row['messages']
    assert messages[0]['content'] == header['system']
    expected_tools = [{'name': t['function']['name'], 'description': t['function']['description'],
                       'parameters': t['function']['parameters']} for t in row['tools']]
    assert expected_tools == header['tools']
    def text(message):
        return ''.join(b.get('text', '') for b in message['content'] if b['type'] == 'text')
    assert messages[2]['content'] == text(catalog)
    prefix, task = text(question).split('\n\n# Task\n\n', 1)
    assert messages[1]['content'].split('\n\n# Task\n\n', 1)[0] == prefix
    assert normalize_task_prompt(task, task_type) == task
    skills = Path(__file__).resolve().parents[3] / 'workdir-skills/molclaw-l1-workspace/.agents/skills'
    checked = []
    for event in events:
        if event['type'] != 'tool/result':
            continue
        for result in event['data']['message']['content']:
            for block in result.get('content', []):
                body = block.get('text', '')
                match = re.match(r'<skill_content name="([^"]+)">', body)
                if not match:
                    continue
                normalized = re.sub(r'Base directory for this skill: .*?/\.agents/skills/',
                                    'Base directory for this skill: .agents/skills/', body)
                assert render_skill_result(match[1], skills) == normalized
                checked.append(match[1])
    assert checked, 'real skill return required'
    return {'passed': True, 'transcript': str(transcript.resolve()),
            'transcript_sha256': digest(transcript.read_bytes()), 'sft_sha256': digest(sft.read_bytes()),
            'tool_count': len(expected_tools), 'skill_returns': checked,
            'checks': ['system', 'complete_tool_schemas', 'catalog', 'user_prefix', 'task_contract', 'skill_bodies'],
            'allowed_runtime_difference': 'absolute workspace prefix in skill base directory'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transcript', type=Path, required=True)
    p.add_argument('--sft', type=Path, required=True)
    p.add_argument('--task', choices=['pf', 'ac', 'vs'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(validate(args.transcript, args.sft, args.task), indent=2) + '\n')
