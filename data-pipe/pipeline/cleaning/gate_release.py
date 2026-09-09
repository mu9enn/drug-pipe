"""Validate real Qwen tokenization and loss masks; retain complete trajectories only."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from pipeline.benchmark_release import digest


def update_release_manifest(root: Path, report: dict) -> None:
    path = root / 'release_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['context_gate'] = 'passed'
    manifest['training_count'] = report['accepted_count']
    manifest['training_manifest_sha256'] = digest((root/'training/context_gate_manifest.json').read_bytes())
    for name in ['training/probes.jsonl', 'training/qwen35_sft_train.jsonl']:
        manifest.setdefault('files', {})[name] = digest((root/name).read_bytes())
    path.write_text(json.dumps(manifest, indent=2) + '\n')


def verify_mask_roles(tokenizer, masker) -> dict:
    """Check observable supervision boundaries with the actual checkpoint template."""
    markers = {'SYSTEM_MASK_PROBE': 0, 'USER_MASK_PROBE': 0, 'OBSERVATION_MASK_PROBE': 0,
               'REASONING_MASK_PROBE': 1, 'DECISION_MASK_PROBE': 1, 'FINAL_MASK_PROBE': 1}
    messages = [
        {'role': 'system', 'content': 'SYSTEM_MASK_PROBE'},
        {'role': 'user', 'content': 'USER_MASK_PROBE'},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'REASONING_MASK_PROBE',
         'tool_calls': [{'id': 'probe_call', 'type': 'function', 'function':
                        {'name': 'probe', 'arguments': {'text': 'DECISION_MASK_PROBE'}}}]},
        {'role': 'tool', 'tool_call_id': 'probe_call', 'name': 'probe', 'content': 'OBSERVATION_MASK_PROBE'},
        {'role': 'assistant', 'content': 'FINAL_MASK_PROBE', 'reasoning_content': 'Conclude.'},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens, mask = masker.get_loss_mask(messages)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    for marker, expected in markers.items():
        start = text.index(marker); end = start + len(marker)
        selected = [mask[i] for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not selected or set(selected) != {expected}:
            raise ValueError(f'loss mask role mismatch: {marker} {selected}')
    return {'ok': True, 'roles': markers, 'tokens': len(tokens)}


def gate(root: Path, model: Path, max_tokens: int = 245760) -> dict:
    project = Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(project/'slime-wd/slime'))
    from transformers import AutoTokenizer
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from drug_agent.data.validate_sft_messages import validate_record
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True)
    masker = MultiTurnLossMaskGenerator(tokenizer, 'qwen3_5')
    role_check = verify_mask_roles(tokenizer, masker)
    from pipeline.cleaning.deployment_tools import load_deployment_tool_set
    from pipeline.cleaning.skill_native_augmentation import render_catalog, skill_catalog
    expected_tools = load_deployment_tool_set(project/'data-pipe/configs/dsh_molclaw_tool_set.json').qwen_tools()
    expected_system = (project/'data-pipe/pipeline/cleaning/prompts/qwen35_system.md').read_text().strip()
    expected_catalog = render_catalog(skill_catalog(project/'workdir-skills/molclaw-l1-workspace/.agents/skills'))
    source = root/'sft/qwen35_sft.jsonl'
    target = root/'training';target.mkdir(exist_ok=True)
    measurements, excluded, probes = [], [], []
    output = target/'qwen35_sft_train.jsonl'
    with output.open('w') as f, source.open() as rows:
        for index,line in enumerate(rows,1):
            row=json.loads(line)
            if (row['tools'] != expected_tools or row['messages'][0]['content'] != expected_system or
                    row['messages'][2]['content'] != expected_catalog):
                raise ValueError(f"{row['id']}: training inputs differ from the deployment protocol")
            findings=validate_record(row)
            if findings: raise ValueError(f"{row['id']}: {findings}")
            tokens,mask=masker.get_loss_mask(row['messages'],row['tools'])
            if len(tokens)!=len(mask) or not any(mask):raise ValueError('invalid loss mask')
            item={'id':row['id'],'tokens':len(tokens),'trainable_tokens':sum(mask)}
            measurements.append(item)
            if len(tokens)>max_tokens:excluded.append(item)
            else:
                f.write(line);probes.append((len(tokens),row))
            if index%25==0: print(f'validated {index} trajectories',flush=True)
    if not probes:
        raise ValueError("No complete trajectory fits the token budget; source data was preserved")
    probes.sort(key=lambda pair:pair[0])
    selected=[probes[i][1] for i in sorted({0,len(probes)//2,min(len(probes)-1,int(len(probes)*.95)),len(probes)-1})]
    with (target/'probes.jsonl').open('w') as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'ok':True,'model':str(model.resolve()),'tokenizer_sha256':digest((model/'tokenizer.json').read_bytes()),
            'tokenizer_config_sha256':digest((model/'tokenizer_config.json').read_bytes()),
            'input_count':len(measurements),'accepted_count':len(measurements)-len(excluded), 'excluded':excluded,
            'max_tokens':max_tokens,'record_lengths':measurements, 'source_sha256':digest(source.read_bytes()),
            'output':str(output.resolve()),'output_sha256':digest(output.read_bytes()),
            'policy':'whole_trajectories_no_truncation','loss_mask':'qwen3_5','role_check':role_check}
    (target/'context_gate_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    update_release_manifest(root, report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    r=gate(a.release,a.model);print(json.dumps({k:v for k,v in r.items() if k!='record_lengths'},indent=2))
