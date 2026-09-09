"""Report within-environment paired differences, retaining every benchmark item."""
import argparse,json,random
from pathlib import Path

def summarize(root:Path,stamp:str,baseline_stamp:str|None=None)->dict:
    output={}
    for variant in ['l1-flat']:
        runs={kind:root/f'aligned-v4-9b-{kind}-{variant}-full-{(baseline_stamp or stamp) if kind == "original" else stamp}' for kind in ['original','sft']}
        manifests={kind:json.loads((path/'run_manifest.json').read_text()) for kind,path in runs.items()}
        left,right=manifests.values()
        for key in ['sample_ids','task_prompt_sha256','benchmark_release_sha256','tool_set_sha256','skill_tree_sha256','evaluation_contract','execution_settings']:
            if left[key]!=right[key]:raise ValueError(f'{variant}: incompatible {key}')
        lruntime = left['protocol_inputs']['runtime_audit']
        rruntime = right['protocol_inputs']['runtime_audit']
        if lruntime['server_args'] != rruntime['server_args']:
            raise ValueError(f'{variant}: different serving settings')
        lsnapshot, rsnapshot = lruntime['runtime_snapshot'], rruntime['runtime_snapshot']
        if lsnapshot['versions'] != rsnapshot['versions'] or lsnapshot['sources'] != rsnapshot['sources']:
            raise ValueError(f'{variant}: runtime or executable source changed between models')
        for name in ['tokenizer.json', 'tokenizer_config.json']:
            if lsnapshot['checkpoint_files'][name]['sha256'] != rsnapshot['checkpoint_files'][name]['sha256']:
                raise ValueError(f'{variant}: tokenizer differs')
        summaries={kind:json.loads((path/'evaluation_summary.json').read_text()) for kind,path in runs.items()}
        if not all(s['publishable'] for s in summaries.values()):raise ValueError('incomplete or unverified pair')
        paired = {}
        for suite, folder, metric in [('ms1', 'rdkit_bench', 'acc'), ('ms2', 'acnet_curated', 'score')]:
            predictions = {kind: {r['id']: r for r in json.loads((path/'preds'/folder/'all.json').read_text())}
                           for kind, path in runs.items()}
            if predictions['original'].keys() != predictions['sft'].keys():
                raise ValueError('per-question score coverage differs')
            items = [{'id': task_id, 'original': predictions['original'][task_id]['metrics'][metric],
                      'sft': predictions['sft'][task_id]['metrics'][metric]}
                     for task_id in sorted(predictions['original'])]
            differences = [item['sft'] - item['original'] for item in items]
            rng = random.Random(20260908)
            bootstrap = sorted(sum(rng.choices(differences, k=len(items))) / len(items) for _ in range(10000))
            paired[suite] = {'items': items, 'sft_only_correct': sum(d > 0 for d in differences),
                             'original_only_correct': sum(d < 0 for d in differences),
                             'paired_bootstrap_95_interval': [bootstrap[249], bootstrap[9749]],
                             'bootstrap_seed': 20260908, 'bootstrap_resamples': 10000}
        output[variant]={'runs':{k:str(v) for k,v in runs.items()},'summaries':summaries,
                         'paired_scientific_scores': paired,
                         'accuracy_deltas':{suite:summaries['sft']['metrics'][suite]['acc']-summaries['original']['metrics'][suite]['acc'] for suite in ['rdkit_bench_all','acnet_curated_all']}}
    path=root/f'aligned-v4-paired-{stamp}.json';path.write_text(json.dumps(output,indent=2)+'\n');return output
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--stamp',required=True);p.add_argument('--baseline-stamp');a=p.parse_args();print(json.dumps(summarize(a.runs_root,a.stamp,a.baseline_stamp),indent=2))
