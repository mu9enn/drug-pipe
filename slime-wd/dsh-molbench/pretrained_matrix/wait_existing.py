#!/usr/bin/env python3
"""Reuse only a completed current-protocol run; never rerun model failures."""
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);name,model=sys.argv[2:4];count=2 if int(sys.argv[4]) else 87
run=root/'dsh_molbench_evals'/name
for _ in range(2880):
    if (run/'evaluation_summary.json').exists():break
    roots=list(root.glob('infra-'+name+'*'))
    if roots:
        latest=max(roots,key=lambda p:p.stat().st_mtime_ns)
        for marker in ('orchestrator.exit','worker.exit'):
            p=latest/marker
            if p.exists() and p.read_text().strip()!='0':raise SystemExit('Existing run failed: '+str(p))
    time.sleep(60)
m=json.loads((run/'run_manifest.json').read_text());s=json.loads((run/'evaluation_summary.json').read_text())
assert s['publishable'] and s['sample_count']==count
assert m['model']==model
assert m['evaluation_contract']['contract_version']=='molbench_answer_contract_v4'
print('Reused current aligned run:',name,flush=True)
