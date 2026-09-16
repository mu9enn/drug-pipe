"""Single-source fence-only scoring; no historical answer selection."""
import json,pathlib,sys,argparse,hashlib
P=pathlib.Path('/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe');W=P/'slime-wd'
sys.path.insert(0,str(W/'dsh-molbench'))
import run_dsh_molbench as r
from score_format_sensitivity import extract
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=pathlib.Path,required=True);args=p.parse_args();run=args.run_dir
m=json.loads((run/'run_manifest.json').read_text());samples=r.load_samples(r.DEFAULT_MOLBENCH_ROOT,set(m['suites']),0);samples=[s for s in samples if s.task_id in m['sample_ids']]
assert len(samples)==165 and not any(s.suite=='ms3' for s in samples)
out=run/'scores_fence_only';sources=[]
for s in samples:
 f=run/'results'/s.task_id/'record.json';rec=r.load_record(run,s)
 if rec is None:rec={'task_id':s.task_id,'status':'missing','final_text':''}
 assert int(rec.get('attempt',1))==1,(s.task_id,rec.get('attempt'))
 if rec['status']=='completed' and not r.project_prediction(s,rec.get('final_text') or '')[1]:
  candidate=extract(rec.get('final_text') or '', 'fence_only')
  if candidate is not None and r.project_prediction(s,candidate)[1]:rec=dict(rec,final_text=candidate)
 r.write_json(out/'results'/s.task_id/'record.json',rec)
 sources.append({'task_id':s.task_id,'source':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None})
r.write_json(out/'provenance.json',sources)
r.materialize_scores(out,r.DEFAULT_MOLBENCH_ROOT,samples)
clean=[s for s in samples if s.suite=='mo-opt' and s.source_id!='48b3a4aa-d573-44f9-8fda-dfc4757a3aaa']
for s in clean:r.write_json(out/'mo_opt_task_disjoint/results'/s.task_id/'record.json',r.load_record(out,s))
r.materialize_scores(out/'mo_opt_task_disjoint',r.DEFAULT_MOLBENCH_ROOT,clean)
