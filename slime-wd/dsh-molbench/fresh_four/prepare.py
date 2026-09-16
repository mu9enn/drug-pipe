import json,pathlib,uuid,hashlib,sys,collections
P=pathlib.Path('/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe');W=P/'slime-wd';E=P.parent/'drug_wd/drug_pipe_regular_v1_20260908/experiments';N=E/'fresh_four_0916d';D=pathlib.Path(__file__).parent
N.mkdir(exist_ok=False)
old=json.loads((E/'cpu_questions_0916a/manifest.json').read_text());orig=next(x for x in old['settings'] if x['label']=='orig-hier')
settings=[]
for label,slot in [('orig-l1',1),('orig-hier',2),('sft-l1',1),('lr2e6-l1',2)]:
 src=orig if label=='orig-l1' else next(x for x in old['settings'] if x['label']==label)
 env={k:src['env'][k] for k in ['MODEL_DIR','MODEL_ID','MODEL_NAME','WORKSPACE_VARIANT']}
 if label=='orig-l1':env['WORKSPACE_VARIANT']='l1-flat'
 env.update(EVAL_RECOVERY='0',SUITES='ms1 ms2 mo-opt mo-edit',EVAL_SEED='42',GPU_COUNT='2',TP_SIZE='2',EVAL_MAX_WORKERS='1',LIMIT_PER_SUITE='0')
 settings.append({'label':label,'slot':slot,'question_count':165,'env':env,'token':str(uuid.uuid4())})
sys.path.insert(0,str(W/'dsh-molbench'));import run_dsh_molbench as r
samples=r.load_samples(r.DEFAULT_MOLBENCH_ROOT,{'ms1','ms2','mo-opt','mo-edit'},0);assert len(samples)==165
ids=[s.task_id for s in samples]
for item in settings:
 (N/(item['label']+'-ids.json')).write_text(json.dumps(ids,indent=2))
 run=W/'outputs/dsh_molbench_evals'/f"aligned-v4-9b-{item['label']}-fresh-0916d";assert not run.exists()
files=[D/'cpu_queue.py',D/'run_cpu_eval.py',D/'score_fresh.py',W/'dsh-molbench/run_dsh_molbench.py',W/'dsh-molbench/pretrained_matrix/run_worker.sh',W/'dsh-molbench/pretrained_matrix/model_lease.py',W/'molclaw-mcp-relay/polling-adapter/stdio_adapter.mjs',P/'data-pipe/pipeline/output_contracts.py']
m={'settings':settings,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'max_concurrent_questions':2,'gpus_per_model_worker':2,'sample_counts':dict(collections.Counter(s.suite for s in samples)),'trajectory_attempts':1,'trajectory_retry':False,'scientific_score_selection':False,'reuse_historical_answers':False,'tool_transport_retry':True,'trajectory_budget_seconds':14400,'scoring_policy':'fence_only; single source; all 165 tasks in denominator','additional_report':'All settings additionally report MO-opt 38 task-disjoint samples excluding known same-source QED overlap','mcp_route':'CPU development workspace -> approved HTTP proxy -> SCP; GPU model-only','queue_policy':'two lanes: orig-l1 then sft-l1; orig-hier then lr2e6-l1','failure_policy':'No trajectory retry; failed/missing tasks retained; startup retry only before run manifest or any task record exists'}
(N/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps({'directory':str(N),'counts':m['sample_counts'],'settings':[i['label'] for i in settings]}))
