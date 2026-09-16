"""Two finite CPU-framework / GPU-model lanes; only model HTTP is SSH-forwarded."""
import fcntl,hashlib,json,os,pathlib,re,shlex,signal,subprocess,sys,time,uuid
P=pathlib.Path('/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe');W=P/'slime-wd';M=W/'dsh-molbench/pretrained_matrix';N=P.parent/'drug_wd/drug_pipe_regular_v1_20260908/experiments/fresh_four_0916d';NS='ailab-ma4agismall'
WP='/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd'
def terminate(signum,frame):raise InterruptedError('CPU queue terminated')
signal.signal(signal.SIGTERM,terminate)
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new']

def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(path)
def state(slot,event,**kw):
 d={'at':time.time(),'slot':slot,'event':event,**kw};save(N/f'queue{slot}_state.json',d);print(json.dumps(d),flush=True)
def worker_path(s):return str(s).replace('/home/sunxiangyu/slime_sxy','/root/slime_sxy')
def command(item,role,attempt=1):
 e=item['env'];name=f"av4-fresh-{item['label']}-0916d-{role}{attempt}"
 base=f"infra-aligned-v4-9b-{item['label']}-cpu-0916d";infra=f'{base}-{role}{attempt}';modelinfra=f'{base}-gpu{attempt}'
 env={k:e[k] for k in ['MODEL_DIR','MODEL_ID','MODEL_NAME','WORKSPACE_VARIANT']};env['MODEL_DIR']=worker_path(env['MODEL_DIR'])
 env.update(SETTING_LABEL=item['label'],INFRA_NAME=infra,MODEL_INFRA_NAME=modelinfra,RUN_NAME=f"aligned-v4-9b-{item['label']}-fresh-0916d",SAMPLE_IDS_FILE=worker_path(N/(item['label']+'-ids.json')),CPU_FRAMEWORK='1',CPU_RUN_TOKEN=item['token'],GPU_COUNT='2',TP_SIZE='2',EVAL_SEED='42',EVAL_MAX_WORKERS='1',EVAL_RECOVERY='0',SUITES='ms1 ms2 mo-opt mo-edit',NCCL_IB_DISABLE='1',DISTRIBUTED_JOB='true',OWNER_UID=str(os.getuid()),OWNER_GID=str(os.getgid()))
 driver=f'exec {WP}/dsh-molbench/pretrained_matrix/run_worker.sh' if role=='gpu' else f'exec /usr/bin/python3 {WP}/dsh-molbench/pretrained_matrix/run_cpu_eval.py'
 cmd=['rjob','submit',f'--name={name}',f'--metadata-name={name}',f'--namespace={NS}','--task-type=normal','--priority=9','--restart-policy=never','--preemptible=no','--enable-sshd','--image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529','--image-pull-policy=IfNotPresent','--mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu','--mount=gpfs://gpfs2/gpfs2-shared-public/huggingface:/root/slime_sxy/group-space/huggingface','--charged-group=ma4agismall_gpu','--private-machine=group',f'--gpu={2 if role=="gpu" else 0}',f'--cpu={32 if role=="gpu" else 8}',f'--memory={265000 if role=="gpu" else 32768}']
 for k,v in env.items():cmd+=['-e',k+'='+v]
 cmd+=['--','bash','-lc',driver]
 return name,W/'outputs'/infra,cmd

def validate(item,role):
 import yaml
 name,infra,cmd=command(item,role);idx=cmd.index('--');cmd.insert(idx,'--dry-run=true')
 r=subprocess.run(cmd,text=True,capture_output=True,check=True);text=r.stdout+r.stderr;(N/f'{name}.dry-run.log').write_text(text)
 o=yaml.safe_load(text[text.index('apiVersion:'):]);meta=o['metadata'];t=o['spec']['taskSpecs']['t0']['template'];c=t['spec']['containers'][0];env={x['name']:str(x['value']) for x in c['env']}
 assert meta['annotations']['volcano.brainpp.cn/priority']=='9';assert meta['labels']['quotagroup.brainpp.cn/quotagroup']=='ma4agismall_gpu';assert t['metadata']['labels']['privatemachine.brainpp.cn/privatemachine']=='group'
 limits=c['resources']['limits'];assert int(limits.get('nvidia.com/gpu',0))==(2 if role=='gpu' else 0);assert str(limits['cpu'])==('32' if role=='gpu' else '8');assert str(limits['memory'])==('265000Mi' if role=='gpu' else '32768Mi')
 assert c['command']==cmd[-3:];assert env['CPU_FRAMEWORK']=='1' and env['MODEL_DIR']==worker_path(item['env']['MODEL_DIR']);assert 'gpfs1/sdpdev-fs/sunxiangyu' in t['metadata']['annotations']['mount.brainpp.cn/gpfs']
 return {'job':name,'role':role,'priority':9,'resources':limits,'command':c['command'],'env':env,'image':c['image']}

def get(job):
 r=subprocess.run(['rjob','get',job,'--namespace',NS],text=True,capture_output=True,timeout=30);return r.stdout+r.stderr

def pair(item,slot,attempt):
 gpu,G,gcmd=command(item,'gpu',attempt)
 I=W/'outputs'/f"infra-aligned-v4-9b-{item['label']}-cpu-0916d-framework{attempt}"
 G.mkdir(parents=True,exist_ok=True);I.mkdir(parents=True,exist_ok=True)
 run=W/'outputs/dsh_molbench_evals'/f"aligned-v4-9b-{item['label']}-fresh-0916d"
 run.mkdir(parents=True,exist_ok=True)
 if not os.access(run,os.W_OK):raise PermissionError('CPU output directory is not writable: '+str(run))
 children=[];submitted=False;outcome=1
 def heartbeat():save(G/'cpu_control.json',{'token':item['token'],'state':'active','updated':time.time()})
 def pause(seconds=5):
  heartbeat();time.sleep(seconds)
 def target():
  deadline=time.monotonic()+7200
  while time.monotonic()<deadline:
   if cpu.poll() is not None:raise RuntimeError('CPU framework exited while GPU queued')
   s=get(gpu);m=re.search(r'replica ([^: ]+):',s)
   if "'failed': 1" in s or "'stopped': 1" in s:raise RuntimeError('RJob terminated: '+gpu)
   if m:
    t=m.group(1)+'.sunxiangyu+root.'+NS+'.pod@h.pjlab.org.cn'
    if subprocess.run(SSH+['-o','ClearAllForwardings=yes',t,'true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:return t
   pause()
  raise TimeoutError('Worker readiness: '+gpu)
 try:
  heartbeat();port=34000+slot
  env=dict(os.environ);env.update(item['env']);env.update(DRUG_PROJECT=str(P),INFRA_NAME=I.name,MODEL_INFRA_NAME=G.name,RUN_NAME=f"aligned-v4-9b-{item['label']}-fresh-0916d",SAMPLE_IDS_FILE=str(N/(item['label']+'-ids.json')),SETTING_LABEL=item['label'],CPU_RUN_TOKEN=item['token'],MODEL_PORT=str(port),DSH_PORT=str(3080+slot))
  cpu=subprocess.Popen(['/usr/bin/python3','-u',str(pathlib.Path(__file__).with_name('run_cpu_eval.py'))],env=env,stdout=(I/'controller.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True);children.append(cpu)
  deadline=time.monotonic()+1860
  while not (I/'framework.ready').exists():
   if cpu.poll() is not None or time.monotonic()>deadline:raise RuntimeError('CPU framework preflight failed')
   pause(1)
  state(slot,'cpu_framework_ready',setting=item['label'],cpu_pid=cpu.pid)
  with (G/'submit.log').open('w') as f:subprocess.run(gcmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  submitted=True;gt=target();save(G/'worker.json',{'target':gt,'job':gpu})
  forward=subprocess.Popen(SSH+['-F','/dev/null','-NT','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-L',f'127.0.0.1:{port}:127.0.0.1:30000',gt],stdout=(G/'forward.log').open('a'),stderr=subprocess.STDOUT);children.append(forward)
  state(slot,'running_or_starting',setting=item['label'],gpu_job=gpu,cpu_pid=cpu.pid,questions=item['question_count'],cpu_infra=str(I))
  while cpu.poll() is None:
   if forward.poll() is not None:raise RuntimeError('Model SSH channel exited')
   if (G/'worker.exit').exists():raise RuntimeError('GPU model process exited')
   pause(10)
  outcome=cpu.returncode;return outcome
 finally:
  save(G/'cpu_control.json',{'token':item['token'],'state':'done','updated':time.time(),'exit_code':outcome})
  for c in reversed(children):
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  if submitted:
   if outcome!=0 and "'active': 0" not in get(gpu):subprocess.run(['rjob','stop',gpu,'--namespace',NS],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   for _ in range(60):
    if "'active': 0" in get(gpu):break
    time.sleep(5)
   else:raise RuntimeError('Resource release unconfirmed: '+gpu)

def main():
 m=json.loads((N/'manifest.json').read_text());N.mkdir(exist_ok=True)
 if sys.argv[1]=='validate':
  save(N/'dry_run_validation.json',[validate(i,r) for i in m['settings'] if i['question_count'] for r in ['gpu']]);return
 slot=int(sys.argv[1]);assert slot in [1,2]
 with (N/f'queue{slot}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for item in m['settings']:
    if item['slot']!=slot or not item['question_count']:continue
    for path,digest in m['source_sha256'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest,path
    # Only pre-trajectory startup can be retried. Never repeat any task.
    journal=[]
    for attempt in range(1,4):
     try:rc=pair(item,slot,attempt)
     except Exception as exc:rc=1;journal.append({'attempt':attempt,'error':repr(exc)})
     else:journal.append({'attempt':attempt,'exit_code':rc})
     save(N/(item['label']+'-workers.json'),journal)
     if rc==0:break
     run=W/'outputs/dsh_molbench_evals'/f"aligned-v4-9b-{item['label']}-fresh-0916d"
     if any((run/'results').glob('*/record.json')) or (run/'run_manifest.json').exists():
      raise RuntimeError('Fresh single-pass run interrupted; no automatic trajectory restart: '+item['label'])
     if attempt==3:raise RuntimeError('Pre-trajectory startup exhausted')
     time.sleep(15)
    run=W/'outputs/dsh_molbench_evals'/f"aligned-v4-9b-{item['label']}-fresh-0916d"
    with (N/(item['label']+'-score.log')).open('w') as f:
     subprocess.run([str(W/'outputs/dsh_eval_runtime/mo-score-venv/bin/python'),str(pathlib.Path(__file__).with_name('score_fresh.py')),'--run-dir',str(run)],stdout=f,stderr=subprocess.STDOUT,check=True)
    state(slot,'setting_complete',setting=item['label'])
   state(slot,'complete')
  except BaseException as e:state(slot,'stopped_on_error',error=repr(e));raise
if __name__=='__main__':main()
