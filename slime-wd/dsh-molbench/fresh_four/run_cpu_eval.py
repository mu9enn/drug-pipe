"""Run DSH in the CPU workspace with its native Bubblewrap sandbox. MCP uses the approved HTTP proxy directly."""
import json,os,pathlib,signal,subprocess,sys,time,urllib.request
P=pathlib.Path(os.environ['DRUG_PROJECT']);W=P/'slime-wd';M=W/'dsh-molbench/pretrained_matrix'
I=W/'outputs'/os.environ['INFRA_NAME'];G=W/'outputs'/os.environ['MODEL_INFRA_NAME'];R=W/'outputs/dsh_molbench_evals'/os.environ['RUN_NAME'];I.mkdir(parents=True,exist_ok=True)
def terminate(signum,frame):raise InterruptedError('CPU evaluation terminated')
signal.signal(signal.SIGTERM,terminate)
children=[];rc=1
model_port=int(os.environ['MODEL_PORT']);dsh_port=int(os.environ['DSH_PORT'])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def log(event,**kw):
 print(json.dumps({'event':event,'time':time.time(),**kw}),flush=True)
def check_lease():
 state=json.loads((G/'cpu_control.json').read_text())
 if state.get('token')!=os.environ['CPU_RUN_TOKEN'] or state.get('state')!='active' or time.time()-state['updated']>600:
  raise RuntimeError('CPU controller lease expired or completed')
def wait(test,label,seconds=1800):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  check_lease()
  for p in children:
   if p.poll() is not None:raise RuntimeError('Child process exited during '+label)
  try:
   if test():return
  except OSError:pass
  time.sleep(3)
 raise TimeoutError(label+' readiness deadline')
def http(url):
 with opener.open(url,timeout=3) as r:return r.status==200
try:
 os.umask(0o002)
 launcher=W/'outputs/dsh_eval_runtime/bubblewrap/root/usr/bin/bwrap'
 import tempfile
 with tempfile.TemporaryDirectory(dir=I) as td:
  q=pathlib.Path(td).resolve();(q/'work').mkdir()
  subprocess.run([str(launcher),'--ro-bind','/','/','--dev','/dev','--proc','/proc','--die-with-parent','--tmpfs','/tmp','--bind',str(q/'work'),str(q/'work'),'--','bash','-c','echo ok > "$1/ok"; test "$(cat "$1/ok")" = ok && ! touch "$2/denied" 2>/dev/null','_',str(q/'work'),str(q)],check=True,stdout=(I/'sandbox_behavior.log').open('w'),stderr=subprocess.STDOUT)
 (I/'sandbox_preflight.log').write_text('Bubblewrap: workspace write allowed; outside write denied. Native DSH backend.\n')
 (I/'sandbox.ready').touch();log('sandbox_ready')
 secret=pathlib.Path.home()/'.dsh/molclaw.env';wait(lambda:secret.is_file(),'CPU credential')
 H=pathlib.Path.home()/'.cache/drug-pipe-cpu-eval'/I.name/'dsh-home';H.mkdir(parents=True,exist_ok=True)
 (I/'dsh_home_path.txt').write_text(str(H)+'\n')
 env=dict(os.environ,DSH_HOME=str(H),SLIME_LOCAL_API_KEY='local',NODE_USE_ENV_PROXY='1',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',PYTHONPATH=str(W/'outputs/dsh_eval_runtime/python'),PATH=str(W/'outputs/dsh_eval_runtime/node-v24.19.0/bin')+':'+str(launcher.parent)+':'+os.environ['PATH'])
 proxy='http://httpproxy-headless.kubebrain.svc.pjlab.local:3128'
 for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:env[k]=proxy
 settings=(M/'settings.template.yaml').read_text().replace('__MODEL_ID__',env['MODEL_ID']).replace('__MODEL_NAME__',env['MODEL_NAME']).replace(':30000/v1',f':{model_port}/v1')+'\nagent-loop:\n  maxParallelToolCalls: 1\n'
 (H/'settings.yaml').write_text(settings)
 config=[{'insert':[{'id':'mcp-molclaw-scp','name':'@deepseek-ai/dsh-mcp-client','config':{
 'serverName':'molclaw-scp','transport':'stdio','command':'bash','args':[str(W/'molclaw-mcp-relay/polling-adapter/run_mcp.sh')],
 'env':{'DRUG_PROJECT':str(P),'SECRET_FILE':str(secret),'HTTP_PROXY':proxy,'HTTPS_PROXY':proxy,'http_proxy':proxy,'https_proxy':proxy,'MOLCLAW_POLL_INTERVAL_SECONDS':'300','MOLCLAW_ADAPTER_LOG_FILE':str(I/'adapter.jsonl')},
 'toolCallTimeoutMs':14400000,'failOnStartupError':True}}]}]
 (H/'cordis.patch.yml').write_text(json.dumps(config,indent=2))
 (I/'route.json').write_text(json.dumps({'framework':'CPU workspace with native Bubblewrap','model':f'http://127.0.0.1:{model_port}/v1 over model-only SSH forward','mcp_proxy':proxy,'molclaw_relay_used':False,'recovery_policy':'single_attempt_no_trajectory_retry','trajectory_budget_seconds':14400,'trajectory_budget_retry':False},indent=2))
 node=W/'outputs/dsh_eval_runtime/node-v24.19.0/bin/node'
 dsh=subprocess.Popen([str(node),'--import',str(W/'molclaw-mcp-relay/install_mcp_headers_timeout.mjs'),'--import','tsx/esm','apps/cli/src/bin.ts','web','--port',str(dsh_port),'--no-open'],cwd=W/'deepseek-harness',env=env,stdout=(I/'dsh.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True);children.append(dsh)
 wait(lambda:http(f'http://127.0.0.1:{dsh_port}/'),'CPU DSH',1800);log('dsh_ready',pid=dsh.pid)
 (I/'framework.ready').touch()
 wait(lambda:(G/'model.ready').exists(),'GPU model',9000)
 wait(lambda:http(f'http://127.0.0.1:{model_port}/v1/models'),'CPU-to-GPU model forward')
 skill=P/'workdir-skills/molclaw-l1-workspace' if env['WORKSPACE_VARIANT']=='l1-flat' else G/'workspace_template';count=52 if env['WORKSPACE_VARIANT']=='l1-flat' else 68
 cmd=[sys.executable,'-u',str(W/'dsh-molbench/run_dsh_molbench.py'),'--dsh-url',f'http://127.0.0.1:{dsh_port}','--rollout-only','--sample-ids-file',env['SAMPLE_IDS_FILE'],'--run-dir',str(R),'--task-timeout-sec','14400','--max-workers','1','--model-provider','slime-local','--model-id',env['MODEL_ID'],'--agent-preset','molclaw-v8-eval','--skill-source',str(skill),'--required-mcp-tools','81','--required-skill-count',str(count),'--system-prompt-file',str(P/'data-pipe/pipeline/cleaning/prompts/qwen35_system.md'),'--tokenizer-config',env['MODEL_DIR']+'/tokenizer_config.json','--runtime-audit-file',str(G/'runtime_audit.json')]
 for suite in ['ms1','ms2','mo-opt','mo-edit']:cmd+=['--suite',suite]
 if (R/'run_manifest.json').exists():raise RuntimeError('Fresh evaluation refuses an existing manifest')
 (I/'cpu.ready').touch();log('runner_start',run=str(R))
 runner=subprocess.Popen(cmd,cwd=W,env=env,stdout=(I/'rollout.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True);children.append(runner)
 (I/'processes.json').write_text(json.dumps({'controller':os.getpid(),'dsh':dsh.pid,'runner':runner.pid}))
 while runner.poll() is None:
  check_lease()
  if dsh.poll() is not None:raise RuntimeError('CPU DSH exited')
  time.sleep(3)
 rc=runner.returncode
 (I/'runner.exit').write_text(str(rc))
 if rc==0:
  subprocess.run([sys.executable,str(W/'dsh-molbench/audit_protocol.py'),'--run-dir',str(R),'--system-prompt-file',str(P/'data-pipe/pipeline/cleaning/prompts/qwen35_system.md'),'--expected-mcp-tools','81','--expected-skill-count',str(count)],env=env,check=True,stdout=(I/'protocol_audit.log').open('w'),stderr=subprocess.STDOUT)
  (I/'rollout.complete').touch()
except BaseException as e:
 log('cpu_evaluation_error',error=repr(e));rc=1
finally:
 for c in reversed(children):
  if c.poll() is None:
   os.killpg(c.pid,signal.SIGTERM)
   try:c.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
 (I/'cpu.exit').write_text(str(rc));log('cpu_exit',code=rc)
sys.exit(rc)
