import json,os,pathlib,subprocess,sys,tempfile,unittest
P=pathlib.Path(__file__).resolve().parents[2]
class Routing(unittest.TestCase):
 def test_actual_datapipe_launch_claude_and_dsh(self):
  for harness in ['claude','deepseek']:
   with self.subTest(harness=harness),tempfile.TemporaryDirectory() as td:
    d=pathlib.Path(td);(d/'skills').mkdir();(d/'skills/system.md').write_text('test')
    wrapper=d/'python';capture=d/'captured.json'
    wrapper.write_text('#!'+sys.executable+'\n'+'''import sys,os,json,pathlib
if sys.argv[1].endswith('/run_claude.py'):
 args=sys.argv[2:];p=args[args.index('--mcp-config-file')+1];cfg=json.loads(pathlib.Path(p).read_text());s=cfg['mcpServers']['molclaw-scp'];assert s['type']=='stdio';assert s['timeout']==14400000;assert os.environ['MOLCLAW_SCP_API_KEY']=='TEST_ONLY';assert 'TEST_ONLY' not in json.dumps(cfg);assert '--strict-mcp-config' in args
 pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(cfg));sys.exit(0)
os.execv(sys.executable,[sys.executable]+sys.argv[1:])
''');wrapper.chmod(0o755)
    env=dict(os.environ,ROOT_ENV_FILE=str(d/'absent'),MOLCLAW_SCP_MCP_URL='https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool',MOLCLAW_SCP_MCP_AUTH='TEST_ONLY',PYTHON_BIN=str(wrapper),CAPTURE=str(capture),AGENT_HARNESS=harness,SKILLS_ROOT=str(d/'skills'),SYSTEM_PROMPT_FILE='system.md')
    subprocess.run(['bash',str(P/'data-pipe/pipeline/claude_agent/launch_claude.sh'),'--run-dataset','--task','kg','--dataset-csv',str(d/'unused.csv'),'--skip-provider-switch'],env=env,check=True,capture_output=True,text=True)
    cfg=json.loads(capture.read_text());s=cfg['mcpServers']['molclaw-scp'];assert s['args']==[str(P/'runtime/molclaw_mcp.sh')]
    sys.path.insert(0,str(P/'data-pipe/pipeline/claude_agent'))
    from session_capture import _load_mcp_config
    converted=_load_mcp_config(capture)[0];assert converted['transport']=='stdio' and converted['toolCallTimeoutMs']==14400000
 def test_generic_dsh_template(self):
  s=(P/'slime-wd/dsh-molbench/pretrained_matrix/molclaw.cordis.patch.yml').read_text();assert 'transport: stdio' in s and 'runtime/molclaw_mcp.sh' in s and 'transport: streamable-http' not in s
 def test_legacy_http_config_is_private_and_preserves_other_servers(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('shared_config',P/'runtime/molclaw_mcp_config.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
  with tempfile.TemporaryDirectory() as td:
   f=pathlib.Path(td)/'old.json';original={'mcpServers':{'molclaw-scp':{'type':'http','url':'https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool','headers':{'SCP-HUB-API-KEY':'TEST_ONLY'}},'other':{'type':'http','url':'http://example.invalid/mcp'}}};f.write_text(json.dumps(original))
   new=h.adapt_existing_config(f,P)
   try:
    assert new!=f and new.stat().st_mode&0o777==0o600
    d=json.loads(new.read_text());assert d['mcpServers']['molclaw-scp']['type']=='stdio';assert d['mcpServers']['other']==original['mcpServers']['other'];assert json.loads(f.read_text())==original
    assert h.adapt_existing_config(new,P)==new
   finally:new.unlink()
if __name__=='__main__':unittest.main()
