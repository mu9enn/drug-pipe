"""Write credential-free MCP routing config; credentials stay in inherited env."""
import argparse,json,os
from pathlib import Path

def build_config(project_root: Path, name='molclaw-scp', timeout=14400000):
    launcher=project_root/'runtime/molclaw_mcp.sh'
    if not launcher.is_file():raise FileNotFoundError(launcher)
    if timeout<1000:raise ValueError('tool timeout must be >= 1000 ms')
    return {'mcpServers':{name:{'type':'stdio','command':'bash','args':[str(launcher)],'timeout':timeout,
        'env':{'DRUG_PROJECT':str(project_root),'MOLCLAW_POLL_INTERVAL_SECONDS':'300'}}}}

def temporary_config(project_root: Path) -> Path:
    """Default direct runner calls to the shared adapter, not ambient MCP config."""
    import atexit, tempfile
    fd, name = tempfile.mkstemp(prefix='molclaw_default_', suffix='.json')
    result = Path(name)
    with os.fdopen(fd, 'w') as stream:
        json.dump(build_config(project_root), stream)
    atexit.register(lambda: result.unlink(missing_ok=True))
    return result

def adapt_existing_config(config: Path, project_root: Path) -> Path:
    """Route legacy SCP HTTP entries through stdio; leave other servers alone."""
    import atexit, tempfile
    from urllib.parse import urlparse
    payload=json.loads(config.read_text())
    changed=False
    for name,server in payload.get('mcpServers',{}).items():
        url=server.get('url','')
        if server.get('type') not in ('http','streamable-http','sse') or urlparse(url).hostname!='scp.intern-ai.org.cn':
            continue
        if urlparse(url).path.rstrip('/')!='/api/v1/mcp/2/DrugSDA-Tool':
            continue
        headers=server.get('headers',{})
        auth=next(((k,v) for k,v in headers.items() if k.lower()=='scp-hub-api-key'),None)
        if set(k.lower() for k in headers)-{'scp-hub-api-key'}:
            raise ValueError('Unsupported extra SCP headers; configure shared adapter explicitly')
        replacement=build_config(project_root,name,int(server.get('timeout') or server.get('toolCallTimeoutMs') or 14400000))['mcpServers'][name]
        replacement['env'].update(MOLCLAW_SCP_SERVER_URL=url)
        if auth:replacement['env'].update(MOLCLAW_SCP_AUTH_HEADER=auth[0],MOLCLAW_SCP_API_KEY=auth[1])
        replacement['env'].update({k:v for k,v in server.get('env',{}).items() if k not in replacement['env']})
        payload['mcpServers'][name]=replacement;changed=True
    if not changed:return config
    fd,name=tempfile.mkstemp(prefix='molclaw_polling_',suffix='.json')
    result=Path(name)
    with os.fdopen(fd,'w') as f:json.dump(payload,f)
    atexit.register(lambda:result.unlink(missing_ok=True))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--name',default='molclaw-scp');p.add_argument('--timeout',type=int,default=14400000);a=p.parse_args()
    payload=build_config(a.project_root.resolve(),a.name,a.timeout)
    with os.fdopen(os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:json.dump(payload,f,indent=2);f.write('\n')
    a.output.chmod(0o600)
