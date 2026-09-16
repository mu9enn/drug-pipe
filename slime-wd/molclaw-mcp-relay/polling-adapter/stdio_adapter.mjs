import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import { load } from './sdk_loader.mjs';
import { createAdapter } from './adapter_server.mjs';
import {createLocalLog} from './local_log.mjs';
const { Client }=await load('client/index.js');
const { Server }=await load('server/index.js');
const { StdioServerTransport }=await load('server/stdio.js');
const { StreamableHTTPClientTransport }=await load('client/streamableHttp.js');
const schemas=await load('types.js');
const interval=Number(process.env.MOLCLAW_POLL_INTERVAL_SECONDS ?? 300);
if(!(interval>0 && interval<=300))throw new Error('Invalid MOLCLAW_POLL_INTERVAL_SECONDS');
await import(pathToFileURL(process.env.DRUG_PROJECT+'/slime-wd/molclaw-mcp-relay/install_mcp_headers_timeout.mjs'));
const key=process.env.MOLCLAW_SCP_API_KEY || fs.readFileSync(process.env.SECRET_FILE,'utf8')
  .match(/^(?:export\s+)?MOLCLAW_SCP_API_KEY\s*=\s*(.*?)\s*$/m)?.[1].replace(/^['"]|['"]$/g,'');
if(!key)throw new Error('Missing SCP credential');
const log=createLocalLog({root:process.env.DRUG_PROJECT,secret:key});
log.write({event:'adapter_start',version:'recovery-20260914',headersTimeoutMs:Number(process.env.MOLCLAW_HEADERS_TIMEOUT_MS??14400000),pollSeconds:interval});
const remote=new Client({name:'molclaw-polling-adapter',version:'20260914'},{capabilities:{}});
const remoteTransport=new StreamableHTTPClientTransport(
  new URL('https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool'),
  {requestInit:{headers:{'SCP-HUB-API-KEY':key}}});
try {await remote.connect(remoteTransport);}catch(error){log.write({event:'startup_failure',error});throw error;}
const server=createAdapter({Server,schemas,remote,waitSeconds:interval,
  onEvent:e=>log.write(e)});
server.onclose=()=>{void remote.close();};
await server.connect(new StdioServerTransport());
log.write({event:'adapter_ready',version:'recovery-20260914'});
