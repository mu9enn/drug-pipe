// Shared entry for Data-Pipe, native slime and generic DSH. Frozen experiments
// keep stdio_adapter.mjs; both use the same silent polling/recovery core.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {load} from './sdk_loader.mjs';
import {createAdapter} from './adapter_server.mjs';
import {createLocalLog} from './local_log.mjs';
import {transientFetch} from './transport_recovery.mjs';
const {Client}=await load('client/index.js');
const {Server}=await load('server/index.js');
const {StdioServerTransport}=await load('server/stdio.js');
const {StreamableHTTPClientTransport}=await load('client/streamableHttp.js');
const schemas=await load('types.js');
const interval=Number(process.env.MOLCLAW_POLL_INTERVAL_SECONDS??300);
if(!(interval>0&&interval<=300))throw new Error('Invalid polling interval');
await import(pathToFileURL(process.env.DRUG_PROJECT+'/slime-wd/molclaw-mcp-relay/install_mcp_headers_timeout.mjs'));
const env=process.env;
let key=env.MOLCLAW_SCP_API_KEY||env.MOLCLAW_SCP_MCP_AUTH;
if(!key&&env.SECRET_FILE&&fs.existsSync(env.SECRET_FILE)) {
 const text=fs.readFileSync(env.SECRET_FILE,'utf8');
 key=text.match(/^(?:export\s+)?(?:MOLCLAW_SCP_API_KEY|MOLCLAW_SCP_MCP_AUTH)\s*=\s*(.*?)\s*$/m)?.[1].replace(/^['"]|['"]$/g,'');
}
if(!key)throw new Error('Missing SCP credential');
const url=env.MOLCLAW_SCP_SERVER_URL||env.MOLCLAW_SCP_MCP_URL||'https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool';
const header=env.MOLCLAW_SCP_AUTH_HEADER||env.MOLCLAW_SCP_MCP_AUTH_HEADER||'SCP-HUB-API-KEY';
if(!header||/[\r\n:]/.test(header))throw new Error('Invalid SCP auth header');
const log=createLocalLog({root:env.DRUG_PROJECT,secret:key});
log.write({event:'adapter_start',version:'shared-entry-20260916',pollSeconds:interval});
let remote;
for(let attempt=0;;attempt++) {
 remote=new Client({name:'molclaw-polling-adapter',version:'20260916'},{capabilities:{}});
 const transport=new StreamableHTTPClientTransport(new URL(url),{requestInit:{headers:{[header]:key}}});
 try {await remote.connect(transport);break;}
 catch(error) {
  log.write({event:'startup_failure',attempt:attempt+1,error});
  try {await remote.close();}catch{}
  if(attempt>=3||!transientFetch(error))throw error;
  await delay([1000,3000,9000][attempt]);
 }
}
const server=createAdapter({Server,schemas,remote,waitSeconds:interval,onEvent:e=>log.write(e)});
server.onclose=()=>{void remote.close();};
await server.connect(new StdioServerTransport());
log.write({event:'adapter_ready',version:'shared-entry-20260916'});
