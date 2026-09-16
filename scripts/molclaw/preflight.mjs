// Safe, cheap live check: one SMILES validation, no model/API invocation.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const root=process.env.DRUG_PROJECT;
const {load}=await import(pathToFileURL(root+'/slime-wd/molclaw-mcp-relay/polling-adapter/sdk_loader.mjs'));
const {Client}=await load('client/index.js'),{StdioClientTransport}=await load('client/stdio.js');
const client=new Client({name:'drug-pipe-preflight',version:'1'},{capabilities:{}});
const transport=new StdioClientTransport({command:'bash',args:[root+'/runtime/molclaw_mcp.sh'],env:{...process.env},stderr:'inherit'});
let notifications=0;client.fallbackNotificationHandler=async()=>{notifications++};
try {
 await client.connect(transport,{timeout:120000});
 const catalog=await client.listTools();assert(catalog.tools.some(t=>t.name==='is_valid_smiles'));
 const result=await client.callTool({name:'is_valid_smiles',arguments:{smiles_list:['CCO']}},undefined,{timeout:120000});
 assert(!result.isError, 'Tool returned an error');
 assert(!JSON.stringify(result).includes('__molclaw_poll_v1'),'Pending envelope escaped adapter');
 const payload=result.structuredContent??JSON.parse(result.content.find(c=>c.type==='text').text);
 assert.equal(payload.status,'success');assert(payload.valid_res.some(r=>r.smiles==='CCO'&&r.is_valid===true));
 assert.equal(notifications,0);
 console.log(JSON.stringify({passed:true,node:process.version,tools:catalog.tools.length,tool:'is_valid_smiles',notifications}));
} finally {await client.close()}
