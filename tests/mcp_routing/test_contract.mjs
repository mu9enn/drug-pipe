import assert from 'node:assert/strict';import {pathToFileURL} from 'node:url';

const A=process.env.DRUG_PROJECT+'/slime-wd/molclaw-mcp-relay/polling-adapter/';
const {load}=await import(pathToFileURL(A+'sdk_loader.mjs'));const {createAdapter}=await import(pathToFileURL(A+'adapter_server.mjs'));
const {Client}=await load('client/index.js'),{Server}=await load('server/index.js'),{InMemoryTransport}=await load('inMemory.js'),schemas=await load('types.js');
for(const progressAware of [false,true]){
 const tool={name:'slow_tool',description:'Original',inputSchema:{type:'object',properties:{x:{type:'integer'}},required:['x']}};
 let release,reached,settled=false;const gate=new Promise(r=>release=r),atFinal=new Promise(r=>reached=r);let n=0;const ids=[],actions=[],wire=[];
 const remote={listTools:async()=>({tools:[tool]}),request:async rpc=>{
  n++;const c=rpc.params.arguments._molclaw_poll;ids.push(c.id);actions.push(c.action);
  if(n===1)throw new TypeError('fetch failed',{cause:new Error('Proxy response (502) !== 200 when HTTP Tunneling')});
  if(n===3)throw new TypeError('fetch failed',{cause:Object.assign(new Error('socket lost'),{code:'UND_ERR_SOCKET'})});
  if(n===2||n===4)return {content:[{type:'text',text:JSON.stringify({__molclaw_poll_v1:{id:c.id,state:'running',elapsed_seconds:n},message:'waiting'})}]};
  reached();await gate;return {content:[{type:'text',text:'REAL TOOL RESULT'}]};
 }};
 const server=createAdapter({Server,schemas,remote,waitSeconds:10});const [a,b]=InMemoryTransport.createLinkedPair();const send=a.send.bind(a);a.send=async m=>{wire.push(m);return send(m);};
 const client=new Client({name:'unmodified-client',version:'1'},{capabilities:{}});await server.connect(a);await client.connect(b);assert.deepEqual((await client.listTools()).tools,[tool]);
 const p=client.request({method:'tools/call',params:{name:'slow_tool',arguments:{x:42}}},schemas.CallToolResultSchema,progressAware?{onprogress:()=>{throw Error('Unexpected progress');}}:{}).then(r=>{settled=true;return r;});
 await atFinal;assert.equal(settled,false);assert.equal(wire.filter(m=>m.result?.content).length,0);assert.equal(wire.filter(m=>m.method?.startsWith('notifications/')).length,0);assert.equal(new Set(ids).size,1);assert.deepEqual(actions,['start','start','poll','poll','poll']);
 release();assert.equal((await p).content[0].text,'REAL TOOL RESULT');assert.equal(wire.filter(m=>m.result?.content).length,1);assert.equal(JSON.stringify(wire).includes('__molclaw_poll_v1'),false);
 await client.close();await server.close();console.log(JSON.stringify({passed:true,progressAware,transient_failures:2,outer_final_results:1,notifications:0,same_id_for_start_and_poll_recovery:true}));
}
