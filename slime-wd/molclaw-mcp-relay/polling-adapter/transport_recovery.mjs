// Recover only known transient transport errors; callers must preserve idempotency.
import {setTimeout as delay} from 'node:timers/promises';
export function transientFetch(error) {
 if(error?.name!=='TypeError'||error.message!=='fetch failed')return false;
 const allowed=new Set(['UND_ERR_SOCKET','UND_ERR_CONNECT_TIMEOUT','ECONNRESET','ECONNREFUSED','EPIPE','ETIMEDOUT']);
 for(let e=error.cause;e;e=e.cause) {
  if(allowed.has(e.code))return true;
  if(/Proxy response \((502|503|504)\) !== 200 when HTTP Tunneling/.test(e.message))return true;
 }
 return false;
}
export function retrySameRequest(request,{delays=[1000,3000,9000],onRetry=()=>{},onFailure=()=>{}}={}) {
 return async (rpc,options={})=>{
  const deadline=Date.now()+(options.timeout??360000);
  for(let attempt=0;;attempt++){
   options.signal?.throwIfAborted();
   const remaining=deadline-Date.now();if(remaining<=0)throw new Error('Transport recovery exhausted original RPC budget');
   try{return await request(rpc,{...options,timeout:remaining});}
   catch(error){
    onFailure({attempt:attempt+1,jobId:rpc.params?.arguments?._molclaw_poll?.id,action:rpc.params?.arguments?._molclaw_poll?.action,error,retryable:transientFetch(error)&&!options.signal?.aborted});
    if(options.signal?.aborted||attempt>=delays.length||!transientFetch(error))throw error;
    const waitMs=delays[attempt];if(Date.now()+waitMs>=deadline)throw error;
    onRetry({attempt:attempt+1,jobId:rpc.params?.arguments?._molclaw_poll?.id,action:rpc.params?.arguments?._molclaw_poll?.action});
    await delay(delays[attempt],undefined,{signal:options.signal});
   }
  }
 };
}
