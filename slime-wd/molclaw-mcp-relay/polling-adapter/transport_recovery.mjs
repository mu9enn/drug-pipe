// Retry transport requests only: the durable polling id and arguments never change.
import {setTimeout as delay} from 'node:timers/promises';
export function transientFetch(error) {
 if(error?.name!=='TypeError'||error.message!=='fetch failed')return false;
 const allowed=new Set(['UND_ERR_SOCKET','UND_ERR_CONNECT_TIMEOUT','UND_ERR_HEADERS_TIMEOUT','UND_ERR_BODY_TIMEOUT','ECONNRESET','ECONNREFUSED','EPIPE','ETIMEDOUT','EAI_AGAIN']);
 for(let e=error.cause;e;e=e.cause) {
  if(allowed.has(e.code))return true;
  if(/Proxy response \((502|503|504)\) !== 200 when HTTP Tunneling/.test(e.message))return true;
 }
 return !error.cause;
}
export function forwardedTransportEof(error) {
 // Match the SCP forwarding error, not arbitrary application/internal EOFs.
 return error?.code===-32603 && /calling "tools\/call": sending "tools\/call": rejected by transport: Post "[^"\r\n]+": (?:unexpected )?EOF\s*$/.test(error.message??'');
}
export function transientTransport(error) {
 return transientFetch(error) ||
  (error?.code===-32001 && /Request timed out/i.test(error.message)) ||
  (error?.name==='StreamableHTTPError' && [502,503,504].includes(error.code)) || forwardedTransportEof(error);
}
export function retrySameRequest(request,{delays=[1000,3000,9000,30000],
 recoveryBudgetMs=1800000,onRetry=()=>{},onFailure=()=>{}}={}) {
 if(!Number.isFinite(recoveryBudgetMs)||recoveryBudgetMs<=0||!delays.length||delays.some(n=>!Number.isFinite(n)||n<0))throw new Error('Invalid transport recovery policy');
 return async (rpc,options={})=>{
  const {recoveryTimeoutMs=recoveryBudgetMs,...requestOptions}=options;
  const budget=Math.min(recoveryBudgetMs,recoveryTimeoutMs);
  if(!Number.isFinite(budget)||budget<=0)throw new Error('Invalid transport recovery budget');
  const deadline=performance.now()+budget;
  const exhausted=()=>Object.assign(new Error('MolClaw transport recovery budget exhausted; background job was not restarted'),{code:'MOLCLAW_TRANSPORT_RECOVERY_EXHAUSTED'});
  for(let attempt=0;;attempt++){
   options.signal?.throwIfAborted();
   const remaining=deadline-performance.now();if(remaining<=0)throw exhausted();
   try{return await request(rpc,{...requestOptions,timeout:Math.min(requestOptions.timeout??120000,remaining)});}
   catch(error){
    const control=rpc.params?.arguments?._molclaw_poll;
    const safeEof=!forwardedTransportEof(error) || (rpc.method==='tools/call' && typeof control?.id==='string' && /^[A-Za-z0-9_-]{24,128}$/.test(control.id) && ['start','poll'].includes(control.action));
    const retryable=transientTransport(error)&&safeEof&&!options.signal?.aborted;
    onFailure({attempt:attempt+1,jobId:rpc.params?.arguments?._molclaw_poll?.id,action:rpc.params?.arguments?._molclaw_poll?.action,error,retryable});
    if(!retryable)throw error;
    const waitMs=delays[Math.min(attempt,delays.length-1)];
    if(performance.now()+waitMs>=deadline)throw exhausted();
    onRetry({attempt:attempt+1,jobId:rpc.params?.arguments?._molclaw_poll?.id,action:rpc.params?.arguments?._molclaw_poll?.action,remainingRecoveryMs:Math.round(deadline-performance.now())});
    await delay(waitMs,undefined,{signal:options.signal});
   }
  }
 };
}
