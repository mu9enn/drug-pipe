import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {randomUUID} from 'node:crypto';
export function errorDetail(error,seen=new Set(),depth=0){
 if(!error||depth>6||seen.has(error))return undefined;
 seen.add(error);return {name:error.name,message:error.message,code:error.code,cause:errorDetail(error.cause,seen,depth+1)};
}
export function createLocalLog({root,secret='',file=process.env.MOLCLAW_ADAPTER_LOG_FILE}){
 const dir=path.join(root,'slime-wd/outputs/molclaw_adapter_logs');
 if(!file){fs.mkdirSync(dir,{recursive:true,mode:0o700});file=path.join(dir,`${os.hostname().replace(/[^a-zA-Z0-9_.-]/g,'_')}-${process.pid}-${randomUUID()}.jsonl`);}
 const fd=fs.openSync(file,'a',0o600);fs.fchmodSync(fd,0o600);
 return {file,write(event){
  try{
   let line=JSON.stringify({...event,...(event.error?{error:errorDetail(event.error)}:{}),utc:new Date().toISOString(),pid:process.pid});
   if(secret)line=line.split(secret).join('[REDACTED]');
   fs.writeSync(fd,line+'\n');
  }catch{/* Diagnostic I/O must not convert a real tool result into a failure. */}
 },close(){fs.closeSync(fd);}};
}
