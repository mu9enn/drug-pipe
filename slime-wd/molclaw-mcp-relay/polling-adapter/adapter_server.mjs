import { randomUUID } from 'node:crypto';
import {retrySameRequest} from './transport_recovery.mjs';
import { awaitToolResult } from './polling_client_core.mjs';

// A standard MCP server boundary: polling envelopes never leave this handler.
export function createAdapter({ Server, schemas, remote, waitSeconds = 60, recoveryBudgetMs = 1800000, onEvent = () => {} }) {
  const server = new Server({ name:'molclaw-scp', version:'polling-20260919' },
    { capabilities:{ tools:{} } });
  const recover=retrySameRequest((rpc,options)=>remote.request(rpc, schemas.CallToolResultSchema, options), {recoveryBudgetMs,onRetry:e=>onEvent({event:'transport_retry',...e}),onFailure:e=>onEvent({event:'transport_failure',...e})});
  server.setRequestHandler(schemas.ListToolsRequestSchema, async request => remote.listTools(request.params));
  server.setRequestHandler(schemas.CallToolRequestSchema, async (request, extra) => {
    const name=request.params.name;
    const jobId=randomUUID();
    const started=performance.now();
    onEvent({event:'tool_wait_started',tool:name,jobId});
    const result=await awaitToolResult({ name, arguments:request.params.arguments,
      waitSeconds, jobId, signal:extra.signal,
      request:recover,
      onHeartbeat:async heartbeat=>{
        onEvent({event:'waiting',tool:name,...heartbeat});

      },
    });
    onEvent({event:'final_result',tool:name,jobId,elapsedSeconds:(performance.now()-started)/1000,isError:result.isError===true});
    return result;
  });
  return server;
}
