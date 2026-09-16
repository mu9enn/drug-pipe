import { randomUUID } from 'node:crypto';
import {retrySameRequest} from './transport_recovery.mjs';
import { awaitToolResult } from './polling_client_core.mjs';

// A standard MCP server boundary: polling envelopes never leave this handler.
export function createAdapter({ Server, schemas, remote, waitSeconds = 300, onEvent = () => {} }) {
  const server = new Server({ name:'molclaw-scp', version:'polling-20260914' },
    { capabilities:{ tools:{} } });
  const recover=retrySameRequest((rpc,options)=>remote.request(rpc, schemas.CallToolResultSchema, options), {onRetry:e=>onEvent({event:'transport_retry',...e}),onFailure:e=>onEvent({event:'transport_failure',...e})});
  server.setRequestHandler(schemas.ListToolsRequestSchema, async request => remote.listTools(request.params));
  server.setRequestHandler(schemas.CallToolRequestSchema, async (request, extra) => {
    const name=request.params.name;
    const jobId=randomUUID();
    onEvent({event:'tool_wait_started',tool:name,jobId});
    const result=await awaitToolResult({ name, arguments:request.params.arguments,
      waitSeconds, jobId, signal:extra.signal,
      request:recover,
      onHeartbeat:async heartbeat=>{
        onEvent({event:'waiting',tool:name,...heartbeat});

      },
    });
    onEvent({event:'final_result',tool:name,isError:result.isError===true});
    return result;
  });
  return server;
}
