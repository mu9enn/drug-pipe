import { randomUUID } from 'node:crypto';
const MARKER = '__molclaw_poll_v1';

// This promise resolves only for the original final CallToolResult, never for a pending envelope.
export async function awaitToolResult({ request, name, arguments: args = {}, waitSeconds = 300,
  signal, onHeartbeat = async () => {}, jobId = randomUUID(), timeoutMs = 14400000 }) {
  if (!(waitSeconds > 0 && waitSeconds <= 300)) throw new Error('Invalid polling interval');
  if (Object.hasOwn(args, '_molclaw_poll')) throw new Error('Reserved polling argument supplied by caller');
  const deadline = Date.now() + timeoutMs;
  let action = 'start';
  while (true) {
    signal?.throwIfAborted();
    const remaining = deadline-Date.now();
    if (remaining <= 0) throw new Error(`MolClaw job wait exceeded its budget: ${jobId}`);
    const result = await request({ method:'tools/call', params:{ name, arguments:{...args,
      _molclaw_poll:{id:jobId,action,wait_seconds:Math.min(waitSeconds,remaining/1000)} } } },
      {signal, timeout:Math.min(remaining,(waitSeconds+60)*1000)});
    let payload;
    if (!result.isError && result.content?.length===1 && result.content[0].type==='text') {
      try { payload=JSON.parse(result.content[0].text); } catch {}
    }
    if (payload && Object.hasOwn(payload, MARKER)) {
      const status = payload[MARKER];
      if (status?.id!==jobId || status.state!=='running' || !Number.isFinite(status.elapsed_seconds)) {
        throw new Error('Invalid MolClaw pending envelope; it was not returned as a final tool result');
      }
      await onHeartbeat({jobId,elapsedSeconds:status.elapsed_seconds,
        message:typeof payload.message==='string'?payload.message:`${name} is still running`});
      action='poll';
      continue;
    }
    return result;
  }
}
