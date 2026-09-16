// Passive Node/Undici diagnostics. No headers, request bodies or credentials.
import { channel } from 'node:diagnostics_channel';
const requests = new WeakMap();
let nextId = 0;
function detail(error, depth = 0) {
  if (!error || depth > 4) return undefined;
  return {name: error.name, message: error.message, code: error.code,
    syscall: error.syscall, address: error.address, port: error.port,
    cause: detail(error.cause, depth + 1),
    errors: error.errors?.map(e => detail(e, depth + 1))};
}
function emit(event, request, extra = {}) {
  const info = requests.get(request);
  if (!info) return;
  console.error('[mcp-http]', JSON.stringify({utc: new Date().toISOString(), event,
    request_id: info.id, method: request.method, elapsed_ms: Date.now() - info.start, ...extra}));
}
channel('undici:request:create').subscribe(({request}) => {
  if (!String(request.origin).includes('scp.intern-ai.org.cn')) return;
  requests.set(request, {id: ++nextId, start: Date.now()});
  emit('start', request);
});
channel('undici:request:headers').subscribe(({request, response}) => {
  emit('headers', request, {status: response.statusCode});
});
channel('undici:request:error').subscribe(({request, error}) => {
  emit('error', request, {error: detail(error)});
});
