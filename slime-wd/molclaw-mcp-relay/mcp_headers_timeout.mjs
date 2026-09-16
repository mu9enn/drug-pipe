// Deployment-only fetch policy; retain the native Node dispatcher and proxy routing.
import { createRequire } from 'node:module';

export function withHeadersTimeout(fetchImpl, getDispatcher, origin, headersTimeout) {
  return function fetchWithMcpTimeout(input, init) {
    const url = new URL(input instanceof Request ? input.url : input);
    if (url.origin !== origin) return fetchImpl(input, init);
    const upstream = init?.dispatcher ?? getDispatcher();
    const dispatcher = {
      dispatch(options, handler) {
        return upstream.dispatch({ ...options, headersTimeout }, handler);
      },
    };
    return fetchImpl(input, { ...init, dispatcher });
  };
}

export async function install() {
  const headersTimeout = Number(process.env.MOLCLAW_HEADERS_TIMEOUT_MS ?? '14400000');
  if (!Number.isSafeInteger(headersTimeout) || headersTimeout <= 0) {
    throw new Error('MOLCLAW_HEADERS_TIMEOUT_MS must be a positive integer');
  }
  // Initialize Node's native fetch and environment proxy before loading Undici's public accessor.
  const nativeFetch = globalThis.fetch;
  await nativeFetch('data:,');
  const require = createRequire(process.env.MOLCLAW_DEPS_ROOT ? process.env.MOLCLAW_DEPS_ROOT + '/package.json' : new URL('../deepseek-harness/node_modules/.pnpm/node_modules/undici/package.json', import.meta.url));
  const { getGlobalDispatcher } = require('undici');
  globalThis.fetch = withHeadersTimeout(nativeFetch, getGlobalDispatcher,
    'https://scp.intern-ai.org.cn', headersTimeout);
  console.error('[mcp-http-policy]', JSON.stringify({ headersTimeoutMs: headersTimeout,
    scope: 'https://scp.intern-ai.org.cn', node: process.version,
    dispatcher: getGlobalDispatcher().constructor.name }));
}
