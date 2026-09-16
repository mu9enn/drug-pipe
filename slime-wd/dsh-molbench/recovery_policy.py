"""Conservative retry decisions for observable failures, never diagnostic history."""
import json
import re

BUDGET = re.compile(r'task exceeded \d+ seconds|trajectory_budget_exhausted', re.I)
PERMANENT = re.compile(
    r'(?:HTTP|status(?: code)?|Client Error)\s*[:=]?\s*(?:400|401|403|404|405|409|410|413|415|422)\b|'
    r'\b(?:400|401|403|404|405|409|410|413|415|422) Client Error\b|'
    r'invalid (?:input|argument|parameter|smiles)|validation error|schema validation|'
    r'pre-condition violation|input_rejected|input_validation|unsupported_|'
    r'file not found|no such file|filenotfounderror|missing (?:file|dependency)|'
    r'GetAtoms|implicitValence|CERT_|certificate|unauthorized|forbidden|'
    r'no sandbox backend|lossless JSON|out of memory|OOMKilled|out_of_memory|std::bad_alloc', re.I)
TRANSIENT = re.compile(
    r'fetch failed|ECONNRESET|ECONNREFUSED|ETIMEDOUT|EPIPE|EAI_AGAIN|ENETUNREACH|'
    r'UND_ERR_(?:SOCKET|CONNECT_TIMEOUT|HEADERS_TIMEOUT|BODY_TIMEOUT)|'
    r'connection (?:reset|refused|closed)|broken pipe|temporarily unavailable|'
    r'stream (?:idle )?timeout|timed out|TimeoutError|'
    r'Proxy response \((?:502|503|504)\)|'
    r'(?:HTTP|status(?: code)?)\s*[:=]?\s*(?:408|429|502|503|504)\b|'
    r'\b(?:408|429|502|503|504) (?:Server|Client) Error\b', re.I)


def error_class(text, code=None):
    if BUDGET.search(text):
        return 'trajectory_budget_exhausted'
    if PERMANENT.search(text):
        return 'nonretryable_tool_or_dependency'
    if TRANSIENT.search(text) or code in {'TIMEOUT', 'TRANSPORT', 'MCP_CONNECTION'}:
        return 'retryable_infra'
    return 'unclassified_failure'


def observation(data):
    """Return (failed, class, error text); successful payload text is never scanned."""
    message = data.get('message') or {}
    blocks = message.get('content') or []
    explicit = bool(data.get('error')) or any(b.get('isError') or b.get('is_error') for b in blocks)
    errors = []
    for block in blocks:
        for part in block.get('content') or []:
            value = part.get('text', '')
            if not isinstance(value, str):
                continue
            prefixed = value.startswith(('Error:', 'McpError:', 'TypeError:', 'TimeoutError:'))
            candidate = value.split(':', 1)[1].strip() if prefixed else value
            try:
                payload = json.loads(candidate)
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict):
                diag = payload.get('diagnostics') or {}
                failed = payload.get('status') in ('error', 'failed', False) or bool(payload.get('error'))
                failed |= bool(diag.get('error_code')) if isinstance(diag, dict) else False
                if failed:
                    explicit = True
                    errors.append(json.dumps({k: payload.get(k) for k in ('msg', 'message', 'error', 'diagnostics')}, ensure_ascii=False))
                elif prefixed or explicit:
                    errors.append(value)
            elif prefixed or explicit:
                explicit = True
                errors.append(value)
    if data.get('error'):
        errors.append(json.dumps(data['error'], ensure_ascii=False))
    text = '\n'.join(errors)
    return explicit, error_class(text) if explicit else None, text


def unresolved_infrastructure(events, record):
    # Exhausting the trajectory budget must not trigger a network retry, even if
    # an earlier tool failure is also present in the same trajectory.
    if record.get('budget_exhausted') or BUDGET.search(str(record.get('error', ''))):
        return []
    calls, unresolved = {}, {}
    for event in events:
        data = event.get('data') or {}
        if event.get('type') == 'tool/call':
            args = data.get('arguments')
            if isinstance(args, str):
                try: args = json.loads(args)
                except ValueError: pass
            calls[data.get('callId')] = (data.get('name'), json.dumps(args, sort_keys=True, ensure_ascii=False))
        elif event.get('type') == 'tool/result':
            cid = ((data.get('message') or {}).get('source') or {}).get('callId')
            key = calls.get(cid)
            if key is None: continue
            failed, classification, text = observation(data)
            if failed and classification == 'retryable_infra':
                unresolved[key] = {'tool': key[0], 'call_id': cid, 'seq': event.get('seq'), 'error': text[:2000], 'classification': classification}
            else:
                # A later definitive response supersedes an earlier connection
                # error for identical arguments, even if its business result fails.
                unresolved.pop(key, None)
    reason = record.get('turn_reason') or {}
    error = reason.get('error') or record.get('error')
    if error:
        code = error.get('code') if isinstance(error, dict) else None
        text = json.dumps(error, ensure_ascii=False)
        if error_class(text, code) == 'retryable_infra':
            unresolved[('terminal', '')] = {'tool': None, 'error': text[:2000], 'classification': 'retryable_infra'}
    return list(unresolved.values())
