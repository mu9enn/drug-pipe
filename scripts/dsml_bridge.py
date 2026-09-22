"""Loopback Anthropic-to-OpenAI bridge with validated DeepSeek DSML rescue.

Credentials are read from the selected cc-switch provider, never logged.
Buffer each upstream completion before exposing tool calls to Claude. Pings keep
the downstream SSE connection alive; malformed/truncated DSML fails closed.
"""
import argparse
import concurrent.futures
import json
import re
import sqlite3
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from jsonschema import validate
from jsonschema.exceptions import ValidationError


MARKER = r"[|｜]DSML[|｜]"
INVOKE = re.compile(r'<'+MARKER+r'invoke name="([^"]+)">([\s\S]*?)</'+MARKER+r'invoke>')
PARAM = re.compile(r'<'+MARKER+r'parameter name="([^"]+)" string="(true|false)">([\s\S]*?)</'+MARKER+r'parameter>')
WRAPPER = re.compile(r'</?'+MARKER+r'tool_calls>')


def rescue(text, tools):
    if not re.search(MARKER, text):
        return text, []
    # Do not turn quoted/code-fenced examples or malformed tails into actions.
    if '```' in text:
        raise ValueError('DSML in code fence is ambiguous')
    calls = []
    for match in INVOKE.finditer(text):
        name, body = match.groups()
        if name not in tools:
            raise ValueError('DSML requested an unavailable tool')
        args = {}
        for p in PARAM.finditer(body):
            key, string, value = p.groups()
            if key in args:
                raise ValueError('duplicate DSML parameter')
            args[key] = value if string == 'true' else json.loads(value)
        if PARAM.sub('', body).strip():
            raise ValueError('unparsed DSML parameters')
        validate(args, tools[name])
        calls.append({'id': 'toolu_'+uuid.uuid4().hex, 'type': 'tool_use', 'name': name, 'input': args})
    remaining = WRAPPER.sub('', INVOKE.sub('', text)).strip()
    if not calls or re.search(MARKER, remaining):
        raise ValueError('incomplete or unsupported DSML')
    return remaining, calls


def plain(content):
    if isinstance(content, str):
        return content
    if any(b.get('type') != 'text' for b in (content or [])):
        raise ValueError('non-text content requires an explicit modality adapter')
    return '\n'.join(b.get('text', '') for b in (content or []) if b.get('type') == 'text')


def convert_request(body, model):
    messages = []
    if body.get('system'):
        messages.append({'role': 'system', 'content': plain(body['system'])})
    for message in body['messages']:
        content = message.get('content', '')
        if isinstance(content, str):
            messages.append({'role': message['role'], 'content': content})
            continue
        text, thinking, calls = [], [], []
        for b in content:
            kind = b.get('type')
            if kind == 'tool_result':
                messages.append({'role': 'tool', 'tool_call_id': b['tool_use_id'], 'content': plain(b.get('content'))})
            elif kind == 'tool_use':
                calls.append({'id': b['id'], 'type': 'function', 'function': {'name': b['name'], 'arguments': json.dumps(b['input'], ensure_ascii=False)}})
            elif kind == 'text':
                text.append(b['text'])
            elif kind == 'thinking':
                thinking.append(b['thinking'])
            elif kind not in ('redacted_thinking',):
                raise ValueError('unsupported input block: '+str(kind))
        if text or calls or thinking:
            m = {'role': message['role'], 'content': '\n'.join(text)}
            if calls:
                m['tool_calls'] = calls
            if message['role'] == 'assistant':
                m['reasoning_content'] = '\n'.join(thinking)
            messages.append(m)
    request = {'model': model, 'messages': messages, 'stream': False,
               'max_tokens': min(body.get('max_tokens', 16384), 65536)}
    tools = body.get('tools', [])
    if tools:
        request['tools'] = [{'type': 'function', 'function': {'name': t['name'], 'description': t.get('description', ''), 'parameters': t['input_schema']}} for t in tools]
    choice = body.get('tool_choice', {})
    if choice.get('type') == 'tool':
        request['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    elif choice.get('type') in ('auto', 'none', 'any'):
        request['tool_choice'] = {'any': 'required'}.get(choice['type'], choice['type'])
    return request


def convert_response(data, tools, client_model):
    choice = data['choices'][0]
    message = choice['message']
    if choice.get('finish_reason') == 'length' and message.get('tool_calls'):
        raise ValueError('truncated native tool generation')
    schemas = {t['name']: t['input_schema'] for t in tools}
    text = message.get('content') or ''
    rescued = []
    if re.search(MARKER, text):
        if message.get('tool_calls'):
            raise ValueError('ambiguous mixed native and DSML calls')
        if choice.get('finish_reason') == 'length':
            raise ValueError('truncated DSML generation')
        text, rescued = rescue(text, schemas)
    blocks = []
    if message.get('reasoning_content'):
        blocks.append({'type': 'thinking', 'thinking': message['reasoning_content'], 'signature': ''})
    if text.strip():
        blocks.append({'type': 'text', 'text': text})
    for c in message.get('tool_calls') or []:
        f = c['function']
        if f['name'] not in schemas:
            raise ValueError('native call requested unavailable tool')
        args = json.loads(f['arguments'])
        validate(args, schemas[f['name']])
        blocks.append({'type': 'tool_use', 'id': c['id'], 'name': f['name'], 'input': args})
    blocks.extend(rescued)
    has_tools = any(b['type'] == 'tool_use' for b in blocks)
    usage = data.get('usage') or {}
    return {'id': 'msg_'+uuid.uuid4().hex, 'type': 'message', 'role': 'assistant', 'model': client_model,
            'content': blocks, 'stop_reason': 'tool_use' if has_tools else ('max_tokens' if choice.get('finish_reason') == 'length' else 'end_turn'),
            'stop_sequence': None, 'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0)}}, len(rescued)


def events(message):
    yield {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None, 'usage': {**message['usage'], 'output_tokens': 0}}}
    for index, block in enumerate(message['content']):
        kind = block['type']
        start = {**block}
        field = {'text': 'text', 'thinking': 'thinking', 'tool_use': 'input'}[kind]
        start[field] = {} if kind == 'tool_use' else ''
        yield {'type': 'content_block_start', 'index': index, 'content_block': start}
        value = json.dumps(block[field], ensure_ascii=False) if kind == 'tool_use' else block[field]
        delta_type, delta_field = {'tool_use': ('input_json_delta', 'partial_json'), 'text': ('text_delta', 'text'), 'thinking': ('thinking_delta', 'thinking')}[kind]
        for offset in range(0, len(value), 4096):
            yield {'type': 'content_block_delta', 'index': index, 'delta': {'type': delta_type, delta_field: value[offset:offset+4096]}}
        if kind == 'thinking':
            yield {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'signature_delta', 'signature': ''}}
        yield {'type': 'content_block_stop', 'index': index}
    yield {'type': 'message_delta', 'delta': {'stop_reason': message['stop_reason'], 'stop_sequence': None}, 'usage': message['usage']}
    yield {'type': 'message_stop'}



def safe_error(exc):
    """Stable diagnostics without request bodies, schema values, URLs or secrets."""
    if isinstance(exc, ValidationError):
        return 'tool_schema_validation'
    if isinstance(exc, json.JSONDecodeError):
        return 'invalid_json'
    if type(exc) is ValueError:
        codes = {
            'DSML in code fence is ambiguous': 'ambiguous_dsml_code_fence',
            'DSML requested an unavailable tool': 'dsml_unavailable_tool',
            'duplicate DSML parameter': 'duplicate_dsml_parameter',
            'unparsed DSML parameters': 'unparsed_dsml_parameters',
            'incomplete or unsupported DSML': 'incomplete_or_unsupported_dsml',
            'non-text content requires an explicit modality adapter': 'unsupported_modality',
            'ambiguous mixed native and DSML calls': 'mixed_native_dsml',
            'truncated DSML generation': 'truncated_dsml',
            'truncated native tool generation': 'truncated_native_tool',
            'native call requested unavailable tool': 'native_unavailable_tool',
        }
        if str(exc).startswith('unsupported input block:'):
            return 'unsupported_input_block'
        return codes.get(str(exc), 'invalid_value')
    if type(exc) is RuntimeError and re.fullmatch(r'upstream HTTP [1-5][0-9]{2}', str(exc)):
        return str(exc).replace(' ', '_')
    return type(exc).__name__


def provider_config(provider):
    with sqlite3.connect('file:'+str(Path.home()/'.cc-switch/cc-switch.db')+'?mode=ro', uri=True) as conn:
        row = conn.execute('select settings_config from providers where id=? and app_type=?', (provider, 'claude')).fetchone()
    return json.loads(row[0])['env']


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send_json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        self.send_json(200, {'ok': True, 'provider': self.server.provider, 'model': self.server.config['ANTHROPIC_MODEL']})

    def sse(self, event):
        self.wfile.write(('event: '+event['type']+'\ndata: '+json.dumps(event, ensure_ascii=False)+'\n\n').encode()); self.wfile.flush()

    def do_POST(self):
        streaming = False
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        phase = 'request_conversion'
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.split('?')[0] != '/v1/messages':
                self.send_json(404, {'error': {'type': 'not_found_error', 'message': 'unsupported endpoint'}}); return
            request = convert_request(body, self.server.config['ANTHROPIC_MODEL'])
            config = self.server.config
            def upstream():
                nonlocal phase
                phase = 'upstream_request'
                key = config.get('ANTHROPIC_API_KEY') or config['ANTHROPIC_AUTH_TOKEN']
                with httpx.Client(timeout=1800, trust_env=False) as client:
                    response = client.post(config['ANTHROPIC_BASE_URL'].rstrip('/')+'/v1/chat/completions', json=request, headers={'Authorization': 'Bearer '+key})
                    if response.status_code != 200:
                        raise RuntimeError('upstream HTTP '+str(response.status_code))
                    phase = 'response_conversion'
                    return convert_response(response.json(), body.get('tools', []), body['model'])
            future = self.server.pool.submit(upstream)
            if body.get('stream'):
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache'); self.end_headers(); streaming = True
                while True:
                    try:
                        message, rescued = future.result(timeout=10); break
                    except concurrent.futures.TimeoutError:
                        self.sse({'type': 'ping'})
                for event in events(message): self.sse(event)
            else:
                message, rescued = future.result(); self.send_json(200, message)
            print(json.dumps({'request_id': request_id, 'elapsed_seconds': time.monotonic()-started, 'time': time.time(), 'status': 'ok', 'rescued': rescued, 'tools': [b['name'] for b in message['content'] if b['type'] == 'tool_use']}), flush=True)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            # No upstream body, prompt, credential, or parameter values in logs/errors.
            error = safe_error(exc)
            deterministic = isinstance(exc, (ValueError, ValidationError, KeyError, TypeError)) and phase != 'upstream_request'
            status = (400 if phase == 'request_conversion' else 422) if deterministic else 502
            if type(exc) is RuntimeError and re.fullmatch(r'upstream HTTP [1-5][0-9]{2}', str(exc)):
                status = int(str(exc).rsplit(' ', 1)[1])
            print(json.dumps({'request_id': request_id, 'elapsed_seconds': time.monotonic()-started, 'phase': phase, 'streaming': streaming, 'http_status': status, 'time': time.time(), 'status': 'error', 'error': error}), flush=True)
            event = {'type': 'error', 'error': {'type': 'invalid_request_error' if deterministic else 'api_error', 'message': 'DSML bridge: '+error}}
            if streaming: self.sse(event)
            else: self.send_json(status, event)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', default='dsv4flash'); parser.add_argument('--port', type=int, default=18765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.provider = args.provider; server.config = provider_config(args.provider)
    server.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    server.serve_forever()
