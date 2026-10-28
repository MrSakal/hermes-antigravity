"""Pure ASGI limits/metrics; preserve the original disconnect receive channel."""
from __future__ import annotations

import asyncio
import json
import logging
import time

import anyio

from app.errors import OpenAIError


class RequestGuard:
    def __init__(self, app, settings, metrics):
        self.app, self.settings, self.metrics = app, settings, metrics

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path = scope.get('path', '')
        # Bound all API bodies (input validation). No concurrency cap: a
        # provider serves every call; completions are only counted.
        is_chat = path == '/v1/chat/completions'
        guarded = path.startswith(('/v1/', '/api/', '/mcp'))
        state = scope.setdefault('state', {})
        started = time.monotonic()
        if is_chat:
            self.metrics.http_active += 1
            self.metrics.counters['requests'] += 1
            state['request_started'] = started
            state['resource_releases'] = []
        status = 500
        async def observed_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                ignored = state.get('ignored_parameters')
                if ignored:
                    message = dict(message)
                    message['headers'] = list(message.get('headers', [])) + [
                        (b'x-agi-ignored-parameters', ','.join(ignored).encode())]
            if is_chat and message['type'] == 'http.response.body':
                body = message.get('body', b'')
                if body.startswith(b'data: {'):
                    try:
                        obj = json.loads(body[6:])
                        if obj.get('error'):
                            code = obj['error'].get('type', '')
                            self.metrics.counters['stream_errors'] += 1
                            if code == 'timeout_error':
                                self.metrics.counters['timeouts'] += 1
                    except (ValueError, TypeError):
                        pass
            await send(message)
        try:
            if guarded and scope.get('method') in ('POST','PUT','PATCH'):
                cap = self.settings.max_request_bytes
                for key, value in scope.get('headers', []):
                    if key == b'content-length':
                        try:
                            length = int(value)
                        except ValueError:
                            return await OpenAIError('invalid Content-Length', status_code=400).json_response()(scope, receive, observed_send)
                        if length < 0 or length > cap:
                            return await OpenAIError('HTTP body exceeds byte limit', status_code=413,
                                code='request_too_large').json_response()(scope, receive, observed_send)
                body = bytearray()
                body_deadline = time.monotonic() + self.settings.body_timeout_s
                while True:
                    try:
                        message = await asyncio.wait_for(receive(), timeout=max(0, body_deadline-time.monotonic()))
                    except TimeoutError:
                        return await OpenAIError('request body timeout', status_code=408,
                            code='request_timeout').json_response()(scope, receive, observed_send)
                    if message['type'] == 'http.disconnect':
                        status = 499
                        return
                    body.extend(message.get('body', b''))
                    if len(body) > cap:
                        return await OpenAIError('HTTP body exceeds byte limit', status_code=413,
                            code='request_too_large').json_response()(scope, receive, observed_send)
                    if not message.get('more_body', False):
                        break
                replayed = False
                async def bounded_receive():
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {'type':'http.request', 'body':bytes(body), 'more_body':False}
                    return await receive()
                await self.app(scope, bounded_receive, observed_send)
            else:
                await self.app(scope, receive, observed_send)
        except asyncio.CancelledError:
            if is_chat:
                self.metrics.counters['cancelled'] += 1
            raise
        finally:
            if is_chat:
                # Own cleanup even if StreamingResponse never begins its iterator.
                with anyio.CancelScope(shield=True):
                    try:
                        close = state.get('session_cleanup')
                        if close is not None:
                            await close()
                        conv_close = state.get('conversation_cleanup')
                        if conv_close is not None:
                            await conv_close()
                    except Exception:
                        logging.getLogger('agi.cleanup').exception('request cleanup failed')
                        self.metrics.counters['cleanup_errors'] += 1
                    finally:
                        for release in reversed(state.get('resource_releases', [])):
                            release()
                self.metrics.http_active -= 1
                self.metrics.counters[f'http_{status}'] += 1
                if status == 499:
                    self.metrics.counters['disconnects'] += 1
                elif status == 504:
                    self.metrics.counters['timeouts'] += 1
                self.metrics.duration.append(time.monotonic()-started)
                if state.get('first_token_at') is not None:
                    self.metrics.ttft.append(state['first_token_at']-started)
