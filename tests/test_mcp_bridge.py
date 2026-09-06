"""Conversation-local contracts and actual MCP HTTP session routing."""
import asyncio
import json

import httpx
import pytest

from app.mcp_bridge import ConversationBridge, McpBridge, ToolResultError, current_conv_id
from app.openai_models import ToolDef


def tools(name='allowed', schema=None):
    return [ToolDef(function={'name': name, 'parameters': schema if schema is not None else {
        'type': 'object', 'required': ['count'], 'properties': {'count': {'type': 'integer'}},
        'additionalProperties': False,
    }})]


@pytest.mark.parametrize('name,args', [('unknown', {'count': 1}), ('allowed', {}),
    ('allowed', {'count': 'wrong'}), ('allowed', {'count': 1, 'extra': True}),
    ('allowed', []), ('allowed', {'count': float('nan')})])
async def test_invalid_dispatch_never_enqueues(name, args):
    bridge = ConversationBridge('a', tools())
    with pytest.raises(ToolResultError):
        await bridge.dispatch(name, args)
    assert not bridge.pending_ids and bridge._incoming.empty()


async def test_local_schema_reference_and_no_remote_retrieval(monkeypatch):
    import urllib.request
    def forbidden(*args, **kwargs):
        pytest.fail('remote schema retrieval attempted')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    local = ConversationBridge('a', tools(schema={'type': 'object', '$defs': {
        'integer': {'type': 'integer'}}, 'properties': {'n': {'$ref': '#/$defs/integer'}}}))
    with pytest.raises(ToolResultError):
        await local.dispatch('allowed', {'n': 'wrong'})
    task = asyncio.create_task(local.dispatch('allowed', {'n': 1}))
    call = await local.next_incoming(.5)
    assert call is not None
    local.resolve(call.id, 'ok')
    assert await task == 'ok'
    remote = ConversationBridge('b', tools(schema={'$ref': 'https://example.invalid/schema'}))
    with pytest.raises(ToolResultError):
        await remote.dispatch('allowed', {})
    assert not remote.pending_ids and remote._incoming.empty()


async def test_bounded_queue_and_cancelled_dispatch_reclaims_capacity():
    bridge = ConversationBridge('a', tools(), max_pending_tool_calls=2)
    tasks = [asyncio.create_task(bridge.dispatch('allowed', {'count': i})) for i in range(2)]
    await asyncio.sleep(0)
    assert bridge._incoming.qsize() == len(bridge.pending_ids) == 2
    with pytest.raises(ToolResultError, match='capacity'):
        await bridge.dispatch('allowed', {'count': 3})
    tasks[0].cancel()
    await asyncio.gather(tasks[0], return_exceptions=True)
    assert bridge._incoming.qsize() == len(bridge.pending_ids) == 1
    new = asyncio.create_task(bridge.dispatch('allowed', {'count': 3}))
    await asyncio.sleep(0)
    bridge.fail_all('closed')
    assert not bridge.pending_ids and bridge._incoming.empty()
    results = await asyncio.gather(tasks[1], new, return_exceptions=True)
    assert all(isinstance(result, ToolResultError) for result in results)
    with pytest.raises(ToolResultError, match='closed'):
        await bridge.dispatch('allowed', {'count': 4})


async def test_default_pending_limit_is_32():
    bridge = ConversationBridge('a', tools())
    tasks = [asyncio.create_task(bridge.dispatch('allowed', {'count': i})) for i in range(32)]
    await asyncio.sleep(0)
    try:
        with pytest.raises(ToolResultError, match='32'):
            await bridge.dispatch('allowed', {'count': 33})
        assert bridge._incoming.maxsize == bridge._incoming.qsize() == 32
    finally:
        bridge.fail_all('done')
        await asyncio.gather(*tasks, return_exceptions=True)


def rpc_result(response):
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith('data: '):
            return json.loads(line[6:])
    return response.json()


async def test_real_http_sessions_remain_bound_to_conversation():
    bridge = McpBridge()
    bridge.register(ConversationBridge('a', tools('only_a')))
    bridge.register(ConversationBridge('b', tools('only_b')))
    transport = httpx.ASGITransport(app=bridge.asgi_app())
    headers = {'accept': 'application/json, text/event-stream'}
    token = current_conv_id.set('caller')
    try:
        async with bridge.lifespan():
            async with httpx.AsyncClient(transport=transport, base_url='http://localhost', headers=headers) as client:
                malformed = await client.post('/mcp/a', json={})
                assert malformed.status_code == 400
                assert len(bridge.session_manager._server_instances) == len(bridge._session_conversations)
                async def initialize(name):
                    response = await client.post('/mcp/' + name, json={'jsonrpc': '2.0', 'id': 1,
                        'method': 'initialize', 'params': {'protocolVersion': '2025-03-26',
                        'capabilities': {}, 'clientInfo': {'name': 'audit', 'version': '1'}}})
                    assert 'result' in rpc_result(response)
                    session = response.headers['mcp-session-id']
                    assert (await client.post('/mcp/' + name, headers={'mcp-session-id': session},
                        json={'jsonrpc': '2.0', 'method': 'notifications/initialized'})).status_code == 202
                    return session
                a, b = await asyncio.gather(initialize('a'), initialize('b'))
                assert a != b
                async def list_tools(name, session):
                    return await client.post('/mcp/' + name, headers={'mcp-session-id': session},
                        json={'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
                responses = await asyncio.gather(list_tools('a', a), list_tools('b', b))
                assert [rpc_result(r)['result']['tools'][0]['name'] for r in responses] == ['only_a', 'only_b']
                assert (await list_tools('b', a)).status_code == 404
                assert (await list_tools('a', b)).status_code == 404
                assert current_conv_id.get() == 'caller'
                # Exercise SDK dispatch as well as discovery, with both sessions live.
                response = await client.post('/mcp/a', headers={'mcp-session-id': a},
                    json={'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                          'params': {'name': 'only_b', 'arguments': {'count': 1}}})
                assert rpc_result(response)['result']['isError']
                assert not bridge.get('a').pending_ids and not bridge.get('b').pending_ids
                await bridge.close_conversation('a')
                assert a not in bridge.session_manager._server_instances
                assert a not in bridge._session_conversations
                assert b in bridge.session_manager._server_instances
                assert (await list_tools('b', b)).status_code == 200
                await bridge.close_conversation('b')
                assert not bridge.session_manager._server_instances
    finally:
        current_conv_id.reset(token)
