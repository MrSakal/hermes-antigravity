"""Audit assertions for desired behavior, run against the unmodified repository.
All writes/processes are test fixtures under /tmp; no upstream model calls.
Originally reproduced against 5cf6d78; all assertions must pass after remediation.
"""
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.agent_config import AgentPluginManager
from app.agy_session import AgySession, STREAM_CLOSED
from app.config import Settings
from app.errors import OpenAIError
from app.conversation import (Conversation, ConversationManager, DoneChunk, ErrorChunk,
                              TextChunk, history_key)
from app.events import TextDelta, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge
from app.openai_models import ChatCompletionRequest, ChatMessage, ToolDef
from conftest import kill_pid, pid_alive


def messages(image):
    return [ChatMessage(role='user', content=[
        {'type': 'text', 'text': 'Describe this image'},
        {'type': 'image_url', 'image_url': {'url': image}},
    ])]


def test_changed_image_changes_cache_key():
    import base64
    import struct
    import zlib
    def png(pixel):
        def chunk(kind, data):
            return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
        raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0)) + chunk(b'IDAT',zlib.compress(b'\x00'+pixel)) + chunk(b'IEND',b'')
        return 'data:image/png;base64,' + base64.b64encode(raw).decode()
    assert history_key('sig', messages(png(b'\xff\x00\x00'))) != history_key('sig', messages(png(b'\x00\x00\xff')))


async def test_old_prefix_does_not_resume_already_extended_conversation(tmp_path):
    """An IDLE process is consumed by its first claim: a second branch off the
    same prefix must miss (and start fresh) instead of adopting a process that
    already contains the first branch."""
    mgr = manager(tmp_path)
    prefix = [ChatMessage(role='user', content='root question'),
              ChatMessage(role='assistant', content='root answer')]
    key = history_key('sig', prefix)
    conv = Conversation(conv_id='c1', slot=None, session=SimpleNamespace(running=True),
                        bridge=ConversationBridge('c1', []), model='m', state='idle',
                        reuse_key=key)
    mgr._conversations['c1'] = conv
    mgr._idle[key] = 'c1'
    assert await mgr._claim_idle(key) is conv   # branch A adopts it
    assert await mgr._claim_idle(key) is None, 'Old prefix still points to a process already containing branch A'


class StubSession:
    def __init__(self, events, delay=0):
        self.events = list(events)
        self.delay = delay
        self.aclose = AsyncMock()
    async def next_event(self, timeout=None):
        if self.events:
            await asyncio.sleep(self.delay)
            return self.events.pop(0)
        await asyncio.sleep(timeout or 10)
        return None
    def push_back(self, ev):
        self.events.insert(0, ev)
    def diagnostics(self):
        return 'fixture process terminated without result'


async def test_stream_preserves_result_only_response(tmp_path):
    mgr = manager(tmp_path)
    sess = StubSession([TurnDone(response_text='AUDIT-RESULT-ONLY')])
    conv = Conversation('fixture', None, sess, ConversationBridge('fixture', []), 'model')
    chunks = [ch async for ch in mgr.run_turn(conv)]
    assert [c.text for c in chunks if isinstance(c, TextChunk)] == ['AUDIT-RESULT-ONLY']
    assert isinstance(chunks[-1], DoneChunk)


def manager(tmp_path):
    settings = Settings(_env_file=None, default_workdir=tmp_path, mcp_config_path=tmp_path/'mcp.json', request_timeout_s=1)
    return ConversationManager(McpBridge(), settings, agents=AgentPluginManager(tmp_path/'plugin'))


async def test_tool_eof_without_result_is_error(tmp_path):
    mgr = manager(tmp_path)
    conv = Conversation('fixture', 'hermes', StubSession([STREAM_CLOSED]), ConversationBridge('fixture', []), 'model')
    chunks = [ch async for ch in mgr.run_turn(conv)]
    assert any(isinstance(ch, ErrorChunk) for ch in chunks), repr(chunks)


async def test_tool_request_timeout_is_total_not_per_event(tmp_path):
    mgr = manager(tmp_path)
    sess = StubSession([TextDelta('a'), TextDelta('b'), TextDelta('c'), TurnDone()], delay=0.4)
    conv = Conversation('fixture', 'hermes', sess, ConversationBridge('fixture', []), 'model')
    started = time.monotonic()
    chunks = [ch async for ch in mgr.run_turn(conv)]
    elapsed = time.monotonic() - started
    assert elapsed < 1.25 and any(isinstance(ch, ErrorChunk) for ch in chunks), f'configured timeout=1s; elapsed={elapsed:.3f}s; chunks={chunks!r}'


async def test_mixed_continuation_does_not_consume_other_conversation_ids(tmp_path):
    from app.conversation import SUSPENDED, ExpiredContinuation
    from app.mcp_bridge import PendingCall
    mgr = manager(tmp_path)
    for label in ('A', 'B'):
        bridge = ConversationBridge(label, [])
        pc = PendingCall('call_'+label, 'fixture', {}, asyncio.get_running_loop().create_future())
        bridge._pending[pc.id] = pc
        mgr._conversations[label] = Conversation(label, 'slot_'+label, StubSession([]), bridge, 'model', state=SUSPENDED)
        mgr._pending_index[pc.id] = label
    request = ChatCompletionRequest(messages=[
        {'role':'tool','tool_call_id':'call_A','content':'answer A'},
        {'role':'tool','tool_call_id':'call_B','content':'answer B'},
    ])
    rejected = False
    try:
        await mgr.resume(request)
    except ExpiredContinuation:
        rejected = True
    assert rejected and mgr._pending_index.get('call_B') == 'B', f'rejected={rejected}; remaining_index={mgr._pending_index}'


async def test_agent_write_failure_rolls_back_bridge_registration(tmp_path):
    mgr = manager(tmp_path)
    mgr.agents._write_atomic = Mock(side_effect=OSError('fixture disk failure'))
    mgr.agents._ensure_plugin = Mock()
    req = ChatCompletionRequest(messages=[{'role':'user','content':'test'}])
    with pytest.raises(OpenAIError) as error:
        await mgr.create(req, model='model', workdir=tmp_path)
    assert error.value.status_code == 503
    assert not mgr.mcp._registry, repr(mgr.mcp._registry)


async def test_dispatch_rejects_undeclared_tool():
    bridge = ConversationBridge('fixture', [ToolDef(function={'name':'allowed'})])
    task = asyncio.create_task(bridge.dispatch('not_declared', {}))
    try:
        await asyncio.sleep(0)
        emitted = list(bridge._pending.values())
        for call in emitted:
            bridge.resolve(call.id, 'fixture result')
        await asyncio.gather(task, return_exceptions=True)
        assert not emitted, f'Undeclared tool was enqueued: {[c.name for c in emitted]}'
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_tool_choice_none_does_not_spawn_unrestricted_mcp_turn(tmp_path, monkeypatch):
    import app.main as main
    import app.routes.chat as chat
    settings = Settings(_env_file=None, default_workdir=tmp_path, mcp_config_path=tmp_path/'mcp.json', agent_plugin_path=tmp_path/'agents', agy_brain_path=tmp_path/'brain')
    monkeypatch.setattr(main, 'get_settings', lambda:settings)
    monkeypatch.setattr(chat, 'get_settings', lambda:settings)
    spawned = []
    booted = []
    async def fake_start(sess):
        spawned.append(sess)
        # What agy reads at boot (the transient file is gone after the first line).
        booted.append((settings.agent_plugin_path/'agents'/f'{sess.agent}.md').read_text())
        sess._queue.put_nowait(TextDelta('fixture answer'))
        sess._queue.put_nowait(TurnDone(response_text='fixture answer'))
    monkeypatch.setattr(AgySession, 'start', fake_start)
    application = main.create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        response = await client.post('/v1/chat/completions', json={
            'messages':[{'role':'user','content':'test'}],
            'tools':[{'type':'function','function':{'name':'allowed'}}],
            'tool_choice':'none',
        })
    assert response.status_code == 200
    assert spawned
    assert 'tools: "*"' not in booted[0], booted[0]
    assert not list((settings.agent_plugin_path/'agents').glob('*.md'))


async def test_close_reaps_child_processes(tmp_path, monkeypatch):
    script = tmp_path/'parent.py'
    script.write_text('import subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nprint(p.pid,flush=True)\ntime.sleep(30)\n')
    monkeypatch.setattr(AgySession, '_build_args', lambda self:[sys.executable,str(script)])
    sess = AgySession(agy_bin=sys.executable, model=None, workdir=tmp_path, prompt='fixture')
    child_pid = None
    try:
        await sess.start()
        async with asyncio.timeout(3):
            while not sess.raw_tail:
                await asyncio.sleep(0.01)
        child_pid = int(sess.raw_tail[-1])
        await sess.aclose()
        async with asyncio.timeout(3):
            while pid_alive(child_pid):  # taskkill / SIGKILL land asynchronously
                await asyncio.sleep(0.02)
    finally:
        await sess.aclose()
        if child_pid:
            kill_pid(child_pid)
