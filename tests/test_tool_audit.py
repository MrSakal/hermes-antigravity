"""Tool audit regressions; all configuration and subprocess fixtures are isolated."""
import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from app.agy_session import AgySession, STREAM_CLOSED
from app.config import Settings
from app.conversation import (Conversation, ConversationManager, DoneChunk, ErrorChunk,
                              ExpiredContinuation, SUSPENDED, TextChunk, ToolCallsChunk)
from app.events import TextDelta, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge, ToolResultError
from app.mcp_config import McpConfigManager
from app.openai_models import ChatCompletionRequest, ToolDef


TOOLS = [ToolDef(function={'name': 'allowed', 'parameters': {
    'type': 'object', 'required': ['count'], 'properties': {'count': {'type': 'integer'}},
    'additionalProperties': False,
}})]


class Session:
    def __init__(self, events=(), delay=0):
        self.events = list(events)
        self.delay = delay
        self.aclose = AsyncMock()

    async def next_event(self, timeout=None):
        if self.events:
            await asyncio.sleep(self.delay)
            return self.events.pop(0)
        await asyncio.Event().wait()

    def push_back(self, event):
        self.events.insert(0, event)


@pytest.fixture
def manager(tmp_path):
    return ConversationManager(McpBridge(), McpConfigManager(tmp_path / 'mcp.json'), Settings(
        _env_file=None, default_workdir=tmp_path, mcp_config_path=tmp_path / 'mcp.json',
        agy_brain_path=tmp_path / 'brain', request_timeout_s=1,
    ))


def conversation(manager, name='a', events=(), delay=0):
    bridge = ConversationBridge(name, TOOLS)
    conv = Conversation(name, 'slot_' + name, Session(events, delay), bridge, 'model',
                        workdir=manager.settings.default_workdir.resolve())
    manager._conversations[name] = conv
    manager.mcp.register(bridge)
    return conv


async def collect(manager, conv):
    return [chunk async for chunk in manager.run_turn(conv)]


async def suspend(manager, name='a', count=1):
    conv = conversation(manager, name)
    tasks = [asyncio.create_task(conv.bridge.dispatch('allowed', {'count': i})) for i in range(count)]
    chunks = await collect(manager, conv)
    calls = next(c.calls for c in chunks if isinstance(c, ToolCallsChunk))
    return conv, tasks, calls


def continuation(ids, **kwargs):
    return ChatCompletionRequest(messages=[{'role': 'tool', 'tool_call_id': tid, 'content': 'ok'}
                                           for tid in ids], **kwargs)


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'unknown', 'partial', 'mixed', 'model', 'workdir', 'tools'])
async def test_invalid_continuation_is_atomic(manager, case):
    conv, tasks, calls = await suspend(manager, count=2)
    other, other_tasks, other_calls = await suspend(manager, 'b')
    ids = [c.id for c in calls]
    kwargs = {}
    if case == 'missing': ids[0] = None
    if case == 'duplicate': ids[1] = ids[0]
    if case == 'unknown': ids[1] = 'call_unknown'
    if case == 'partial': ids.pop()
    if case == 'mixed': ids.append(other_calls[0].id)
    if case == 'model': kwargs['model'] = 'different'
    if case == 'workdir': kwargs['workdir'] = str(manager.settings.default_workdir / 'different')
    if case == 'tools': kwargs['tools'] = []
    before = dict(manager._pending_index)
    try:
        with pytest.raises(ExpiredContinuation):
            await manager.resume(continuation(ids, **kwargs))
        assert manager._pending_index == before
        assert conv.state == other.state == SUSPENDED
        assert all(not c.future.done() for c in calls + other_calls)
        assert await manager.resume(continuation([c.id for c in calls], model='alias',
            workdir=str(conv.workdir), tools=TOOLS), model='model', workdir=conv.workdir) is conv
        assert await asyncio.gather(*tasks) == ['ok', 'ok']
    finally:
        await manager.close_all()
        await asyncio.gather(*tasks, *other_tasks, return_exceptions=True)


@pytest.mark.parametrize('events', [[STREAM_CLOSED], [TextDelta('partial'), STREAM_CLOSED]])
async def test_eof_never_success(manager, events):
    conv = conversation(manager, events=events)
    chunks = await collect(manager, conv)
    assert isinstance(chunks[-1], ErrorChunk)
    assert not any(isinstance(c, DoneChunk) for c in chunks)
    conv.session.aclose.assert_awaited_once()


@pytest.mark.parametrize('prefix,expected', [([], ['final']), ([TextDelta('delta')], ['delta'])])
async def test_finalresponse_fallback(manager, prefix, expected):
    chunks = await collect(manager, conversation(manager, events=prefix + [TurnDone(response_text='final')]))
    assert [c.text for c in chunks if isinstance(c, TextChunk)] == expected
    assert isinstance(chunks[-1], DoneChunk)


async def test_absolute_deadline_includes_events(manager):
    conv = conversation(manager, events=[TextDelta('x')] * 5 + [TurnDone()], delay=.3)
    started = time.monotonic()
    chunks = await collect(manager, conv)
    assert time.monotonic() - started < 1.25
    assert isinstance(chunks[-1], ErrorChunk) and chunks[-1].status_code == 504


async def test_absolute_deadline_includes_batch_wait(manager):
    conv = conversation(manager)
    async def delayed_call():
        await asyncio.sleep(.92)
        return await conv.bridge.dispatch('allowed', {'count': 1})
    task = asyncio.create_task(delayed_call())
    started = time.monotonic()
    chunks = await collect(manager, conv)
    assert time.monotonic() - started < 1.25
    assert isinstance(chunks[-1], ErrorChunk) and chunks[-1].status_code == 504
    assert not manager._pending_index and not conv.bridge.pending_ids
    await asyncio.gather(task, return_exceptions=True)


async def test_suspended_capacity_closes_excess_and_releases(manager):
    manager.settings.max_suspended_sessions = 1
    first, tasks, _ = await suspend(manager)
    conv = conversation(manager, 'b')
    release = conv.release_resources = Mock()
    task = asyncio.create_task(conv.bridge.dispatch('allowed', {'count': 1}))
    chunks = await collect(manager, conv)
    assert isinstance(chunks[-1], ErrorChunk) and chunks[-1].status_code == 503
    assert first.state == SUSPENDED and conv.state == 'closed'
    assert not conv.bridge.pending_ids and conv.bridge._incoming.empty()
    assert set(manager._pending_index.values()) == {'a'}
    release.assert_called_once()
    await manager.close_all()
    await asyncio.gather(task, *tasks, return_exceptions=True)


async def test_resources_held_suspended_and_released_once_on_failed_close(manager):
    conv, tasks, _ = await suspend(manager)
    release = conv.release_resources = Mock()
    release.assert_not_called()
    manager.mcp_config.remove_conversation = AsyncMock(side_effect=OSError('disk'))
    conv.session.aclose = AsyncMock(side_effect=RuntimeError('process'))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='process'):
            await manager._close(conv)
    release.assert_called_once()
    conv.session.aclose.assert_awaited_once()
    assert conv.release_resources is None
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_concurrent_close_keeps_resources_until_process_closed(manager):
    conv = conversation(manager)
    entered, finish = asyncio.Event(), asyncio.Event()
    async def close():
        entered.set()
        await finish.wait()
    conv.session.aclose = AsyncMock(side_effect=close)
    release = conv.release_resources = Mock()
    first = asyncio.create_task(manager._close(conv))
    await entered.wait()
    second = asyncio.create_task(manager._close(conv))
    await asyncio.sleep(0)
    release.assert_not_called()
    first.cancel()
    finish.set()
    await asyncio.gather(first, second)
    release.assert_called_once()
    conv.session.aclose.assert_awaited_once()


@pytest.mark.parametrize('failure', [OSError('disk'), asyncio.CancelledError()])
async def test_config_failure_unregisters(manager, failure):
    manager.mcp_config.add_conversation = AsyncMock(side_effect=failure)
    with pytest.raises(type(failure)):
        await manager.create(ChatCompletionRequest(messages=[{'role': 'user', 'content': 'go'}]),
                             model='model', workdir=manager.settings.default_workdir)
    assert not manager.mcp._registry and not manager._conversations


@pytest.mark.parametrize('failure', [RuntimeError('start'), asyncio.CancelledError()])
async def test_start_failure_rolls_back_real_config(manager, monkeypatch, failure):
    closed = AsyncMock()
    monkeypatch.setattr(AgySession, 'start', AsyncMock(side_effect=failure))
    monkeypatch.setattr(AgySession, 'aclose', closed)
    with pytest.raises(type(failure)):
        await manager.create(ChatCompletionRequest(messages=[{'role': 'user', 'content': 'go'}]),
                             model='model', workdir=manager.settings.default_workdir)
    assert not manager.mcp._registry and not manager._conversations
    assert not manager.mcp_config._owned
    closed.assert_awaited_once()


async def test_cancellation_during_slot_allocation_rolls_back(manager):
    original = manager.mcp_config.add_conversation
    entered, finish = asyncio.Event(), asyncio.Event()
    async def allocate(*args):
        slot = await original(*args)
        entered.set()
        await finish.wait()
        return slot
    manager.mcp_config.add_conversation = allocate
    task = asyncio.create_task(manager.create(ChatCompletionRequest(messages=[{'role': 'user', 'content': 'go'}]),
                              model='model', workdir=manager.settings.default_workdir))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not manager.mcp._registry and not manager.mcp_config._owned
