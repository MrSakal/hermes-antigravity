"""Warm pool and rebuild-on-expiry (ported from claude-code-openai-server)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import app.main as main
import app.routes.chat as chat
from app.agy_session import AgySession
from app.config import Settings
from app.conversation import DuplicateContinuation, ExpiredContinuation
from app.warmpool import Signature, WarmPool


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, default_workdir=tmp_path / 'ws',
                    mcp_config_path=tmp_path / 'mcp.json', agent_plugin_path=tmp_path / 'plugin',
                    agy_brain_path=tmp_path / 'brain', warm_pool_size=1)


@pytest.fixture
def application(settings, monkeypatch):
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr(chat, 'get_settings', lambda: settings)
    return main.create_app()


def request(messages=None, **kw):
    return {'messages': messages or [{'role': 'user', 'content': 'hi'}], **kw}


# ── AgySession deferred prompt ────────────────────────────────────────────────

async def test_deferred_prompt_is_sent_exactly_once(fake_agy, tmp_path):
    fake_agy('echoprompt')
    sess = AgySession(agy_bin='agy', model='m', workdir=tmp_path, prompt=None)
    await sess.start()
    assert sess.running and sess.idle_s >= 0
    sess.send('late-prompt')
    with pytest.raises(RuntimeError, match='already sent'):
        sess.send('again')
    seen = []
    while True:
        ev = await sess.next_event(timeout=5)
        seen.append(ev)
        if getattr(ev, 'response_text', None) or ev is None:
            break
    await sess.aclose()
    assert any('tail=e-prompt' in (getattr(e, 'response_text', '') or '') for e in seen)


async def test_send_before_start_rejected(tmp_path):
    sess = AgySession(agy_bin='agy', model='m', workdir=tmp_path, prompt=None)
    with pytest.raises(RuntimeError, match='not started'):
        sess.send('x')


# ── WarmPool ──────────────────────────────────────────────────────────────────

async def test_pool_miss_then_hit_same_signature(settings, fake_agy, tmp_path):
    fake_agy('echoprompt')
    pool = WarmPool(settings, 1)
    sig = Signature(model='m', workdir=str(tmp_path), agent=None)
    assert await pool.acquire(sig) is None  # cold: re-targets and refills
    await asyncio.wait_for(pool._refill_task, 5)
    assert pool.idle == 1
    warm = await pool.acquire(sig)
    assert warm is not None and warm.running
    warm.send('x')
    await warm.aclose()
    await asyncio.wait_for(pool._refill_task, 5)
    assert pool.idle == 1 and pool.hits == 1 and pool.misses == 1
    await pool.stop()
    assert pool.idle == 0 and not pool.enabled


async def test_pool_discards_stale_signature_and_overaged(settings, fake_agy, tmp_path):
    fake_agy('echoprompt')
    pool = WarmPool(settings, 1, max_idle_s=0.2)
    a = Signature(model='a', workdir=str(tmp_path), agent=None)
    b = Signature(model='b', workdir=str(tmp_path), agent=None)
    await pool.acquire(a)
    await asyncio.wait_for(pool._refill_task, 5)
    pooled = pool._entries[0][1]
    assert await pool.acquire(b) is None  # signature mismatch -> miss, old one dropped
    await asyncio.wait_for(pool._refill_task, 5)
    assert not pooled.running and pool._entries[0][0] == b
    await asyncio.sleep(0.25)
    aged = pool._entries[0][1]
    assert await pool.acquire(b) is None  # over-aged -> not adopted
    await asyncio.wait_for(pool._refill_task, 5)
    assert not aged.running and pool.idle == 1
    await pool.stop()


async def test_disabled_pool_is_inert(settings):
    pool = WarmPool(settings, 0)
    assert await pool.acquire(Signature('m', 'w', None)) is None
    assert pool._refill_task is None and pool.idle == 0
    await pool.stop()


async def test_route_adopts_pooled_proc_and_metrics(application, settings, fake_agy):
    fake_agy('simple')
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
            first = await client.post('/v1/chat/completions', json=request())
            assert first.status_code == 200
            pool: WarmPool = application.state.warm_pool
            await asyncio.wait_for(pool._refill_task, 5)
            assert pool.idle == 1
            second = await client.post('/v1/chat/completions', json=request([{'role': 'user', 'content': 'again'}]))
            assert second.status_code == 200
            assert second.json()['choices'][0]['message']['content'] == 'Hello world'
            metrics = (await client.get('/metrics')).json()
            assert metrics['warm_pool_hits'] == 1 and metrics['warm_pool_misses'] == 1
            assert metrics['sessions_active'] == 0
    assert application.state.warm_pool.idle == 0  # lifespan drained it


async def test_conversation_resume_never_uses_pool(application, settings, fake_agy):
    """--conversation is spawn-time argv; a cache hit must cold-spawn."""
    fake_agy('simple', FAKE_AGY_CONV_ID='conv-pool-1')
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
            r = await client.post('/v1/chat/completions', json=request())
            pool: WarmPool = application.state.warm_pool
            await asyncio.wait_for(pool._refill_task, 5)
            follow = [{'role': 'user', 'content': 'hi'},
                      {'role': 'assistant', 'content': r.json()['choices'][0]['message']['content']},
                      {'role': 'user', 'content': 'more'}]
            r2 = await client.post('/v1/chat/completions', json=request(follow))
            assert r2.status_code == 200
            assert pool.hits == 0 and pool.idle == 1


# ── rebuild on expiry ─────────────────────────────────────────────────────────

async def test_lost_agy_conversation_falls_back_to_full_fold(application, settings, fake_agy):
    """agy silently starts a NEW conversation when --conversation points at a
    pruned DB (stderr warning only). The bridge must detect the id mismatch and
    refold the whole history rather than answer from the last message alone."""
    fake_agy('lostconv')
    settings.warm_pool_size = 0
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
            r = await client.post('/v1/chat/completions', json=request())
            assert r.status_code == 200
            follow = [{'role': 'user', 'content': 'hi'},
                      {'role': 'assistant', 'content': r.json()['choices'][0]['message']['content']},
                      {'role': 'user', 'content': 'which word?'}]
            r2 = await client.post('/v1/chat/completions', json=request(follow))
            assert r2.status_code == 200, r2.text
            content = r2.json()['choices'][0]['message']['content']
            # The SECOND spawn (after the id-mismatch fallback) sees the folded
            # transcript ("Conversation so far: ..."), not just "which word?".
            assert content.startswith('len=') and int(content.split(';')[0][4:]) > 60, content
            metrics = (await client.get('/metrics')).json()
            assert metrics['sessions_active'] == 0 and metrics['http_active'] == 0
    # The stale cache entry was consumed; the fresh conversation was stored under the new id.
    assert len(application.state.conv_cache) == 1

def tool_messages(call_id='call_gone'):
    return [
        {'role': 'user', 'content': 'call the tool'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': 'echo', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': call_id, 'content': 'FIXTURE-RESULT'},
    ]


TOOL = {'type': 'function', 'function': {'name': 'echo', 'parameters': {'type': 'object', 'properties': {}}}}


def test_fold_carries_tool_calls_and_results():
    from app.openai_models import ChatMessage
    from app.translate import fold_conversation
    folded = fold_conversation([ChatMessage(**m) for m in tool_messages('call_x')])
    assert '[called tools: echo({})]' in folded
    assert 'Tool: [result of call_x] FIXTURE-RESULT' in folded


async def test_expired_continuation_rebuilds_from_history(application, settings, fake_agy):
    fake_agy('echoprompt')
    settings.warm_pool_size = 0
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
            r = await client.post('/v1/chat/completions', json=request(tool_messages(), tools=[TOOL]))
            assert r.status_code == 200, r.text
            # The fake echoes prompt length; the folded 3-message transcript plus the
            # MCP hint is far longer than a lone "call the tool" user turn (~14 chars).
            content = r.json()['choices'][0]['message']['content']
            assert int(content.split(';')[0].split('=')[1]) > 150, content
            metrics = (await client.get('/metrics')).json()
            assert metrics['sessions_active'] == 0 and metrics['mcp_transport_sessions'] == 0
            assert not application.state.conv_manager._conversations


async def test_expired_continuation_409_when_rebuild_disabled(application, settings, fake_agy):
    fake_agy('echoprompt')
    settings.rebuild_on_expiry = False
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
            r = await client.post('/v1/chat/completions', json=request(tool_messages(), tools=[TOOL]))
    assert r.status_code == 409
    assert r.json()['error']['code'] == 'tool_result_expired'
    assert application.state.admission.active == 0


async def test_live_conversation_wrong_ids_is_duplicate_not_rebuilt(application, settings, fake_agy):
    """A conversation that exists but is not waiting on these ids must stay 409."""
    fake_agy('hang', FAKE_AGY_HANG_S='30')
    mgr = application.state.conv_manager
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
            task = asyncio.create_task(client.post('/v1/chat/completions', json=request(tools=[TOOL])))
            for _ in range(100):
                if mgr._conversations:
                    break
                await asyncio.sleep(0.05)
            conv = next(iter(mgr._conversations.values()))
            # Simulate a suspended conversation with a different pending id.
            conv.state = 'suspended'
            mgr._pending_index['call_other'] = conv.conv_id
            from app.openai_models import ChatCompletionRequest
            req = ChatCompletionRequest(messages=tool_messages('call_other'), tools=[TOOL])
            req2 = ChatCompletionRequest(messages=tool_messages('call_unknown'), tools=[TOOL])
            with pytest.raises(DuplicateContinuation):
                await mgr.resume_or_rebuild(req, model=conv.model, workdir=conv.workdir)
            with pytest.raises(ExpiredContinuation):
                mgr.settings.rebuild_on_expiry = False
                try:
                    await mgr.resume_or_rebuild(req2, model=conv.model, workdir=conv.workdir)
                finally:
                    mgr.settings.rebuild_on_expiry = True
            mgr._pending_index.pop('call_other', None)
            conv.state = 'running'
            await mgr._close(conv)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
