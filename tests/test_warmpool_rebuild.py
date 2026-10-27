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


# ── AgySession deferred prompt / persistent turns ────────────────────────────

async def _until_done(sess):
    seen = []
    while True:
        ev = await sess.next_event(timeout=5)
        seen.append(ev)
        if ev is None or getattr(ev, 'response_text', None) is not None:
            return seen


async def test_deferred_prompt_then_more_turns_on_one_process(fake_agy, tmp_path):
    fake_agy('echoprompt')
    sess = AgySession(agy_bin='agy', model='m', workdir=tmp_path, prompt=None)
    await sess.start()
    assert sess.running and sess.idle_s >= 0
    sess.send('late-prompt')
    first = await _until_done(sess)
    assert any('tail=e-prompt' in (getattr(e, 'response_text', '') or '') for e in first)
    pid = sess.pid
    sess.send('second!!')  # stdin stays open: same process, next turn
    second = await _until_done(sess)
    assert any('tail=second!!' in (getattr(e, 'response_text', '') or '') for e in second)
    assert sess.running and sess.pid == pid and sess.turns_sent == 2
    await sess.aclose(graceful=True)
    assert not sess.running and sess.closed_event.is_set()


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
    assert warm is not None and warm.session.running and warm.conv_id.startswith('pool')
    warm.session.send('x')
    await warm.session.aclose()
    await asyncio.wait_for(pool._refill_task, 5)
    assert pool.idle == 1 and pool.hits == 1 and pool.misses == 1
    await pool.stop()
    assert pool.idle == 0 and not pool.enabled


async def test_pool_discards_stale_signature_and_overaged(settings, fake_agy, tmp_path):
    """size=1: a single miss of another signature does not evict the warm one
    (no thrash); once that signature is clearly hotter the pool re-targets."""
    fake_agy('echoprompt')
    pool = WarmPool(settings, 1, max_idle_s=0.4)
    a = Signature(model='a', workdir=str(tmp_path), agent=None)
    b = Signature(model='b', workdir=str(tmp_path), agent=None)
    await pool.acquire(a)
    await asyncio.wait_for(pool._refill_task, 5)
    pooled = pool._entries[0].session
    assert await pool.acquire(b) is None  # one miss of b: a stays warm
    await asyncio.wait_for(pool._refill_task, 5)
    assert pooled.running and pool._entries[0].signature == a
    assert await pool.acquire(b) is None  # b now clearly hotter -> re-target
    await asyncio.wait_for(pool._refill_task, 5)
    assert not pooled.running and pool._entries[0].signature == b
    await asyncio.sleep(0.45)
    aged = pool._entries[0].session
    assert await pool.acquire(b) is None  # over-aged -> not adopted
    await asyncio.wait_for(pool._refill_task, 5)
    assert not aged.running and pool.idle == 1
    await pool.stop()


async def test_mixed_traffic_keeps_two_signatures_warm(settings, fake_agy, tmp_path):
    """Alternating fresh turns of two signatures (hermes' main loop + its side
    calls): with size=2/signatures=2 every turn after the first of each
    signature is a hit, and no warm entry is discarded for the other one."""
    fake_agy('echoprompt')
    pool = WarmPool(settings, 2, max_signatures=2)
    a = Signature(model='a', workdir=str(tmp_path), agent=None)
    b = Signature(model='b', workdir=str(tmp_path), agent=None)
    results = []
    for sig in [a, b, a, b, a, b]:
        entry = await pool.acquire(sig)
        results.append(entry is not None)
        if entry is not None:
            assert entry.signature == sig
            await entry.session.aclose()
        await asyncio.wait_for(pool._refill_task, 10)
    assert results == [False, False, True, True, True, True]
    assert pool.hits == 4 and pool.misses == 2
    assert sorted(e.signature.model for e in pool._entries) == ['a', 'b']
    await pool.stop()


async def test_boot_eof_is_a_spawn_failure_with_stderr(settings, fake_agy, tmp_path):
    """agy exiting before its first line (logged out, bad model) must raise —
    never return a dead entry the refill would drop and respawn forever."""
    from app.agent_config import SpawnFailed
    fake_agy('bootfail')
    pool = WarmPool(settings, 1)
    with pytest.raises(SpawnFailed, match='not signed in'):
        await pool._spawn_entry(Signature(model='m', workdir=str(tmp_path), agent=None))
    assert pool.idle == 0
    await pool.stop()


async def test_failed_boot_backs_off_instead_of_respawning(settings, tmp_path, monkeypatch, caplog):
    """Repeated boot failures: exponential backoff (one log line per step),
    not a tight respawn loop."""
    import app.warmpool as wp
    from app.agent_config import SpawnFailed
    monkeypatch.setattr(wp, '_BACKOFF_MIN_S', 0.05)
    monkeypatch.setattr(wp, '_BACKOFF_MAX_S', 0.2)
    pool = WarmPool(settings, 1)
    calls = []

    async def failing(sig):
        calls.append(asyncio.get_running_loop().time())
        raise SpawnFailed('agy exited during startup (stderr: not signed in)')
    monkeypatch.setattr(pool, '_spawn_entry', failing)
    try:
        with caplog.at_level('WARNING', logger='agi.warmpool'):
            await pool.acquire(Signature(model='m', workdir=str(tmp_path), agent=None))
            await asyncio.sleep(0.6)
    finally:
        await pool.stop()
    # 0.05 + 0.1 + 0.2 + 0.2 ... -> ~5 attempts; a tight loop would be thousands.
    assert 3 <= len(calls) <= 7
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert gaps[0] >= 0.04 and gaps[1] >= 0.09 and gaps[2] >= 0.19
    steps = [r.getMessage() for r in caplog.records if 'next boot in' in r.getMessage()]
    assert len(steps) == pool.spawn_failures == len(calls) and 'not signed in' in steps[0]
    assert pool._backoff == 0.2 and pool.idle == 0


async def test_pool_spawn_failure_resets_backoff_on_success(settings, fake_agy, tmp_path, monkeypatch):
    import app.warmpool as wp
    monkeypatch.setattr(wp, '_BACKOFF_MIN_S', 0.05)
    fake_agy('bootfail')
    pool = WarmPool(settings, 1)
    sig = Signature(model='m', workdir=str(tmp_path), agent=None)
    await pool.acquire(sig)
    for _ in range(100):
        if pool.spawn_failures:
            break
        await asyncio.sleep(0.02)
    fake_agy('echoprompt')
    await asyncio.wait_for(pool._refill_task, 10)
    assert pool.idle == 1 and pool._backoff == 0.0
    await pool.stop()


async def test_cancelled_refill_does_not_leak_booted_entry(settings, fake_agy, tmp_path, monkeypatch):
    """stop() cancelling the refill right after a boot completed must not
    leave that proc running outside the pool."""
    fake_agy('echoprompt')
    pool = WarmPool(settings, 1)
    sig = Signature(model='m', workdir=str(tmp_path), agent=None)
    booted = []
    orig = pool._spawn_entry

    async def spawn_entry(s):
        entry = await orig(s)
        booted.append(entry)
        return entry
    monkeypatch.setattr(pool, '_spawn_entry', spawn_entry)
    await pool.acquire(sig)
    await asyncio.wait_for(pool._refill_task, 10)
    await pool.stop()
    assert booted and all(not e.session.running for e in booted)


async def test_acquire_waits_for_inflight_boot_of_same_signature(settings, fake_agy, tmp_path, monkeypatch):
    """A request arriving while the pool is booting a proc for exactly its
    signature adopts that proc instead of cold-spawning a second one."""
    fake_agy('echoprompt', FAKE_AGY_BOOT_S='0.6')
    pool = WarmPool(settings, 1)
    sig = Signature(model='m', workdir=str(tmp_path), agent=None)
    assert await pool.acquire(sig) is None  # cold; starts the pooled boot
    for _ in range(100):
        if pool._booting is not None:
            break
        await asyncio.sleep(0.01)
    assert pool._booting is not None
    warm = await pool.acquire(sig)
    assert warm is not None and warm.session.running and pool.hits == 1
    await warm.session.aclose()
    other = await pool.acquire(Signature(model='other', workdir=str(tmp_path), agent=None))
    assert other is None  # a different signature never waits
    await pool.stop()


async def test_disabled_pool_is_inert(settings):
    pool = WarmPool(settings, 0)
    assert await pool.acquire(Signature('m', 'w', None)) is None
    assert pool._refill_task is None and pool.idle == 0
    await pool.stop()


def test_warm_pool_is_on_by_default(monkeypatch):
    monkeypatch.delenv('AGI_WARM_POOL_SIZE', raising=False)
    st = Settings(_env_file=None)
    assert st.warm_pool_size == 2 and st.warm_pool_signatures == 2
    assert st.agent_start_timeout_s == 30


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
    """A follow-up reuses its parked process; the pool is for NEW conversations."""
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
            # Evict the parked process: the follow-up must go the dormant
            # (--conversation) route, which this fake answers with a new id.
            assert await application.state.conv_manager.evict_idle() == 1
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
    stats = application.state.conv_manager.stats
    assert stats['dormant_lost'] == 1 and stats['dormant_resumes'] == 0

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
            # The rebuilt turn's process is parked for reuse, nothing busy.
            assert [c.state for c in application.state.conv_manager._conversations.values()] == ['idle']


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


def test_retarget_recurring_signatures_displace_stale_history(settings, tmp_path):
    """A signature hot in the past (still in the window) yields to two
    signatures that now recur; a one-off never displaces a warm target."""
    pool = WarmPool(settings, 2, max_signatures=2)
    x, a, b, c = (Signature(model=m, workdir=str(tmp_path), agent=None) for m in 'xabc')
    for sig in [x] * 8 + [a, b, a]:
        pool._window.append(sig)
        pool._retarget()
    assert pool._targets == [a, x]
    for sig in [b, c]:  # b recurs -> displaces x; c is a one-off
        pool._window.append(sig)
        pool._retarget()
    assert pool._targets == [b, a]
