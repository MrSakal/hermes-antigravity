"""WP2: persistent processes, cross-turn reuse, dormant resume, LRU/TTL,
warm pool for tool turns, duplicate continuations and lifecycle hardening —
all against the persistent fake agy (init at boot, one stdin line per turn)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import app.conversation as conversation_mod
import app.main as main
import app.routes.chat as chat
from app.agent_config import AgentPluginManager
from app.config import Settings
from app.conversation import (
    ConversationManager,
    DoneChunk,
    DuplicateContinuation,
    ErrorChunk,
    TextChunk,
    ToolCallsChunk,
    _Dormant,
)
from app.mcp_bridge import McpBridge
from app.openai_models import ChatCompletionRequest

TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
OTHER_TOOL = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}


def starts(log_path):
    if not log_path.exists():
        return []
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line.get("start")]


def turns(log_path):
    if not log_path.exists():
        return []
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if "turn" in line]


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "ws").mkdir()
    return Settings(_env_file=None, default_workdir=tmp_path / "ws",
                    mcp_config_path=tmp_path / "mcp_config.json",
                    agent_plugin_path=tmp_path / "plugin", agy_brain_path=tmp_path / "brain",
                    request_timeout_s=20, warm_pool_size=0)


@pytest.fixture
def log(tmp_path, fake_agy, monkeypatch):
    path = tmp_path / "agy.log"
    monkeypatch.setenv("FAKE_AGY_LOG", str(path))
    monkeypatch.setenv("FAKE_AGY_PLUGIN", str(tmp_path / "plugin"))
    return path


@pytest.fixture
async def mgr(settings, log):
    m = ConversationManager(McpBridge(), settings, agents=AgentPluginManager(settings.agent_plugin_path))
    yield m
    await m.close_all()


async def collect(mgr, conv):
    return [c async for c in mgr.run_turn(conv)]


def text_of(chunks):
    return "".join(c.text for c in chunks if isinstance(c, TextChunk))


async def turn(mgr, messages, **kw):
    conv = await mgr.start_turn(ChatCompletionRequest(messages=messages, **kw), model="m",
                                workdir=mgr.settings.default_workdir)
    return conv, await collect(mgr, conv)


# ── cross-turn reuse ─────────────────────────────────────────────────────────


async def test_follow_up_sends_only_newest_message_to_same_process(mgr, fake_agy, log):
    fake_agy("echo")
    m1 = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "first"}]
    conv, chunks = await turn(mgr, m1)
    assert text_of(chunks) == "echo: first" and conv.state == "idle"
    m2 = m1 + [{"role": "assistant", "content": "echo: first"}, {"role": "user", "content": "second"}]
    conv2, chunks2 = await turn(mgr, m2)
    assert conv2 is conv and text_of(chunks2) == "echo: second"
    assert len(starts(log)) == 1
    assert [t["content"] for t in turns(log)] == ["first", "second"]
    assert mgr.stats["reuse_hits"] == 1


async def test_different_signature_never_reuses(mgr, fake_agy, log):
    fake_agy("echo")
    m1 = [{"role": "user", "content": "first"}]
    conv, _ = await turn(mgr, m1)
    m2 = m1 + [{"role": "assistant", "content": "echo: first"}, {"role": "user", "content": "x"}]
    # Another system prompt = another agent file = another process.
    other, _ = await turn(mgr, [{"role": "system", "content": "new persona"}] + m2)
    assert other is not conv and len(starts(log)) == 2
    assert mgr.stats["reuse_misses"] == 1


async def test_abandoned_turn_is_killed_never_parked(mgr, fake_agy, log):
    """A turn the client walked away from diverged from what the client saw:
    the process dies (agy has no per-turn cancel) and nothing is parked."""
    fake_agy("hang", FAKE_AGY_HANG_S="30")
    conv = await mgr.start_turn(ChatCompletionRequest(messages=[{"role": "user", "content": "q"}]),
                                model="m", workdir=mgr.settings.default_workdir)
    task = asyncio.create_task(collect(mgr, conv))
    for _ in range(100):
        if conv.driven:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()  # a second cancel must not interrupt the kill
    with pytest.raises(asyncio.CancelledError):
        await task
    assert conv.state == "closed" and not conv.session.running
    assert conv.session.closed_event.is_set()
    assert not mgr._idle and not mgr._dormant


async def test_tool_loop_then_follow_up_reuses_tool_process(mgr, fake_agy, log, tmp_path):
    release = tmp_path / "release"
    fake_agy("hang;echo", FAKE_AGY_RELEASE=str(release), FAKE_AGY_HANG_S="20")
    msgs = [{"role": "user", "content": "weather?"}]
    conv = await mgr.start_turn(ChatCompletionRequest(messages=msgs, tools=[TOOL]),
                                model="m", workdir=mgr.settings.default_workdir)
    call = asyncio.create_task(conv.bridge.dispatch("get_weather", {"city": "Bp"}))
    chunks = await collect(mgr, conv)
    pc = next(c for c in chunks if isinstance(c, ToolCallsChunk)).calls[0]
    assistant = {"role": "assistant", "content": text_of(chunks), "tool_calls": [pc.openai_tool_call()]}
    cont = msgs + [assistant, {"role": "tool", "tool_call_id": pc.id, "content": "sunny"}]
    resumed = await mgr.resume(ChatCompletionRequest(messages=cont, tools=[TOOL]))
    assert await asyncio.wait_for(call, 5) == "sunny"
    release.write_text("go")
    final = await collect(mgr, resumed)
    assert isinstance(final[-1], DoneChunk) and conv.state == "idle"
    follow = cont + [{"role": "assistant", "content": text_of(final)},
                     {"role": "user", "content": "thanks"}]
    again, chunks3 = await turn(mgr, follow, tools=[TOOL])
    assert again is conv and text_of(chunks3) == "echo: thanks"
    assert len(starts(log)) == 1


# ── dormant resume, LRU, TTL ─────────────────────────────────────────────────


async def test_lru_eviction_keeps_conversation_resumable(mgr, fake_agy, log):
    fake_agy("echo")
    mgr.settings.max_sessions = 1
    a1 = [{"role": "user", "content": "alpha"}]
    conv_a, _ = await turn(mgr, a1)
    conv_b, _ = await turn(mgr, [{"role": "user", "content": "beta"}])
    # Room for B was made by evicting idle A (LRU), remembered dormant.
    assert conv_a.state == "closed" and conv_b.state == "idle"
    assert mgr.stats["evictions"] == 1 and len(mgr._dormant) == 1
    follow = a1 + [{"role": "assistant", "content": "echo: alpha"}, {"role": "user", "content": "more"}]
    conv_a2, chunks = await turn(mgr, follow)
    assert text_of(chunks) == "echo: more"  # only the newest message, on A's agy conversation
    last_start = starts(log)[-1]["argv"]
    assert last_start[last_start.index("--conversation") + 1] == "conv-fake-1"
    assert mgr.stats["dormant_resumes"] == 1
    assert conv_b.state == "closed"  # evicted in turn to stay within max_sessions=1
    assert mgr.count() == 1


async def test_idle_ttl_gc_then_resume(mgr, fake_agy, log):
    fake_agy("echo")
    m1 = [{"role": "user", "content": "q"}]
    conv, _ = await turn(mgr, m1)
    conv.last_activity -= mgr.settings.idle_session_ttl_s + 1
    assert await mgr.gc_once() == 1
    assert conv.state == "closed" and conv.session.closed_event.is_set()
    assert conv.session._proc.returncode == 0  # graceful: stdin EOF, not a kill
    follow = m1 + [{"role": "assistant", "content": "echo: q"}, {"role": "user", "content": "r"}]
    _, chunks = await turn(mgr, follow)
    assert text_of(chunks) == "echo: r" and mgr.stats["dormant_resumes"] == 1


async def test_revival_waits_for_predecessor_exit(monkeypatch):
    gone = asyncio.Event()
    monkeypatch.setattr(conversation_mod, "_DORMANT_EXIT_WAIT_S", 0.1)
    assert await ConversationManager._predecessor_gone(_Dormant("x", gone)) is False
    monkeypatch.setattr(conversation_mod, "_DORMANT_EXIT_WAIT_S", 5)
    waiter = asyncio.create_task(ConversationManager._predecessor_gone(_Dormant("x", gone)))
    await asyncio.sleep(0.05)
    assert not waiter.done()  # a resume never overlaps the previous process
    gone.set()
    assert await waiter is True


async def test_dead_idle_process_is_resumed_not_reused(mgr, fake_agy, log):
    fake_agy("echo")
    m1 = [{"role": "user", "content": "q"}]
    conv, _ = await turn(mgr, m1)
    conv.session.kill_now()
    for _ in range(100):
        if not conv.session.running:
            break
        await asyncio.sleep(0.02)
    follow = m1 + [{"role": "assistant", "content": "echo: q"}, {"role": "user", "content": "r"}]
    conv2, chunks = await turn(mgr, follow)
    assert conv2 is not conv and text_of(chunks) == "echo: r"
    assert mgr.stats["dormant_resumes"] == 1


# ── hardening ───────────────────────────────────────────────────────────────


async def test_death_mid_turn_after_partial_text_is_an_error(mgr, fake_agy, log):
    fake_agy("simple;die")
    m1 = [{"role": "user", "content": "q"}]
    conv, _ = await turn(mgr, m1)
    follow = m1 + [{"role": "assistant", "content": "Hello world"}, {"role": "user", "content": "r"}]
    _, chunks = await turn(mgr, follow)
    assert text_of(chunks) == "partial "
    assert isinstance(chunks[-1], ErrorChunk) and chunks[-1].status_code == 502
    assert "exited mid-turn" in chunks[-1].message
    assert not any(isinstance(c, DoneChunk) for c in chunks)
    assert conv.state == "closed" and not mgr._idle


async def test_claimed_but_undriven_turn_is_reaped(mgr, fake_agy, log, monkeypatch):
    fake_agy("echo")
    conv = await mgr.start_turn(ChatCompletionRequest(messages=[{"role": "user", "content": "q"}]),
                                model="m", workdir=mgr.settings.default_workdir)
    assert conv.state == "running" and not conv.driven
    monkeypatch.setattr(conversation_mod, "_UNDRIVEN_TTL_S", 0.0)
    conv.last_activity -= 1
    assert await mgr.gc_once() == 1
    assert conv.state == "closed" and not conv.session.running


async def test_stale_request_cleanup_never_kills_a_later_turn(mgr, fake_agy, log):
    fake_agy("echo")
    m1 = [{"role": "user", "content": "q"}]
    conv, _ = await turn(mgr, m1)
    old = conv.turn_token
    follow = m1 + [{"role": "assistant", "content": "echo: q"}, {"role": "user", "content": "r"}]
    again = await mgr.start_turn(ChatCompletionRequest(messages=follow), model="m",
                                 workdir=mgr.settings.default_workdir)
    assert again is conv and conv.state == "running"
    await mgr.abandon(conv, old)  # request 1's late cleanup
    assert conv.state == "running" and conv.session.running
    assert text_of(await collect(mgr, conv)) == "echo: r"


# ── HTTP: duplicates, pool for tool turns, metrics ──────────────────────────


@pytest.fixture
def application(settings, monkeypatch, log):
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(chat, "get_settings", lambda: settings)
    return main.create_app()


async def _wait_for(predicate, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def test_duplicate_of_finished_continuation_is_409_not_rebuilt(application, fake_agy, log, tmp_path):
    release = tmp_path / "release"
    fake_agy("hang", FAKE_AGY_RELEASE=str(release), FAKE_AGY_HANG_S="20")
    mgr = application.state.conv_manager
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            msgs = [{"role": "user", "content": "weather?"}]
            first = asyncio.create_task(client.post("/v1/chat/completions",
                                                    json={"messages": msgs, "tools": [TOOL]}))
            await _wait_for(lambda: mgr._conversations)
            conv = next(iter(mgr._conversations.values()))
            call = asyncio.create_task(conv.bridge.dispatch("get_weather", {"city": "Bp"}))
            r1 = await first
            assert r1.json()["choices"][0]["finish_reason"] == "tool_calls"
            message = r1.json()["choices"][0]["message"]
            cont = msgs + [message] + [{"role": "tool", "tool_call_id": tc["id"], "content": "sunny"}
                                       for tc in message["tool_calls"]]
            release.write_text("go")
            r2 = await client.post("/v1/chat/completions", json={"messages": cont, "tools": [TOOL]})
            assert r2.status_code == 200 and r2.json()["choices"][0]["finish_reason"] == "stop"
            assert await call == "sunny"
            # The tool loop finished; a late duplicate of its last continuation
            # must not rebuild (= re-run the turn and its side effects).
            r3 = await client.post("/v1/chat/completions", json={"messages": cont, "tools": [TOOL]})
            assert r3.status_code == 409 and r3.json()["error"]["code"] == "tool_result_expired"
            assert len(starts(log)) == 1
            metrics = (await client.get("/metrics")).json()
            assert metrics["duplicate_continuations"] == 1 and metrics["sessions_active"] == 0


async def test_warm_pool_serves_tool_turns_with_late_bound_tools(settings, monkeypatch, fake_agy, log, tmp_path):
    fake_agy("echo")
    settings.warm_pool_size = 1
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(chat, "get_settings", lambda: settings)
    application = main.create_app()
    mgr = application.state.conv_manager
    pool = application.state.warm_pool
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            r1 = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "one"}], "tools": [TOOL]})
            assert r1.status_code == 200
            await asyncio.wait_for(pool._refill_task, 10)
            assert pool.idle == 1
            [entry] = pool._entries
            # Pre-minted conv id + registered bridge + URL baked into the agent
            # file the pooled process booted with (read at boot, under the lock).
            assert entry.conv_id.startswith("pool") and entry.bridge is not None
            assert mgr.mcp.get(entry.conv_id) is entry.bridge
            boot = next(s for s in starts(log) if f"/mcp/{entry.conv_id}" in (s["agent_file"] or ""))
            assert "inheritMcp: false" in boot["agent_file"]
            assert "hermes-" in boot["agent_file"]

            r2 = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "two"}], "tools": [TOOL]})
            assert r2.status_code == 200
            content = r2.json()["choices"][0]["message"]["content"]
            assert content.startswith("echo: two") and "get_weather" in content  # MCP hint
            conv = mgr._conversations[entry.conv_id]
            assert [t.function.name for t in conv.bridge.tools] == ["get_weather"]
            metrics = (await client.get("/metrics")).json()
            assert metrics["warm_pool_tool_hits"] == 1 and metrics["warm_pool_hits"] == 1

            # A different toolset never adopts the pooled tool process.
            await asyncio.wait_for(pool._refill_task, 10)
            hits = pool.hits
            r3 = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "three"}], "tools": [OTHER_TOOL]})
            assert r3.status_code == 200 and pool.hits == hits
    assert not (tmp_path / "mcp_config.json").exists()
    agents = sorted(p.name for p in (tmp_path / "plugin" / "agents").iterdir())
    assert agents == [".hermes.md.lock", "hermes.md"]  # + the cross-process spawn lock


async def test_stream_death_mid_turn_reports_error_not_stop(application, fake_agy, log):
    fake_agy("die")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.post("/v1/chat/completions", json={
            "stream": True, "messages": [{"role": "user", "content": "q"}]})
        r2 = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "q"}]})
    await application.state.conv_manager.close_all()
    payloads = [json.loads(c[6:]) for c in r.text.split("\n\n")
                if c.startswith("data: ") and c != "data: [DONE]"]
    assert any("error" in p and "exited mid-turn" in p["error"]["message"] for p in payloads)
    assert not any(p.get("choices", [{}])[0].get("finish_reason") == "stop" for p in payloads)
    assert r2.status_code == 502 and "exited mid-turn" in r2.json()["error"]["message"]


# ── review fixes: boot failure, cancellation-safe cleanup, dead adopt ───────


async def test_cold_boot_failure_is_a_clean_502_with_stderr(application, fake_agy, log):
    fake_agy("bootfail")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "q"}]})
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["code"] == "upstream_start_failed" and "not signed in" in err["message"]
    mgr = application.state.conv_manager
    assert not mgr._conversations and not mgr.mcp._registry and mgr._spawning == 0


async def test_cancel_during_spawn_cleanup_still_closes_evicted_victims(mgr, fake_agy, log):
    """A cancellation landing in _spawn's cleanup (here: while the manager lock
    is contended) must neither leak the spawn slot nor strand LRU victims
    CLOSING forever."""
    fake_agy("echo")
    mgr.settings.max_sessions = 1
    victim, _ = await turn(mgr, [{"role": "user", "content": "alpha"}])
    assert victim.state == "idle"
    entered, release = asyncio.Event(), asyncio.Event()
    original = mgr.agents.spawn

    async def blocked(session, content, **kw):
        entered.set()
        await release.wait()
        return await original(session, content, **kw)
    mgr.agents.spawn = blocked
    task = asyncio.create_task(mgr.start_turn(
        ChatCompletionRequest(messages=[{"role": "user", "content": "beta"}]),
        model="m", workdir=mgr.settings.default_workdir))
    await entered.wait()
    assert victim.state == "closing"
    await mgr._lock.acquire()  # victim teardown now has to wait for the lock
    try:
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()  # second cancel while the cleanup is waiting
        with pytest.raises(asyncio.CancelledError):
            await task
        assert mgr._spawning == 0
    finally:
        mgr._lock.release()
    await _wait_for(lambda: victim.session.closed_event.is_set())
    assert victim.state == "closed" and not mgr._conversations


async def test_gc_reaps_conversation_stuck_closing(mgr, fake_agy, log):
    fake_agy("echo")
    conv, _ = await turn(mgr, [{"role": "user", "content": "q"}])
    async with mgr._lock:
        mgr._retire_locked(conv, dormant=True)  # owner cancelled before closing it
    assert conv.state == "closing" and await mgr.gc_once() == 0  # within grace
    conv.closing_since -= conversation_mod._CLOSING_GRACE_S + 1
    assert await mgr.gc_once() == 1
    assert conv.state == "closed" and conv.session.closed_event.is_set()


async def test_process_dead_at_resume_send_is_502_not_500(mgr, fake_agy, log, monkeypatch):
    from app.agy_session import AgySession
    from app.errors import OpenAIError

    fake_agy("echo")
    m1 = [{"role": "user", "content": "q"}]
    conv, _ = await turn(mgr, m1)
    conv.last_activity -= mgr.settings.idle_session_ttl_s + 1
    assert await mgr.gc_once() == 1  # parked dormant
    real_send = AgySession.send

    def dead_send(self, prompt):
        if self.resumed:
            raise RuntimeError("agy process is not running (exit code 1)")
        return real_send(self, prompt)
    monkeypatch.setattr(AgySession, "send", dead_send)
    follow = m1 + [{"role": "assistant", "content": "echo: q"}, {"role": "user", "content": "r"}]
    with pytest.raises(OpenAIError) as exc:
        await mgr.start_turn(ChatCompletionRequest(messages=follow), model="m",
                             workdir=mgr.settings.default_workdir)
    assert exc.value.status_code == 502 and exc.value.code == "upstream_exited"
    assert not mgr._conversations
