"""No concurrency caps: this is a provider, so every call gets its own agy
process immediately — no admission control, no queue cap, no 503 for
capacity, no LRU eviction of idle processes, no cap on parked tool turns."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import app.main as main
import app.routes.chat as chat
from app.agent_config import AgentPluginManager
from app.config import Settings
from app.conversation import ConversationManager, TextChunk, ToolCallsChunk
from app.mcp_bridge import McpBridge
from app.openai_models import ChatCompletionRequest

N = 30
TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "ws").mkdir()
    return Settings(_env_file=None, default_workdir=tmp_path / "ws",
                    mcp_config_path=tmp_path / "mcp_config.json",
                    agent_plugin_path=tmp_path / "plugin", agy_brain_path=tmp_path / "brain",
                    request_timeout_s=120, warm_pool_size=0)


@pytest.fixture
def plugin_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AGY_PLUGIN", str(tmp_path / "plugin"))


async def _wait_for(predicate, timeout=60.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


def test_capacity_knobs_are_gone():
    s = Settings(_env_file=None)
    for knob in ("max_sessions", "max_suspended_sessions", "max_queued_requests",
                 "admission_timeout_s"):
        assert not hasattr(s, knob), knob


async def test_concurrent_requests_each_get_their_own_process(
        settings, monkeypatch, fake_agy, plugin_env, tmp_path):
    """N concurrent fresh chats (distinct system prompts, so none can reuse)
    are all in flight at once on N processes and all answer 200."""
    release = tmp_path / "release"
    fake_agy("hang", FAKE_AGY_RELEASE=str(release), FAKE_AGY_HANG_S="90")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(chat, "get_settings", lambda: settings)
    application = main.create_app()
    async with application.router.lifespan_context(application):
        mgr = application.state.conv_manager
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=120) as client:
            tasks = [asyncio.create_task(client.post("/v1/chat/completions", json={"messages": [
                {"role": "system", "content": f"You are assistant #{i}."},
                {"role": "user", "content": f"q{i}"}]})) for i in range(N)]
            # Every request holds its own RUNNING process simultaneously.
            await _wait_for(lambda: mgr.count("running") == N)
            metrics = (await client.get("/metrics")).json()
            assert metrics["processes_live"] == N and metrics["processes_running"] == N
            assert "sessions_capacity" not in metrics and "sessions_waiting" not in metrics
            pids = {c.session.pid for c in mgr._conversations.values()}
            assert len(pids) == N
            release.write_text("go")
            results = await asyncio.gather(*tasks)
            assert [r.status_code for r in results] == [200] * N, \
                [r.text for r in results if r.status_code != 200][:3]
            metrics = (await client.get("/metrics")).json()
            assert metrics["processes_idle"] == N and metrics["http_active"] == 0
            assert not any(k.startswith("http_5") for k in metrics["counters"])


async def test_many_idle_processes_are_never_evicted(settings, fake_agy, plugin_env):
    fake_agy("echo")
    mgr = ConversationManager(McpBridge(), settings,
                              agents=AgentPluginManager(settings.agent_plugin_path))
    try:
        convs = []
        for i in range(12):
            req = ChatCompletionRequest(messages=[{"role": "system", "content": f"s{i}"},
                                                  {"role": "user", "content": f"q{i}"}])
            conv = await mgr.start_turn(req, model="m", workdir=settings.default_workdir)
            chunks = [c async for c in mgr.run_turn(conv)]
            assert "".join(c.text for c in chunks if isinstance(c, TextChunk)) == f"echo: q{i}"
            convs.append(conv)
        assert all(c.state == "idle" and c.session.running for c in convs)
        assert mgr.count() == 12
        assert await mgr.gc_once() == 0  # within the idle TTL nothing leaves
    finally:
        await mgr.close_all()


async def test_many_suspended_tool_turns_are_all_parked(settings, fake_agy, plugin_env, tmp_path):
    """No cap on conversations parked mid tool call: none is refused (503)."""
    fake_agy("hang", FAKE_AGY_RELEASE=str(tmp_path / "never"), FAKE_AGY_HANG_S="30")
    mgr = ConversationManager(McpBridge(), settings,
                              agents=AgentPluginManager(settings.agent_plugin_path))
    calls = []
    try:
        for i in range(8):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": f"q{i}"}],
                                        tools=[TOOL])
            conv = await mgr.start_turn(req, model="m", workdir=settings.default_workdir)
            calls.append(asyncio.create_task(conv.bridge.dispatch("get_weather", {"city": "Bp"})))
            chunks = [c async for c in mgr.run_turn(conv)]
            assert isinstance(chunks[-1], ToolCallsChunk), chunks[-1]
        assert mgr.count("suspended") == 8
    finally:
        await mgr.close_all()
        await asyncio.gather(*calls, return_exceptions=True)


async def test_request_spawns_boot_in_parallel(tmp_path):
    """No agent-file lock: 40 boots of 0.2 s each overlap completely."""
    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=5)

    class Session:
        agent = None

        async def start(self):
            pass

        async def wait_started(self, timeout):
            await asyncio.sleep(0.2)
            return True

        async def aclose(self):
            pass

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await asyncio.gather(*(mgr.spawn(Session(), f"agent {i}") for i in range(40)))
    assert loop.time() - t0 < 2.0  # serialized would be 8 s
    assert mgr.transient_files() == [] and mgr.booting == 0


async def test_ten_real_boots_do_not_serialize(settings, fake_agy, tmp_path, monkeypatch):
    """10 fresh conversations against a fake agy that needs 0.5 s to boot
    (agent file read at boot, then the init line): together well under the
    5 s a serialized spawn path would take; every process booted with its own
    system prompt, and no transient agent file is left behind."""
    fake_agy("echo", FAKE_AGY_BOOT_S="0.5", FAKE_AGY_PLUGIN=str(settings.agent_plugin_path),
             FAKE_AGY_LOG=str(tmp_path / "log.jsonl"))
    mgr = ConversationManager(McpBridge(), settings,
                              agents=AgentPluginManager(settings.agent_plugin_path))

    async def one(i):
        req = ChatCompletionRequest(messages=[{"role": "system", "content": f"persona {i}"},
                                              {"role": "user", "content": f"q{i}"}])
        conv = await mgr.start_turn(req, model="m", workdir=settings.default_workdir)
        return [c async for c in mgr.run_turn(conv)]

    loop = asyncio.get_running_loop()
    try:
        t0 = loop.time()
        results = await asyncio.gather(*(one(i) for i in range(10)))
        elapsed = loop.time() - t0
    finally:
        await mgr.close_all()
    assert all(any(isinstance(c, TextChunk) for c in r) for r in results)
    assert elapsed < 3.0, elapsed  # serialized: >= 10 x 0.5 s
    import json as _json
    boots = [_json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
             if '"start"' in line]
    personas = sorted(b["agent_file"].rstrip().rsplit("\n", 1)[-1] for b in boots)
    assert personas == sorted(f"persona {i}" for i in range(10))
    assert list((settings.agent_plugin_path / "agents").iterdir()) == []
