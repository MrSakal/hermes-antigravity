"""End-to-end route tests over ASGI with the fake agy subprocess."""

from __future__ import annotations

import json

import httpx
import pytest

import app.main as main_mod
from app.config import get_settings


@pytest.fixture
def client(fake_agy, fake_agy_bin, monkeypatch, tmp_path):
    monkeypatch.setenv("AGI_DEFAULT_WORKDIR", str(tmp_path / "ws"))
    # Never touch the user's real shared Antigravity MCP config / plugin dir
    # from tests.
    monkeypatch.setenv("AGI_MCP_CONFIG_PATH", str(tmp_path / "mcp_config.json"))
    monkeypatch.setenv("AGI_AGENT_PLUGIN_PATH", str(tmp_path / "agent-plugin"))
    monkeypatch.setenv("AGI_AGY_BRAIN_PATH", str(tmp_path / "brain"))
    # `agy models` (model catalog) goes to the fake too; the AgySession spawn
    # path is rewritten by the fake_agy fixture regardless of agy_bin.
    monkeypatch.setenv("AGI_AGY_BIN", fake_agy_bin)
    monkeypatch.setenv("AGI_MODEL_CATALOG_REFRESH_S", "0")
    get_settings.cache_clear()
    application = main_mod.create_app()
    transport = httpx.ASGITransport(app=application)
    yield httpx.AsyncClient(transport=transport, base_url="http://test"), application
    get_settings.cache_clear()


async def test_nonstream_simple(client, fake_agy):
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "model": "gemini-3.7-flash",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 100
    assert body["usage"]["completion_tokens"] == 20
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5


async def test_stream_simple(client, fake_agy):
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "model": "gemini-3.7-flash",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    chunks = [l for l in r.text.split("\n\n") if l.startswith("data: ")]
    assert chunks[-1] == "data: [DONE]"
    payloads = [json.loads(c[6:]) for c in chunks[:-1]]
    text = "".join(
        p["choices"][0]["delta"].get("content") or "" for p in payloads
    )
    assert text == "Hello world"
    finish = [p for p in payloads if p["choices"][0]["finish_reason"]]
    assert finish and finish[-1]["choices"][0]["finish_reason"] == "stop"
    assert finish[-1]["usage"]["prompt_tokens"] == 100


async def test_tool_boundary_inserts_seam(client, fake_agy):
    fake_agy("tool")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    # SegmentJoiner guarantees a blank line where the builtin tool ran.
    assert content == "Hello \n\nworld"


async def test_upstream_error_maps_to_502(client, fake_agy):
    fake_agy("error")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 502
    assert "not recognized" in r.json()["error"]["message"]


async def test_noise_surfaces_diagnostics(client, fake_agy):
    fake_agy("noise")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 502
    assert "no output produced" in r.json()["error"]["message"]


async def test_conv_cache_roundtrip(client, fake_agy):
    fake_agy("simple")
    http, application = client
    first = [{"role": "user", "content": "q1"}]
    r = await http.post("/v1/chat/completions", json={"messages": first})
    assert r.status_code == 200
    assert len(application.state.conv_cache) == 1

    # Follow-up: previous transcript + assistant reply + new user message.
    nxt = first + [
        {"role": "assistant", "content": "Hello world"},
        {"role": "user", "content": "q2"},
    ]
    r2 = await http.post("/v1/chat/completions", json={"messages": nxt})
    assert r2.status_code == 200
    # A claim consumes the old mutable prefix; only its replacement remains.
    assert len(application.state.conv_cache) == 1


async def test_tools_request_without_tool_use_completes(client, fake_agy, tmp_path):
    """A tools-bearing request whose model never calls a tool ends as a normal
    completion — and the per-conversation MCP config entry is added/removed."""
    fake_agy("simple")
    http, application = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"]
    # Conversation closed → our slot entry removed again.
    cfg_path = tmp_path / "mcp_config.json"
    if cfg_path.exists():
        assert "hermes" not in json.loads(cfg_path.read_text())["mcpServers"]


async def test_system_message_materializes_agent(client, fake_agy, tmp_path):
    """A system message becomes a generated agent definition in the managed
    plugin (delivered to agy via --agent) instead of a prompt preamble."""
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "hi"},
        ],
    })
    assert r.status_code == 200
    agents = list((tmp_path / "agent-plugin" / "agents").glob("hermes-*.md"))
    assert len(agents) == 1
    assert "You are a pirate." in agents[0].read_text()


async def test_agent_mode_disabled_writes_nothing(client, fake_agy, tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "system_prompt_via_agent", False)
    monkeypatch.setattr(get_settings(), "plain_chat_tools", "all")
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "hi"},
        ],
    })
    assert r.status_code == 200
    assert not (tmp_path / "agent-plugin").exists()


async def test_image_attachment_written_to_workdir(client, fake_agy, tmp_path):
    fake_agy("simple")
    http, _ = client
    png = ("data:image/png;base64,"
           "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
           "/pLvAAAAAElFTkSuQmCC")
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what color is this?"},
            {"type": "image_url", "image_url": {"url": png}},
        ]}],
    })
    assert r.status_code == 200
    files = list((tmp_path / "ws" / "attachments").glob("img-*.png"))
    assert len(files) == 1


def _write_thinking_transcript(tmp_path, conv="conv-fake-1", thinking="pondering deeply"):
    tf = (tmp_path / "brain" / conv / ".system_generated" / "logs"
          / "transcript_full.jsonl")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tf.write_text(
        json.dumps({"type": "PLANNER_RESPONSE", "content": "x", "thinking": thinking}) + "\n",
        encoding="utf-8",
    )


async def test_reasoning_content_nonstream(client, fake_agy, tmp_path):
    """Thinking from the brain transcript surfaces as message.reasoning_content."""
    _write_thinking_transcript(tmp_path)
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "Hello world"
    assert msg["reasoning_content"] == "pondering deeply"


async def test_reasoning_content_stream(client, fake_agy, tmp_path):
    _write_thinking_transcript(tmp_path)
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    payloads = [json.loads(c[6:]) for c in r.text.split("\n\n")
                if c.startswith("data: ") and c != "data: [DONE]"]
    reasoning = "".join(
        p["choices"][0]["delta"].get("reasoning_content") or "" for p in payloads
    )
    content = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads)
    assert reasoning == "pondering deeply"
    assert content == "Hello world"


async def test_no_transcript_means_no_reasoning(client, fake_agy):
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "reasoning_content" not in r.json()["choices"][0]["message"]


async def test_empty_messages_rejected(client):
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "   "}],
    })
    assert r.status_code == 400


async def test_prompt_size_guard_is_bytes(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(get_settings(), "max_prompt_bytes", 10)
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ő" * 6}],  # 6 chars, 12 bytes
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


async def test_prompt_size_guard_on_tool_path(client, monkeypatch, tmp_path):
    """The tool-passthrough path must reject oversized prompts with a proper
    400 (it used to spawn anyway and die with OSError E2BIG → raw 500)."""
    http, _ = client
    monkeypatch.setattr(get_settings(), "max_prompt_bytes", 10)
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x" * 50}],
        "tools": [{"type": "function", "function": {"name": "noop"}}],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"
    # No MCP slot leaked.
    cfg_path = tmp_path / "mcp_config.json"
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text())["mcpServers"] == {}


async def test_large_prompt_reaches_agy_over_stdin(client, fake_agy):
    fake_agy("echoprompt")
    http, _ = client
    big = "x" * 400_000 + "THE_END!"  # far beyond any argv limit
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": big}],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == f"len={len(big)};tail=THE_END!"


async def test_unknown_model_is_400_model_not_found(client, fake_agy):
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_found"
    assert r.json()["error"]["param"] == "model"


async def test_live_catalog_model_resolves(client, fake_agy):
    """A model only the fake `agy models` knows (9.9) resolves through the
    catalog loaded at startup — no static table entry needed."""
    fake_agy("echoargs")
    http, application = client
    async with application.router.lifespan_context(application):
        assert application.state.model_catalog.loaded
        r = await http.post("/v1/chat/completions", json={
            "model": "gemini-9.9-flash-high",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 200
        assert r.json()["model"] == "Gemini 9.9 Flash (High)"


async def test_plain_chat_agent_is_readonly_by_default(client, fake_agy, tmp_path):
    """Requests without `tools` are answered by an agent whose frontmatter
    allowlists view_file only — no shell for untrusted content."""
    fake_agy("echoargs")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "summarize this web page: ..."}],
    })
    assert r.status_code == 200
    agents = list((tmp_path / "agent-plugin" / "agents").glob("hermes-*.md"))
    assert len(agents) == 1
    assert 'tools: ["view_file"]' in agents[0].read_text()


async def test_tool_turn_agent_keeps_full_tools(client, fake_agy, tmp_path):
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "hi"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    })
    assert r.status_code == 200
    agents = list((tmp_path / "agent-plugin" / "agents").glob("hermes-*.md"))
    assert len(agents) == 1
    assert 'tools: "*"' in agents[0].read_text()


async def test_nonstream_client_disconnect_kills_subprocess(client, fake_agy, monkeypatch):
    """A non-streaming request whose client goes away must not leave the agy
    subprocess running to request_timeout_s (observed live: 5+ min orphans)."""
    import asyncio

    from app.routes import chat as chat_mod

    fake_agy("hang", FAKE_AGY_HANG_S="30")
    http, _ = client
    monkeypatch.setattr(get_settings(), "disconnect_poll_s", 0.05)

    created: list = []
    orig_init = chat_mod.AgySession.__init__

    def spy_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(chat_mod.AgySession, "__init__", spy_init)

    # Make the request look disconnected right after the first poll.
    from fastapi import Request

    async def always_gone(self):
        return bool(created)  # disconnect AFTER admission/start, not while queued

    monkeypatch.setattr(Request, "is_disconnected", always_gone)

    t0 = asyncio.get_running_loop().time()
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    elapsed = asyncio.get_running_loop().time() - t0
    assert r.status_code == 499
    assert elapsed < 10, "request should end on disconnect, not on the fake's 30 s hang"
    assert created and not created[0].running


async def test_models_endpoint_static_fallback(client):
    """Before the lifespan loads the catalog, the static table is served."""
    http, _ = client
    r = await http.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert "gemini-3.8-flash" not in ids  # family remains an input alias only
    assert "gemini-3.8-flash-high" in ids
    assert "Gemini 3.8 Flash (Medium)" not in ids
    assert "gemini-3.8-flash-medium" in ids
    assert "claude-opus-4-6-thinking" in ids  # agy's own slug spelling


async def test_models_endpoint_live_catalog(client):
    http, application = client
    async with application.router.lifespan_context(application):
        r = await http.get("/v1/models")
        ids = [m["id"] for m in r.json()["data"]]
        assert "gemini-9.9-flash" not in ids  # bare family is an input alias
        assert "gemini-9.9-flash-medium" in ids
        assert "Gemini 9.9 Flash (Low)" not in ids
        assert "gemini-9.9-flash-low" in ids
        assert "gemini-3.8-flash" not in ids  # static table not mixed in
        r = await http.get("/v1/models/gemini-9.9-flash")
        assert r.status_code == 200
        r = await http.get("/v1/models/nope")
        assert r.status_code == 404


async def test_healthz(client):
    http, _ = client
    r = await http.get("/healthz")
    assert r.json()["status"] == "ok"

@pytest.mark.parametrize("stream,tools", [(False, False), (True, False), (False, True), (True, True)])
async def test_real_asgi_disconnect(client, fake_agy, monkeypatch, stream, tools):
    """Exercise the actual ASGI receive chain, not a mocked is_disconnected."""
    import asyncio
    from app.agy_session import AgySession
    fake_agy("hang", FAKE_AGY_HANG_S="30")
    _, application = client
    monkeypatch.setattr(get_settings(), "disconnect_poll_s", 0.02)
    sessions = []
    original = AgySession.start
    async def start(self):
        await original(self)
        sessions.append(self)
    monkeypatch.setattr(AgySession, "start", start)
    payload = {"messages": [{"role": "user", "content": "hello"}], "stream": stream}
    if tools:
        payload["tools"] = [{"type": "function", "function": {"name": "noop"}}]
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False})
    async def disconnect():
        while not sessions:
            await asyncio.sleep(0.01)
        incoming.put_nowait({"type": "http.disconnect"})
    sent = []
    async def send(event):
        sent.append(event)
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "query_string": b"", "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 10000), "server": ("127.0.0.1", 8788)}
    watcher = asyncio.create_task(disconnect())
    try:
        await asyncio.wait_for(application(scope, incoming.get, send), timeout=8)
        assert sessions and all(not s.running for s in sessions)
        assert not application.state.conv_manager._conversations
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        for session in sessions:
            await session.aclose()


async def test_readonly_agent_failure_fails_closed(client, monkeypatch):
    from unittest.mock import AsyncMock
    http, application = client
    monkeypatch.setattr(application.state.agent_plugin, "ensure_agent", AsyncMock(return_value=None))
    response = await http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "agent_unavailable"


async def test_readonly_survives_system_agent_disabled(client, fake_agy, monkeypatch, tmp_path):
    fake_agy("simple")
    monkeypatch.setattr(get_settings(), "system_prompt_via_agent", False)
    http, _ = client
    response = await http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert any('tools: ["view_file"]' in p.read_text() for p in (tmp_path / "agent-plugin/agents").glob("*.md"))
