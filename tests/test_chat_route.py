"""End-to-end route tests over ASGI with the fake agy subprocess."""

from __future__ import annotations

import json

import httpx
import pytest

import app.main as main_mod
from app.config import get_settings


@pytest.fixture
def client(fake_agy, monkeypatch, tmp_path):
    monkeypatch.setenv("AGI_DEFAULT_WORKDIR", str(tmp_path / "ws"))
    get_settings.cache_clear()
    application = main_mod.create_app()
    transport = httpx.ASGITransport(app=application)
    yield httpx.AsyncClient(transport=transport, base_url="http://test"), application
    get_settings.cache_clear()


async def test_nonstream_simple(client, fake_agy):
    fake_agy("simple")
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "model": "gemini-3.5-flash",
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
        "model": "gemini-3.5-flash",
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
    # A hit stores the extended transcript as a new entry.
    assert len(application.state.conv_cache) == 2


async def test_tools_rejected_phase1(client):
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    })
    assert r.status_code == 400
    assert "Phase 2" in r.json()["error"]["message"]


async def test_empty_messages_rejected(client):
    http, _ = client
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "   "}],
    })
    assert r.status_code == 400


async def test_prompt_length_guard(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(get_settings(), "max_prompt_chars", 10)
    r = await http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x" * 50}],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


async def test_models_endpoint(client):
    http, _ = client
    r = await http.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert "gemini-3.5-flash" in ids
    assert "Gemini 3.5 Flash (Medium)" in ids


async def test_healthz(client):
    http, _ = client
    r = await http.get("/healthz")
    assert r.json()["status"] == "ok"
