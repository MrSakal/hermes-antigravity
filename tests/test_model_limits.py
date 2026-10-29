from __future__ import annotations

import json

import httpx
import pytest

from app.model_catalog import ModelEntry
from app.model_limits import ModelLimits, ModelLimitsProvider, parse_available_models, resolve_limits
from app.openai_models import ChatCompletionRequest

BACKEND = {
    "models": {
        "gemini-3.8-flash-tiered": {"maxTokens": 1048576, "maxOutputTokens": 65536},
        "gemini-3.6-flash-high": {"displayName": "Gemini 3.6 Flash (High)", "maxTokens": 1048576, "maxOutputTokens": 65536},
        "claude-sonnet-4-6": {"displayName": "Claude Sonnet 4.6 (Thinking)", "maxTokens": 250000, "maxOutputTokens": 64000},
        "gemini-pro-agent": {"displayName": "Gemini 3.1 Pro (High)", "maxTokens": 1048576, "maxOutputTokens": 65535},
        "bogus": {"displayName": "Bogus", "maxTokens": "lots"},
    }
}
ENTRIES = [
    ModelEntry("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
    ModelEntry("gemini-3.6-flash-high", "Gemini 3.6 Flash (High)"),
    ModelEntry("claude-sonnet-4-6", "Claude Sonnet 4.6 (Thinking)"),
    ModelEntry("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)"),
    ModelEntry("unknown-model", "Unknown"),
]


def test_resolution_exact_display_and_tiered():
    backend = parse_available_models(BACKEND)
    got = {e.slug: resolve_limits(e, backend) for e in ENTRIES}
    assert got["gemini-3.6-flash-high"] == ModelLimits(1048576, 65536)  # exact id
    assert got["claude-sonnet-4-6"].context_length == 250000
    assert got["gemini-3.1-pro-high"].max_output_tokens == 65535  # via displayName
    assert got["gemini-3.8-flash-high"].context_length == 1048576  # via <family>-tiered
    assert got["unknown-model"] is None  # never invented


def test_non_int_limits_are_dropped():
    assert resolve_limits(ModelEntry("bogus", "Bogus"), parse_available_models(BACKEND)) is None


def test_parse_rejects_error_payload():
    with pytest.raises(ValueError):
        parse_available_models({"error": {"code": 403}})


def _provider(tmp_path, handler):
    tok = tmp_path / "tok"
    tok.write_text(json.dumps({"token": {"access_token": "ya29.test"}}))
    proj = tmp_path / "proj"
    proj.write_text("proj-1\n")
    return ModelLimitsProvider(endpoint="https://example.invalid/x", token_path=tok, project_path=proj,
                               transport=httpx.MockTransport(handler))


async def test_refresh_success_sends_token_and_project(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=BACKEND)

    p = _provider(tmp_path, handler)
    assert await p.refresh(ENTRIES) is True
    assert seen == {"auth": "Bearer ya29.test", "body": {"project": "proj-1"}}
    assert p.get("claude-sonnet-4-6") == ModelLimits(250000, 64000)
    assert p.get("unknown-model") is None


async def test_refresh_failure_keeps_previous(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=BACKEND) if calls["n"] == 1 else httpx.Response(403, json={})

    p = _provider(tmp_path, handler)
    assert await p.refresh(ENTRIES) is True
    assert await p.refresh(ENTRIES, force=True) is False
    assert p.get("claude-sonnet-4-6") is not None and "403" in p.last_error


async def test_refetch_only_when_catalog_changes(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=BACKEND)

    p = _provider(tmp_path, handler)
    assert await p.refresh(ENTRIES) is True
    assert await p.refresh(list(reversed(ENTRIES))) is False  # same slug set → no RPC
    assert calls["n"] == 1
    assert await p.refresh(ENTRIES[:2]) is True  # model removed → refetch
    assert calls["n"] == 2 and p.get("claude-sonnet-4-6") is None


async def test_missing_token_is_soft(tmp_path):
    p = ModelLimitsProvider(endpoint="https://example.invalid/x", token_path=tmp_path / "nope",
                            project_path=tmp_path / "nope2")
    assert await p.refresh(ENTRIES) is False and not p.loaded


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high"])
def test_reasoning_effort_accepts_openai_values(effort):
    assert ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}],
                                 reasoning_effort=effort).reasoning_effort == effort


def test_model_card_omits_unknown_limits():
    from app.openai_models import ModelCard
    assert "context_length" not in ModelCard(id="x").model_dump(exclude_none=True)
