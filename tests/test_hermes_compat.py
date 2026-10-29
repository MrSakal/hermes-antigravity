"""Contract with hermes-agent's OpenAI-compatible client (custom provider).

hermes decides several behaviours from wire details rather than from the
OpenAI spec: it fingerprints ``/api/v1/models`` as LM Studio and reads limits
from there, classifies 400s by message text (``context length`` → compress and
retry, ``image exceeds`` → shrink images), and reads a mid-stream error's HTTP
status from ``error.status_code``. These tests pin those details.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import PromptTooLarge, Settings
from app.errors import OpenAIError, error_envelope
from app.main import create_app
from app.model_catalog import ModelEntry
from app.model_limits import ModelLimits

ENTRIES = [
    ModelEntry("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
    ModelEntry("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)"),
    ModelEntry("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
    ModelEntry("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"),
    ModelEntry("gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
]
LIMITS = {
    "gemini-3.8-flash-low": ModelLimits(1_048_576, 65_536),
    "gemini-3.8-flash-medium": ModelLimits(1_048_576, 65_536),
    "gemini-3.8-flash-high": ModelLimits(1_000_000, 64_000),  # smaller variant → family min
    "gpt-oss-120b-medium": ModelLimits(131_072, 32_768),
    # gemini-4-argon: backend has no limits → fields omitted, never guessed
}


@pytest.fixture
def client():
    app = create_app()
    cat = app.state.model_catalog
    cat._entries = list(ENTRIES)
    cat._loaded_at = time.monotonic()
    app.state.model_limits._limits = dict(LIMITS)
    return TestClient(app)


def test_v1_models_carries_family_min_limits(client):
    data = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert set(data) == {"gemini-3.8-flash", "gpt-oss-120b", "gemini-4-argon"}
    assert data["gemini-3.8-flash"]["context_length"] == 1_000_000
    assert data["gemini-3.8-flash"]["max_completion_tokens"] == 64_000
    assert data["gpt-oss-120b"]["context_length"] == 131_072
    assert "context_length" not in data["gemini-4-argon"]
    assert "max_completion_tokens" not in data["gemini-4-argon"]


@pytest.mark.parametrize("mid,ctx", [
    ("gemini-3.8-flash-medium", 1_048_576),  # a variant slug: its own limit
    ("Gemini 3.8 Flash (High)", 1_000_000),  # display name
    ("gpt-oss-120b", 131_072),
])
def test_model_detail_limits(client, mid, ctx):
    assert client.get(f"/v1/models/{mid}").json()["context_length"] == ctx


def test_lmstudio_native_shape(client):
    body = client.get("/api/v1/models").json()
    assert [m["id"] for m in body["data"]] == [m["key"] for m in body["models"]]
    lm = {m["key"]: m for m in body["models"]}
    assert lm["gpt-oss-120b"]["loaded_instances"][0]["config"]["context_length"] == 131_072
    assert lm["gpt-oss-120b"]["max_completion_tokens"] == 32_768
    assert lm["gemini-4-argon"]["loaded_instances"] == []  # unknown → no context claim


def test_api_show_context_and_capabilities(client):
    body = client.post("/api/show", json={"model": "gpt-oss-120b"}).json()
    assert body["model_info"]["antigravity.context_length"] == 131_072
    assert {"completion", "tools", "thinking", "vision"} <= set(body["capabilities"])
    body = client.post("/api/show", json={"model": "gemini-4-argon"}).json()
    assert not any(k.endswith("context_length") for k in body["model_info"])


def test_limits_disabled_omits_fields(monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("AGI_MODEL_LIMITS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        app = create_app()
        assert app.state.model_limits is None
        ids = TestClient(app).get("/v1/models").json()["data"]
        assert all("context_length" not in m for m in ids)
    finally:
        monkeypatch.delenv("AGI_MODEL_LIMITS_ENABLED")
        get_settings.cache_clear()


def test_prompt_too_large_message_reads_as_context_overflow():
    from app.routes.chat import check_prompt_size
    with pytest.raises(OpenAIError) as ei:
        check_prompt_size("x" * 11, Settings(max_prompt_bytes=10))
    msg = str(ei.value.message if hasattr(ei.value, "message") else ei.value).lower()
    assert msg.startswith("context length exceeded")
    assert ei.value.status_code == 400


def test_stream_error_envelope_keeps_status():
    assert error_envelope("upstream timed out", status_code=504)["error"]["status_code"] == 504
    assert "status_code" not in error_envelope("x")["error"]  # unchanged when not given
