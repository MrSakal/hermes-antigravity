from __future__ import annotations

import httpx
import pytest

import app.main as main_mod
from app.config import get_settings


@pytest.fixture
def keyed_client(monkeypatch, tmp_path):
    monkeypatch.setenv("AGI_API_KEY", "sekrit")
    monkeypatch.setenv("AGI_DEFAULT_WORKDIR", str(tmp_path / "ws"))
    get_settings.cache_clear()
    application = main_mod.create_app()
    transport = httpx.ASGITransport(app=application)
    yield httpx.AsyncClient(transport=transport, base_url="http://test")
    get_settings.cache_clear()


async def test_v1_requires_bearer(keyed_client):
    r = await keyed_client.get("/v1/models")
    assert r.status_code == 401
    r = await keyed_client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    r = await keyed_client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200


async def test_health_exempt(keyed_client):
    r = await keyed_client.get("/healthz")
    assert r.status_code == 200


def test_non_loopback_without_key_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("AGI_HOST", "0.0.0.0")
    monkeypatch.delenv("AGI_API_KEY", raising=False)
    monkeypatch.setenv("AGI_DEFAULT_WORKDIR", str(tmp_path / "ws"))
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="refusing to start"):
            main_mod.create_app()
    finally:
        get_settings.cache_clear()
