from __future__ import annotations

import json

import pytest

from app.mcp_config import McpConfigManager


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({
        "mcpServers": {
            "Pipedrive": {"command": "cmd", "args": ["/c", "x"], "env": {"T": "secret"}},
        }
    }), encoding="utf-8")
    return path


async def test_add_remove_preserves_foreign_keys(cfg):
    mgr = McpConfigManager(cfg)
    slot = await mgr.add_conversation("c1", "http://127.0.0.1:8788/mcp/c1")
    assert slot == "hermes"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["hermes"] == {"serverUrl": "http://127.0.0.1:8788/mcp/c1"}
    assert data["mcpServers"]["Pipedrive"]["env"]["T"] == "secret"

    await mgr.remove_conversation(slot)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "hermes" not in data["mcpServers"]
    assert "Pipedrive" in data["mcpServers"]


async def test_concurrent_slots(cfg):
    mgr = McpConfigManager(cfg)
    s1 = await mgr.add_conversation("c1", "http://127.0.0.1:8788/mcp/c1")
    s2 = await mgr.add_conversation("c2", "http://127.0.0.1:8788/mcp/c2")
    assert (s1, s2) == ("hermes", "hermes2")
    await mgr.remove_conversation(s1)
    s3 = await mgr.add_conversation("c3", "http://127.0.0.1:8788/mcp/c3")
    assert s3 == "hermes"  # freed slot reused
    await mgr.remove_all()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"]) == ["Pipedrive"]


async def test_cleanup_stale_only_our_port_and_pattern(cfg):
    data = json.loads(cfg.read_text(encoding="utf-8"))
    data["mcpServers"]["hermes"] = {"serverUrl": "http://127.0.0.1:8788/mcp/dead"}
    data["mcpServers"]["hermes7"] = {"serverUrl": "http://127.0.0.1:9999/mcp/other"}
    data["mcpServers"]["hermesx"] = {"serverUrl": "http://127.0.0.1:8788/mcp/notslot"}
    cfg.write_text(json.dumps(data), encoding="utf-8")

    mgr = McpConfigManager(cfg)
    removed = await mgr.cleanup_stale(8788)
    assert removed == 1  # only "hermes" (matching port + slot pattern, unowned)
    left = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
    assert "hermes" not in left
    assert "hermes7" in left and "hermesx" in left and "Pipedrive" in left


async def test_missing_file_created(tmp_path):
    path = tmp_path / "sub" / "mcp_config.json"
    mgr = McpConfigManager(path)
    await mgr.add_conversation("c1", "http://127.0.0.1:8788/mcp/c1")
    assert json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["hermes"]


async def test_invalid_json_refused(tmp_path):
    path = tmp_path / "mcp_config.json"
    path.write_text("{broken", encoding="utf-8")
    mgr = McpConfigManager(path)
    with pytest.raises(json.JSONDecodeError):
        await mgr.add_conversation("c1", "http://x/mcp/c1")
    # File untouched.
    assert path.read_text(encoding="utf-8") == "{broken"
