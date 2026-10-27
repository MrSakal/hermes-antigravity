"""One-time migration of the shared Antigravity MCP config: only our legacy
loopback hermes/hermesN slots go; foreign entries and bad files are untouched."""

from __future__ import annotations

import json

from app.mcp_config import BACKUP_SUFFIX, cleanup_legacy_mcp_entries


def test_removes_only_legacy_loopback_slots(tmp_path):
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": {
        "Pipedrive": {"command": "cmd", "args": ["/c", "x"], "env": {"T": "secret"}},
        "hermes": {"serverUrl": "http://127.0.0.1:8788/mcp/conv1-1712345678"},  # ts id
        "hermes3": {"serverUrl": "http://127.0.0.1:9999/mcp/conv7-0f1e2d3c4b5a69788796a5b4c3d2e1f0"},  # uuid id, any port
        "hermes2": {"serverUrl": "https://example.com/mcp/conv1-1712345678"},  # not loopback: keep
        "hermes-tools": {"serverUrl": "http://127.0.0.1:8788/mcp/conv1-1712345678"},  # not our slot form
    }, "other": 1}), encoding="utf-8")
    assert cleanup_legacy_mcp_entries(path) == 2
    data = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(data["mcpServers"]) == ["Pipedrive", "hermes-tools", "hermes2"]
    assert data["mcpServers"]["Pipedrive"]["env"]["T"] == "secret"
    assert data["other"] == 1


def test_users_own_hermes_server_survives(tmp_path):
    """A user's own server that happens to be called hermes/hermesN is kept
    unless host AND path match the ids old builds minted."""
    path = tmp_path / "mcp_config.json"
    keep = {
        "hermes": {"serverUrl": "http://localhost:3000/mcp/"},
        "hermes2": {"serverUrl": "http://127.0.0.1:9000/mcp/"},
        "hermes4": {"serverUrl": "http://127.0.0.1:9000/mcp/tools"},
        "hermes5": {"serverUrl": "http://127.0.0.1:9000/mcp/conv1-abc"},  # too short
        "hermes6": {"serverUrl": "http://localhost:8788/mcp/conv1-1712345678"},  # not 127.0.0.1
    }
    raw = json.dumps({"mcpServers": keep})
    path.write_text(raw, encoding="utf-8")
    assert cleanup_legacy_mcp_entries(path) == 0
    assert path.read_text(encoding="utf-8") == raw
    assert not (tmp_path / ("mcp_config.json" + BACKUP_SUFFIX)).exists()


def test_backup_written_once_before_first_rewrite(tmp_path):
    path = tmp_path / "mcp_config.json"
    backup = tmp_path / ("mcp_config.json" + BACKUP_SUFFIX)
    first = json.dumps({"mcpServers": {
        "hermes": {"serverUrl": "http://127.0.0.1:8788/mcp/conv1-1712345678"}}})
    path.write_text(first, encoding="utf-8")
    assert cleanup_legacy_mcp_entries(path) == 1
    assert backup.read_text(encoding="utf-8") == first
    # A later run never overwrites the original backup.
    path.write_text(json.dumps({"mcpServers": {
        "hermes2": {"serverUrl": "http://127.0.0.1:8788/mcp/conv2-1712345999"}}}),
        encoding="utf-8")
    assert cleanup_legacy_mcp_entries(path) == 1
    assert backup.read_text(encoding="utf-8") == first


def test_nothing_to_do_leaves_file_bytes(tmp_path):
    path = tmp_path / "mcp_config.json"
    raw = '{"mcpServers": {"x": {"command": "y"}}}'
    path.write_text(raw, encoding="utf-8")
    assert cleanup_legacy_mcp_entries(path) == 0
    assert path.read_text(encoding="utf-8") == raw


def test_missing_or_invalid_file_is_harmless(tmp_path):
    assert cleanup_legacy_mcp_entries(tmp_path / "nope.json") == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cleanup_legacy_mcp_entries(bad) == 0
    assert bad.read_text(encoding="utf-8") == "{not json"


_LEGACY = {"mcpServers": {
    "mine": {"serverUrl": "http://localhost:1/x"},
    "hermes": {"serverUrl": "http://127.0.0.1:8788/mcp/conv1-1712345678"},
}}


def test_symlinked_config_stays_a_symlink(tmp_path):
    import os

    import pytest

    target = tmp_path / "dotfiles" / "mcp_config.json"
    target.parent.mkdir()
    target.write_text(json.dumps(_LEGACY), encoding="utf-8")
    link = tmp_path / "mcp_config.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need a privilege on this host")
    assert cleanup_legacy_mcp_entries(link) == 1
    assert link.is_symlink()
    assert sorted(json.loads(target.read_text(encoding="utf-8"))["mcpServers"]) == ["mine"]


def test_concurrent_edit_aborts_the_rewrite(tmp_path, monkeypatch):
    """The IDE rewrote the file between our read and our replace: keep its
    version (we would otherwise drop its change)."""
    import app.mcp_config as mc

    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps(_LEGACY), encoding="utf-8")
    real_mkstemp = mc.tempfile.mkstemp
    ide = json.dumps({"mcpServers": {**_LEGACY["mcpServers"], "new": {"command": "x"}}})

    def mkstemp(*a, **kw):
        path.write_text(ide, encoding="utf-8")  # the IDE's write lands now
        return real_mkstemp(*a, **kw)

    monkeypatch.setattr(mc.tempfile, "mkstemp", mkstemp)
    assert cleanup_legacy_mcp_entries(path) == 0
    assert path.read_text(encoding="utf-8") == ide
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mcp_config.json", "mcp_config.json" + BACKUP_SUFFIX]  # no temp left
