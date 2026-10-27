"""scripts/prune_agy_state.py against a fake ~/.gemini/antigravity-cli tree:
it prunes while agy processes are alive (the persistent sessions and the warm
pool never stop), skipping only what live processes use."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prune_agy_state.py"
_spec = importlib.util.spec_from_file_location("prune_agy_state", _SCRIPT)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)

OLD = time.time() - 30 * 86400


def _touch(p: Path, mtime: float = OLD, data: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    os.utime(p, (mtime, mtime))
    return p


def _conv(root: Path, mtime: float = OLD) -> str:
    cid = str(uuid.uuid4())
    _touch(root / "conversations" / f"{cid}.db", mtime)
    _touch(root / "conversations" / f"{cid}.db-wal", mtime)
    _touch(root / "brain" / cid / "transcript.jsonl", mtime)
    os.utime(root / "brain" / cid, (mtime, mtime))
    _touch(root / "presence" / f"{cid}.lock", mtime, b"")
    return cid


def _run(root: Path, monkeypatch, capsys, live: list[list[str]] | None, *extra: str):
    monkeypatch.setattr(prune, "live_argvs", lambda: None if live is None else (live, 0))
    rc = prune.main(["--root", str(root), "--keep-days", "7", "--json", *extra])
    out = capsys.readouterr().out.strip()
    return rc, (json.loads(out) if out else None)


def test_prunes_while_agy_runs_but_skips_in_use_conversations(tmp_path, monkeypatch, capsys):
    root = tmp_path / "cli"
    old, used, fresh = _conv(root), _conv(root), _conv(root, time.time())
    live = [
        ["/usr/bin/agy", "--print", "", "--input-format", "stream-json", "--conversation", used],
        ["/usr/bin/agy", "--print", "", "--input-format", "stream-json"],  # pooled / fresh
    ]
    rc, stats = _run(root, monkeypatch, capsys, live)
    assert rc == 0
    assert stats["conversations"] == 1 and stats["in_use_skipped"] == 1
    assert not (root / "conversations" / f"{old}.db").exists()
    assert not (root / "brain" / old).exists() and not (root / "presence" / f"{old}.lock").exists()
    for kept in (used, fresh):
        assert (root / "conversations" / f"{kept}.db").exists()
        assert (root / "brain" / kept).is_dir()


def test_prunes_only_old_hermes_mcp_schema_caches(tmp_path, monkeypatch, capsys):
    root = tmp_path / "cli"
    _touch(root / "mcp" / "hermes-0123abcd" / "get_weather.json")
    os.utime(root / "mcp" / "hermes-0123abcd", (OLD, OLD))
    _touch(root / "mcp" / "hermes-89abcdef" / "t.json", time.time())  # recent
    _touch(root / "mcp" / "hermes" / "t.json")  # the user's own server name
    _touch(root / "mcp" / "Pipedrive" / "t.json")
    _touch(root / "mcp" / "hermes-0123abcdX" / "t.json")
    rc, stats = _run(root, monkeypatch, capsys, [])
    assert rc == 0 and stats["mcp_caches"] == 1
    assert sorted(p.name for p in (root / "mcp").iterdir()) == [
        "Pipedrive", "hermes", "hermes-0123abcdX", "hermes-89abcdef"]


def test_refuses_without_process_listing_unless_forced(tmp_path, monkeypatch, capsys):
    root = tmp_path / "cli"
    cid = _conv(root)
    rc, _ = _run(root, monkeypatch, capsys, None)
    assert rc == 2 and (root / "conversations" / f"{cid}.db").exists()
    rc, stats = _run(root, monkeypatch, capsys, None, "--force")
    assert rc == 0 and stats["conversations"] == 1


def test_dry_run_removes_nothing(tmp_path, monkeypatch, capsys):
    root = tmp_path / "cli"
    old = _conv(root)
    rc, stats = _run(root, monkeypatch, capsys, [], "--dry-run")
    assert rc == 0 and stats["conversations"] == 1
    assert (root / "conversations" / f"{old}.db").exists()


def test_conversation_ids_parsing():
    cid = str(uuid.uuid4())
    assert prune.conversation_ids(["agy", "--conversation", cid]) == {cid}
    assert prune.conversation_ids(["agy", f"--conversation={cid.upper()}"]) == {cid}
    assert prune.conversation_ids(["agy", "--conversation", "not-a-uuid"]) == set()
    assert prune.conversation_ids(["agy", "--conversation"]) == set()


def test_live_listing_sees_a_real_process_argv():
    """The platform listing (/proc on Linux, Toolhelp32 + command line on
    Windows) finds ``--conversation <id>`` in a live process' argv."""
    cid = str(uuid.uuid4())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                              "--conversation", cid])
    try:
        found = None
        for _ in range(50):
            found = prune.in_use_conversations()
            assert found is not None
            if cid in found[0]:
                break
            time.sleep(0.1)
        assert cid in found[0]
    finally:
        child.kill()
        child.wait()


def test_deploy_unit_no_longer_masks_exit_2():
    unit = (_SCRIPT.parents[1] / "deploy" / "agy-prune.service").read_text()
    assert "SuccessExitStatus=2" not in unit
