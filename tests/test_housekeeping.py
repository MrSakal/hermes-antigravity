"""Zero-footprint housekeeping (app/housekeeping.py, app/purge.py) against the
fake agy, which leaves agy 1.4.8's per-conversation footprint in a fake
``~/.gemini/antigravity-cli`` (FAKE_AGY_STATE_ROOT) and honors --log-file.

The invariant every test re-checks: a conversation that is NOT in the ledger
(the user's own agy / IDE data) is never touched."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

import app.main as main_mod
from app import housekeeping as hk_mod
from app.config import Settings, get_settings
from app.housekeeping import (
    Housekeeper,
    InstanceLock,
    Ledger,
    conversation_exists,
    delete_conversation_state,
    process_start_time,
    scrub_shared_caches,
)
from app.purge import purge

from conftest import pid_alive

OLD = time.time() - 30 * 86400


def _make_conv(root: Path, cid: str, cwd: str = "C:/user/project") -> None:
    """A conversation as agy leaves it (also used for the USER's own data)."""
    for p in (root / "conversations" / f"{cid}.db", root / "conversations" / f"{cid}.db-wal",
              root / "presence" / f"{cid}.lock", root / "annotations" / f"{cid}.pbtxt",
              root / "implicit" / f"{cid}.pb", root / "brain" / cid / "transcript.jsonl"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("user data")
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    last = cache / "last_conversations.json"
    data = json.loads(last.read_text()) if last.exists() else {}
    data[cwd] = cid
    last.write_text(json.dumps(data, indent=2))
    meta = cache / "conversation_metadata.json"
    data = json.loads(meta.read_text()) if meta.exists() else {"conversations": {}}
    data["conversations"][cid] = {"is_internal": False, "summary": {"Title": "mine"}}
    meta.write_text(json.dumps(data, indent=2))


def _cache_ids(root: Path) -> tuple[set[str], set[str]]:
    cache = root / "cache"
    last = json.loads((cache / "last_conversations.json").read_text()) \
        if (cache / "last_conversations.json").exists() else {}
    meta = json.loads((cache / "conversation_metadata.json").read_text()) \
        if (cache / "conversation_metadata.json").exists() else {}
    return set(last.values()), set(meta.get("conversations", {}))


def _all_uuids(root: Path) -> set[str]:
    d = root / "conversations"
    return {p.name.split(".")[0] for p in d.iterdir()} if d.is_dir() else set()


@pytest.fixture
def env(fake_agy, fake_agy_bin, monkeypatch, tmp_path):
    """Fake agy with a fake acli root; a USER conversation pre-exists there."""
    root = Path(os.environ["AGI_AGY_STATE_ROOT"])
    state = Path(os.environ["AGI_STATE_DIR"])
    user_id = str(uuid.uuid4())
    _make_conv(root, user_id)
    # A crash marker of somebody else's agy process (never ours to delete).
    (root / "crashes").mkdir(parents=True, exist_ok=True)
    (root / "crashes" / f"crash_4_{uuid.uuid4()}.log").write_text("")
    monkeypatch.setenv("AGI_DEFAULT_WORKDIR", str(tmp_path / "ws"))
    monkeypatch.setenv("AGI_MCP_CONFIG_PATH", str(tmp_path / "config" / "mcp_config.json"))
    monkeypatch.setenv("AGI_AGENT_PLUGIN_PATH", str(tmp_path / "plugins" / "hermes-bridge"))
    monkeypatch.setenv("AGI_AGY_BIN", fake_agy_bin)
    monkeypatch.setenv("AGI_MODEL_CATALOG_REFRESH_S", "0")
    monkeypatch.setenv("FAKE_AGY_STATE_ROOT", str(root))
    monkeypatch.setenv("FAKE_AGY_CONV_ID", "uuid")
    monkeypatch.setenv("FAKE_AGY_LOG", str(tmp_path / "fake.log"))
    fake_agy("simple")
    get_settings.cache_clear()
    yield type("Env", (), {"root": root, "state": state, "user_id": user_id, "tmp": tmp_path})
    get_settings.cache_clear()


def _assert_user_intact(env) -> None:
    files, brain = hk_mod.conversation_files(env.root, env.user_id)
    for p in files[:2] + files[3:]:  # (no -shm in the fixture)
        assert p.read_text() == "user data", p
    assert (brain / "transcript.jsonl").read_text() == "user data"
    last, meta = _cache_ids(env.root)
    assert env.user_id in last and env.user_id in meta
    assert [p.name[:8] for p in (env.root / "crashes").iterdir()] == ["crash_4_"]


@contextlib.asynccontextmanager
async def running_app():
    application = main_mod.create_app()
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http, application


async def _chat(http, text="hi", history=None):
    msgs = list(history or []) + [{"role": "user", "content": text}]
    # The fake `agy models` catalog lists Gemini 9.9 Flash.
    r = await http.post("/v1/chat/completions", json={"model": "gemini-9.9-flash",
                                                      "messages": msgs})
    assert r.status_code == 200, r.text
    return msgs + [{"role": "assistant", "content": r.json()["choices"][0]["message"]["content"]}]


async def _settle(application, timeout=10.0):
    hk = application.state.housekeeper
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        await asyncio.sleep(0.3)
        await hk.reconcile()
        await hk.flush()
        if not hk.owned or all(hk.live.get(u) or u in hk._referenced() for u in hk.owned):
            return


def _started(tmp_path) -> list[dict]:
    p = tmp_path / "fake.log"
    if not p.exists():
        return []
    return [r for r in map(json.loads, p.read_text().splitlines()) if r.get("start")]


# ── B: per-process logs ─────────────────────────────────────────────────────


async def test_spawn_logs_to_state_dir_and_log_is_deleted(env, monkeypatch):
    monkeypatch.setenv("AGI_SESSION_REUSE", "false")
    async with running_app() as (http, application):
        await _chat(http)
        await _settle(application)
        starts = _started(env.tmp)
        assert starts and all(s["log_file"] and Path(s["log_file"]).parent == env.state / "logs"
                              for s in starts)
        assert not (env.root / "log").exists()  # agy's default log dir untouched
        assert not any((env.state / "logs").iterdir())  # deleted at process exit
    _assert_user_intact(env)


async def test_models_refresh_log_is_deleted(env, fake_agy_bin):
    s = Settings()
    hk = Housekeeper(s)
    await hk.start()
    try:
        from app.model_catalog import ModelCatalog

        cat = ModelCatalog(fake_agy_bin)
        cat.housekeeper = hk
        assert await cat.refresh()
        await hk.flush()
        assert list((env.state / "logs").iterdir()) == []
        assert not (env.root / "log").exists()
        _assert_user_intact(env)  # our crash marker went, the foreign one stayed
    finally:
        await hk.stop()


async def test_failed_process_log_kept_only_when_asked(env, monkeypatch, fake_agy):
    monkeypatch.setenv("AGI_KEEP_AGY_LOGS", "true")
    fake_agy("bootfail")
    async with running_app() as (http, application):
        r = await http.post("/v1/chat/completions",
                            json={"model": "gemini-9.9-flash",
                                  "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code >= 500
        await application.state.housekeeper.flush()
        kept = list((env.state / "logs").iterdir())
        assert len(kept) == 1 and kept[0].read_text().startswith("boot")
    fake_agy("simple")
    monkeypatch.setenv("AGI_KEEP_AGY_LOGS", "false")
    get_settings.cache_clear()
    async with running_app() as (http, application):
        await _chat(http)
    # Default: a failed process' log goes too (only the kept one is left,
    # swept after 7 days).
    assert len(list((env.state / "logs").iterdir())) == 1


# ── A + C: ledger and prompt deletion ───────────────────────────────────────


async def test_closed_conversation_is_deleted_promptly(env, monkeypatch):
    monkeypatch.setenv("AGI_SESSION_REUSE", "false")
    async with running_app() as (http, application):
        before = _all_uuids(env.root)
        await _chat(http)
        ours = _all_uuids(env.root) - before
        await _settle(application)
        assert ours and all(not conversation_exists(env.root, u) for u in ours)
        last, meta = _cache_ids(env.root)
        assert not (ours & last) and not (ours & meta)
        snap = await application.state.housekeeper.ledger_snapshot()
        assert snap["convs"] == []
        _assert_user_intact(env)
    _assert_user_intact(env)


async def _gone_without_sweep(hk, root: Path, uuids: set[str], timeout: float = 5.0) -> bool:
    """Poll only — no manual reconcile / sweep / GC: the deletion must be
    driven by the process exit alone (files gone AND the ledger row dropped)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if all(not conversation_exists(root, u) and u not in hk.owned for u in uuids):
            return True
        await asyncio.sleep(0.05)
    return False


async def test_exited_process_state_deleted_at_once(env):
    """No resume of an exited process: once a parked process exits (evicted,
    idle TTL), its agy state goes right away, and a follow-up starts fresh on
    a NEW conversation (no --conversation) with the whole history folded."""
    async with running_app() as (http, application):
        mgr = application.state.conv_manager
        hk = application.state.housekeeper
        before = _all_uuids(env.root)
        hist = await _chat(http)
        (ours,) = _all_uuids(env.root) - before
        await mgr.evict_idle()  # the IDLE process exits
        assert await _gone_without_sweep(hk, env.root, {ours}), "exited process' state kept"
        await _chat(http, "again", hist)
        (second,) = _all_uuids(env.root) - before
        assert second != ours
        starts = _started(env.tmp)
        assert len(starts) == 2 and not any("--conversation" in r["argv"] for r in starts)
        # An idle-TTL expiry deletes just the same.
        for conv in mgr._conversations.values():
            conv.last_activity -= 10 * 86400
        assert await mgr.gc_once() == 1
        assert await _gone_without_sweep(hk, env.root, {second}), "TTL-expired process' state kept"
    _assert_user_intact(env)


async def test_shutdown_deletes_everything_ours(env):
    async with running_app() as (http, application):
        before = _all_uuids(env.root)
        await _chat(http)
        await application.state.conv_manager.evict_idle()  # gone before shutdown
        ours = _all_uuids(env.root) - before
        assert ours
    assert _all_uuids(env.root) == before == {env.user_id}
    led = Ledger(env.state / hk_mod.LEDGER_NAME).open()
    try:
        assert led.convs() == [] and led.procs() == []
    finally:
        led.close()
    assert list((env.state / "run").iterdir()) == []  # state lock released
    assert list((env.state / "logs").iterdir()) == []
    _assert_user_intact(env)


async def test_unadopted_pool_boot_deleted_on_discard(env, monkeypatch):
    monkeypatch.setenv("AGI_WARM_POOL_SIZE", "1")
    monkeypatch.setenv("AGI_WARM_POOL_MAX_IDLE_S", "1")
    monkeypatch.setenv("AGI_SESSION_REUSE", "false")
    async with running_app() as (http, application):
        pool = application.state.warm_pool
        await _chat(http)  # sets the pool target
        await asyncio.sleep(0.5)
        for _ in range(60):  # a boot, then (idle > 1 s) a recycle
            await asyncio.sleep(0.25)
            await pool.gc()
            if pool.spawn_failures == 0 and len(_started(env.tmp)) >= 3:
                break
        assert len(_started(env.tmp)) >= 3, "pool never recycled"
        await asyncio.sleep(0.5)
        await _settle(application)
        live = {u for u, n in application.state.housekeeper.live.items() if n}
        # Only the currently pooled (live) boot may still exist.
        assert _all_uuids(env.root) - {env.user_id} <= live
        assert len(live) <= 1
    assert _all_uuids(env.root) == {env.user_id}
    _assert_user_intact(env)


# ── D: crash recovery and sweep ─────────────────────────────────────────────


def _spawn_sleeper() -> subprocess.Popen:
    kw = {}
    if os.name == "posix":
        kw["start_new_session"] = True
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **kw)


async def test_crash_recovery_kills_ours_and_deletes_ledger_state(env):
    s = Settings()
    crashed = str(uuid.uuid4())
    _make_conv(env.root, crashed, cwd="C:/agi-workspace")
    sleeper = _spawn_sleeper()
    stranger = _spawn_sleeper()
    try:
        for _ in range(50):
            if process_start_time(sleeper.pid) and process_start_time(stranger.pid):
                break
            time.sleep(0.05)
        (env.state / "logs").mkdir(parents=True, exist_ok=True)
        log = env.state / "logs" / "spawn-0123456789ab.log"
        log.write_text("boot")
        led = Ledger(env.state / hk_mod.LEDGER_NAME).open()
        led.add_conv(crashed, "dead-1", str(env.root))
        led.add_proc("spawn-0123456789ab", "dead-1", sleeper.pid,
                     process_start_time(sleeper.pid), str(log))
        # Same pid number, different start time: a recycled pid — never killed.
        led.add_proc("spawn-ba9876543210", "dead-1", stranger.pid, "1", None)
        led.close()
        hk = Housekeeper(s)
        await hk.start()
        try:
            assert hk.stats["killed"] == 1
            sleeper.wait(timeout=10)
            assert pid_alive(stranger.pid)
            assert not conversation_exists(env.root, crashed)
            assert crashed not in _cache_ids(env.root)[0] | _cache_ids(env.root)[1]
            assert not log.exists()
            assert (await hk.ledger_snapshot())["convs"] == []
        finally:
            await hk.stop()
    finally:
        for p in (sleeper, stranger):
            with contextlib.suppress(Exception):
                p.kill()
                p.wait(timeout=10)
    _assert_user_intact(env)


async def test_live_instance_rows_are_never_touched(env):
    """A second server on the same state dir: its conversations stay."""
    other = str(uuid.uuid4())
    _make_conv(env.root, other, cwd="C:/other")
    lock = InstanceLock(env.state / "run", "other-1")
    lock.acquire()
    try:
        led = Ledger(env.state / hk_mod.LEDGER_NAME).open()
        led.add_conv(other, "other-1", str(env.root))
        led.close()
        hk = Housekeeper(Settings())
        await hk.start()
        await hk.stop()
        assert conversation_exists(env.root, other)
        code, report = purge(Settings())
        assert code == 2 and "other-1" in report["refused"]
        assert conversation_exists(env.root, other)
    finally:
        lock.release()
    _assert_user_intact(env)


def test_never_touches_non_ledger_or_bogus_ids(env):
    with pytest.raises(ValueError):
        delete_conversation_state(env.root, "../conversations")
    with pytest.raises(ValueError):
        delete_conversation_state(env.root, env.user_id.upper())
    led = Ledger(env.state / hk_mod.LEDGER_NAME).open()
    led.add_conv("../../etc", "dead-2", str(env.root))
    led.close()
    code, report = purge(Settings())
    assert code == 0, report
    _assert_user_intact(env)


async def test_permission_error_left_for_next_sweep(env, monkeypatch):
    hk = Housekeeper(Settings())
    await hk.start()
    try:
        ours = str(uuid.uuid4())
        _make_conv(env.root, ours, cwd="C:/agi-workspace")

        class S:  # a session that announced `ours` and has exited
            workdir = str(env.tmp)
        sess = S()
        hk.log_file_for(sess)
        hk.on_init(sess, ours)
        real_unlink = Path.unlink

        def busy(self, *a, **k):
            if self.name.startswith(ours) and self.suffix == ".db":
                raise PermissionError("held")
            return real_unlink(self, *a, **k)
        monkeypatch.setattr(Path, "unlink", busy)
        monkeypatch.setattr(hk_mod, "_DELETE_BACKOFF_S", 0.0)
        hk.on_exit(sess)
        await hk.reconcile()
        assert ours in hk.owned and conversation_exists(env.root, ours)
        monkeypatch.setattr(Path, "unlink", real_unlink)
        await hk.sweep()
        assert ours not in hk.owned and not conversation_exists(env.root, ours)
    finally:
        await hk.stop()
    _assert_user_intact(env)


async def test_sweep_removes_our_cruft_only(env):
    s = Settings()
    plugin = Path(os.environ["AGI_AGENT_PLUGIN_PATH"])
    (plugin / "agents").mkdir(parents=True)
    tmp_old = plugin / "agents" / ".agi-x.tmp"
    tmp_new = plugin / "agents" / ".agi-y.tmp"
    for p in (tmp_old, tmp_new):
        p.write_text("x")
    os.utime(tmp_old, (OLD, OLD))
    mcp = env.root / "mcp"
    stale, fresh, legacy, users = (mcp / "hermes-0badf00d", mcp / "hermes-12345678",
                                   mcp / "hermes", mcp / "github")
    for d in (stale, fresh, legacy, users):
        (d / "tools.json").parent.mkdir(parents=True)
        (d / "tools.json").write_text("{}")
    for d in (stale, legacy, users):
        os.utime(d / "tools.json", (OLD, OLD))
        os.utime(d, (OLD, OLD))
    ws = Path(os.environ["AGI_DEFAULT_WORKDIR"])
    (ws / "attachments").mkdir(parents=True)
    crash_tmp = ws / "attachments" / ".agi-abc"
    crash_tmp.write_text("x")
    os.utime(crash_tmp, (OLD, OLD))
    cfg = Path(os.environ["AGI_MCP_CONFIG_PATH"])
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"mcpServers": {"hermes": {"command": "mine"}}}))
    (cfg.parent / "mcp_config.json.hermes.lock").write_text("")
    bak = cfg.parent / "mcp_config.json.hermes-antigravity.bak"
    bak.write_text("{}")
    hk = Housekeeper(s, plugin_path=plugin)
    await hk.start()
    try:
        assert not tmp_old.exists() and tmp_new.exists()
        assert not stale.exists() and fresh.exists() and users.exists()
        assert legacy.exists(), "the user has a server named hermes: its cache stays"
        assert not crash_tmp.exists()
        assert not (cfg.parent / "mcp_config.json.hermes.lock").exists()
        assert bak.exists(), "migration not recorded yet: the backup stays"
        hk.record_migration_done()
        await hk.sweep()
        assert not bak.exists()
        cfg.write_text(json.dumps({"mcpServers": {}}))
        await hk.sweep()
        assert not legacy.exists()
    finally:
        await hk.stop()
    _assert_user_intact(env)


# ── shared cache rewrite ────────────────────────────────────────────────────


def test_scrub_keeps_other_entries_and_skips_unparsable(env):
    ours = str(uuid.uuid4())
    _make_conv(env.root, ours, cwd="C:/agi-workspace")
    meta = env.root / "cache" / "conversation_metadata.json"
    extra = json.loads(meta.read_text())
    extra["other_key"] = 1
    meta.write_text(json.dumps(extra))
    assert scrub_shared_caches(env.root, [ours])
    last, ids = _cache_ids(env.root)
    assert ours not in last and ours not in ids
    assert json.loads(meta.read_text())["other_key"] == 1
    _assert_user_intact(env)
    last_file = env.root / "cache" / "last_conversations.json"
    last_file.write_text("{not json")
    assert scrub_shared_caches(env.root, [env.user_id]) is True  # skipped, not retried
    assert last_file.read_text() == "{not json"
    assert not list((env.root / "cache").glob(".agi-*"))


def test_scrub_abandons_when_agy_rewrites_meanwhile(env, monkeypatch):
    ours = str(uuid.uuid4())
    _make_conv(env.root, ours, cwd="C:/agi-workspace")
    target = env.root / "cache" / "last_conversations.json"
    real_read = Path.read_text
    calls = {"n": 0}

    def racing(self, *a, **k):
        text = real_read(self, *a, **k)
        if self.name == "last_conversations.json":
            calls["n"] += 1
            if calls["n"] == 2:  # the re-read right before the replace
                return text + " "
        return text
    monkeypatch.setattr(Path, "read_text", racing)
    assert scrub_shared_caches(env.root, [ours]) is False
    monkeypatch.setattr(Path, "read_text", real_read)
    assert ours in json.loads(target.read_text()).values()
    assert not list((env.root / "cache").glob(".agi-*"))


# ── F: purge ────────────────────────────────────────────────────────────────


async def test_purge_removes_whole_footprint(env, monkeypatch):
    plugin = Path(os.environ["AGI_AGENT_PLUGIN_PATH"])
    async with running_app() as (http, application):
        await _chat(http, history=None)
        await application.state.conv_manager.evict_idle()
        assert plugin.is_dir()
        # Leftovers as a crash would leave them (written after the sweep).
        ours = str(uuid.uuid4())
        _make_conv(env.root, ours, cwd="C:/agi-workspace")
        led = Ledger(env.state / hk_mod.LEDGER_NAME).open()
        led.add_conv(ours, "crashed-9", str(env.root))
        led.close()
        code, report = purge(Settings())
        assert code == 2  # the server holds its lock
    (env.root / "mcp" / "hermes-0badf00d").mkdir(parents=True)
    ws = Path(os.environ["AGI_DEFAULT_WORKDIR"])
    (ws / "attachments").mkdir(parents=True, exist_ok=True)
    (ws / "attachments" / "img-0123456789abcdef.png").write_bytes(b"png")
    cfg = Path(os.environ["AGI_MCP_CONFIG_PATH"])
    cfg.parent.mkdir(parents=True, exist_ok=True)
    (cfg.parent / "mcp_config.json.hermes-antigravity.bak").write_text("{}")
    code, report = purge(Settings())
    assert code == 0, report
    assert not conversation_exists(env.root, ours)
    assert not plugin.exists() and not env.state.exists()
    assert not (env.root / "mcp" / "hermes-0badf00d").exists()
    assert not ws.exists()
    assert not (cfg.parent / "mcp_config.json.hermes-antigravity.bak").exists()
    assert _all_uuids(env.root) == {env.user_id}
    _assert_user_intact(env)


def test_purge_cli_exit_codes(env, monkeypatch):
    from app import purge as purge_mod

    lock = InstanceLock(env.state / "run", "live-1")
    lock.acquire()
    try:
        assert purge_mod.main([]) == 2
    finally:
        lock.release()
    assert purge_mod.main([]) == 0
    assert not env.state.exists()
    _assert_user_intact(env)


def test_crash_markers_only_of_our_process_lifetime(tmp_path):
    from app.housekeeping import delete_crash_markers

    crashes = tmp_path / "crashes"
    crashes.mkdir()
    spawned, now = time.time() - 5, time.time()
    ours = crashes / f"crash_1234_{uuid.uuid4()}.log"
    stale = crashes / f"crash_1234_{uuid.uuid4()}.log"  # older process, same pid number
    newer = crashes / f"crash_1234_{uuid.uuid4()}.log"  # pid reused after our reap
    other = crashes / f"crash_12345_{uuid.uuid4()}.log"
    for p in (ours, stale, newer, other):
        p.write_text("")
    os.utime(stale, (OLD, OLD))
    os.utime(newer, (now + 60, now + 60))
    assert delete_crash_markers(tmp_path, 1234, before=time.time() + 1, after=spawned) == 1
    assert not ours.exists() and stale.exists() and newer.exists() and other.exists()
