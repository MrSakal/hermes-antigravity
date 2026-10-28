"""Transient per-spawn agent files: frontmatter shape for agy 1.4.8 (list-form
tools allowlist or no field, strict MCP list), toolset-hash naming, parallel
spawns, delete-after-first-line, startup + periodic sweeps."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import pytest

from app.agent_config import (
    AGENT_NAME,
    NEUTRAL_SYSTEM_PROMPT,
    AgentPluginManager,
    AgentUnavailable,
    McpServerSpec,
    SpawnTimeout,
    agent_digest,
    render_agent,
    server_name_for,
)
from app.openai_models import ToolDef


def _front(md: str) -> str:
    return md.split("---\n")[1]


def test_render_strict_mcp_list_form_and_all_tools_omits_field():
    md = render_agent("be brief", servers=[
        McpServerSpec("hermes-ab12cd34", "http://127.0.0.1:8788/mcp/c1")])
    front = _front(md)
    assert f"name: {AGENT_NAME}\n" in front
    assert "inheritMcp: false" in front
    assert ('mcpServers:\n  - name: hermes-ab12cd34\n'
            '    serverUrl: "http://127.0.0.1:8788/mcp/c1"') in front
    # agy 1.4.8 rejects `tools: "*"` (agent skipped → "unknown agent"); the
    # full builtin set incl. call_mcp_tool is what an absent field grants.
    assert "tools:" not in front
    assert md.rstrip().endswith("be brief")


@pytest.mark.parametrize("mode,line", [("readonly", "tools: [view_file]"), ("none", "tools: []")])
def test_restricted_modes_are_yaml_list_allowlists(mode, line):
    front = _front(render_agent("p", tools_mode=mode))
    assert line + "\n" in front
    assert '"*"' not in front


def test_render_inherit_and_no_servers_and_neutral_body():
    md = render_agent(None, inherit_mcp=True)
    front = _front(md)
    assert "inheritMcp: true" in front
    assert "mcpServers" not in front
    # Empty body would leave the model persona-less (verified live) → one-liner.
    assert md.rstrip().endswith(NEUTRAL_SYSTEM_PROMPT)


def test_unknown_tools_mode_rejected():
    with pytest.raises(ValueError):
        render_agent("x", tools_mode="everything")


def test_digest_tracks_content():
    a, b = render_agent("A"), render_agent("B")
    assert agent_digest(a) == agent_digest(render_agent("A")) != agent_digest(b)
    assert agent_digest(render_agent("A", tools_mode="none")) != agent_digest(a)


def test_server_name_is_toolset_hash():
    a = [ToolDef(function={"name": "get_weather", "parameters": {"type": "object"}})]
    b = [ToolDef(function={"name": "get_time", "parameters": {"type": "object"}})]
    na, nb = server_name_for(a), server_name_for(b)
    assert na.startswith("hermes-") and len(na) == len("hermes-") + 8
    assert na != nb
    assert na == server_name_for([ToolDef(function={"name": "get_weather",
                                                     "parameters": {"type": "object"}})])


class _Session:
    """Stand-in for AgySession: 'prints' its first line after ``delay``,
    having read its agent file (like agy at boot)."""

    def __init__(self, mgr: AgentPluginManager, delay: float = 0.0, starts: bool = True) -> None:
        self.mgr, self.delay, self.starts = mgr, delay, starts
        self.agent = None
        self.read: str | None = None
        self.file_at_close: list[bool] = []

    @property
    def path(self):
        return self.mgr.agents_dir / f"{self.agent}.md"

    async def start(self) -> None:
        pass

    async def wait_started(self, timeout: float) -> bool:
        if not self.starts:
            await asyncio.sleep(timeout)
            return False
        await asyncio.sleep(self.delay)
        self.read = self.path.read_text(encoding="utf-8")
        return True

    async def aclose(self) -> None:
        # The kill comes first; the file is deleted after it.
        self.file_at_close.append(self.path.exists())


async def test_spawn_uses_a_transient_file_deleted_after_first_line(tmp_path):
    mgr = AgentPluginManager(tmp_path / "hermes-bridge")
    assert mgr.available()
    s = _Session(mgr)
    await mgr.spawn(s, render_agent("You are a pirate."))
    assert re.fullmatch(r"hermes-[0-9a-f]{12}", s.agent)
    assert s.read.startswith(f"---\nname: {s.agent}\n") and "You are a pirate." in s.read
    manifest = json.loads((tmp_path / "hermes-bridge" / "plugin.json").read_text())
    assert manifest["name"] == "hermes-bridge"
    # Gone once the process printed its first line; no lock file either.
    assert sorted(p.name for p in mgr.agents_dir.iterdir()) == []
    s2 = _Session(mgr)
    await mgr.spawn(s2, render_agent("Other."))
    assert s2.agent != s.agent and "Other." in s2.read


async def test_parallel_spawns_do_not_serialize_and_each_reads_its_own(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    sessions = [_Session(mgr, delay=0.5) for _ in range(10)]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await asyncio.gather(*(mgr.spawn(s, render_agent(f"persona {i}"))
                           for i, s in enumerate(sessions)))
    assert loop.time() - t0 < 2.0  # one after another would be >= 5 s
    for i, s in enumerate(sessions):
        assert f"persona {i}" in s.read
    assert len({s.agent for s in sessions}) == 10
    assert mgr.transient_files() == [] and mgr.booting == 0


async def test_digest_ignores_the_per_spawn_name(tmp_path):
    """Pool signature / reuse key = template digest; two spawns of the same
    template get different file names but the same identity."""
    mgr = AgentPluginManager(tmp_path / "p")
    md = render_agent("same")
    a, b = _Session(mgr), _Session(mgr)
    await mgr.spawn(a, md)
    await mgr.spawn(b, md)
    assert a.agent != b.agent
    assert a.read.replace(a.agent, "X") == b.read.replace(b.agent, "X")
    assert agent_digest(md) == agent_digest(render_agent("same"))


async def test_unwritable_path(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the plugin dir should be")
    mgr = AgentPluginManager(blocker)
    assert not mgr.available()
    with pytest.raises(AgentUnavailable):
        await mgr.spawn(_Session(mgr), "x")


async def test_start_timeout_kills_then_deletes(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=0.05)
    s = _Session(mgr, starts=False)
    with pytest.raises(SpawnTimeout):
        await mgr.spawn(s, "x")
    assert s.file_at_close == [True]  # killed while the file still existed
    assert mgr.transient_files() == [] and mgr.booting == 0


async def test_cancelled_spawn_kills_then_deletes(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    s = _Session(mgr, delay=10)
    task = asyncio.create_task(mgr.spawn(s, "x"))
    await asyncio.sleep(0.05)
    assert mgr.booting == 1 and len(mgr.transient_files()) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert s.file_at_close == [True]
    assert mgr.transient_files() == [] and mgr.booting == 0


async def test_boot_eof_fails_spawn_with_stderr_and_deletes(tmp_path):
    from app.agent_config import SpawnFailed

    mgr = AgentPluginManager(tmp_path / "p")

    class Dead(_Session):
        boot_failed = True

        async def exit_diagnostics(self):
            return "stderr: Error: not signed in; exit code 1"

    s = Dead(mgr)
    with pytest.raises(SpawnFailed, match="not signed in"):
        await mgr.spawn(s, "x")
    assert s.file_at_close == [True] and mgr.transient_files() == []


async def test_start_exception_kills_then_deletes(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")

    class Boom(_Session):
        async def start(self):
            raise RuntimeError("agy not found")

    s = Boom(mgr)
    with pytest.raises(RuntimeError):
        await mgr.spawn(s, "x")
    assert s.file_at_close == [True] and mgr.transient_files() == []


# ── housekeeping ────────────────────────────────────────────────────────────


def _age(p, seconds):
    t = time.time() - seconds
    os.utime(p, (t, t))


def test_startup_cleanup(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=10)
    d = mgr.agents_dir
    d.mkdir(parents=True)
    for name in ("hermes.md", ".hermes.md.lock",            # previous build
                 "hermes-0123456789ab.md",                  # crash leftover / legacy
                 "hermes-fedcba987654.md",
                 "hermes-aaaaaaaaaaaa.md",                  # fresh: maybe another server booting
                 "my-own-agent.md", "hermes-notes.md",      # not ours
                 "hermes-0123456789abc.md", "hermes-0123456789xy.md"):
        (d / name).write_text("x")
    for name in ("hermes-0123456789ab.md", "hermes-fedcba987654.md"):
        _age(d / name, 60)
    assert mgr.startup_cleanup() == 4
    assert sorted(p.name for p in d.iterdir()) == [
        "hermes-0123456789abc.md", "hermes-0123456789xy.md", "hermes-aaaaaaaaaaaa.md",
        "hermes-notes.md", "my-own-agent.md"]
    assert AgentPluginManager(tmp_path / "missing").startup_cleanup() == 0


def test_startup_cleanup_custom_name_leaves_default_agents_alone(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p", agent_name="hermes-bench-8800", start_timeout_s=1)
    d = mgr.agents_dir
    d.mkdir(parents=True)
    for name in ("hermes-bench-8800-0123456789ab.md", "hermes-bench-8800.md", "hermes-x.md"):
        (d / name).write_text("x")
        _age(d / name, 60)
    mgr.startup_cleanup()
    assert sorted(p.name for p in d.iterdir()) == ["hermes-x.md"]


async def test_periodic_sweep_skips_booting_and_young_files(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=5)
    s = _Session(mgr, delay=10)
    task = asyncio.create_task(mgr.spawn(s, "x"))
    await asyncio.sleep(0.05)
    [booting] = mgr.transient_files()
    _age(booting, 3600)  # even an "old" file of a live boot is never swept
    stale = mgr.agents_dir / "hermes-0000000000aa.md"
    young = mgr.agents_dir / "hermes-0000000000bb.md"
    stale.write_text("x")
    young.write_text("x")
    _age(stale, 11)   # > 2 x start_timeout_s
    _age(young, 9)    # < 2 x start_timeout_s
    other = mgr.agents_dir / "someone-else-0000000000aa.md"
    other.write_text("x")
    _age(other, 3600)
    assert mgr.sweep() == 1
    assert not stale.exists() and young.exists() and other.exists() and booting.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_gc_loop_runs_the_sweep(tmp_path, monkeypatch):
    from app.config import Settings
    from app.conversation import ConversationManager
    from app.mcp_bridge import McpBridge

    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=1)
    calls = []
    monkeypatch.setattr(mgr, "sweep", lambda **kw: calls.append(kw) or 0)
    cm = ConversationManager(McpBridge(), Settings(_env_file=None, gc_interval_s=1,
                                                   default_workdir=tmp_path), agents=mgr)
    task = asyncio.create_task(cm.gc_loop())
    await asyncio.sleep(1.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls


async def test_windows_delete_retry_then_sweep(tmp_path, monkeypatch):
    """agy may still hold the file a moment after its first line (Windows
    sharing violation): retried in the background without delaying the
    spawn; a file that stays locked is left to the sweep."""
    import app.agent_config as ac

    monkeypatch.setattr(ac, "_DELETE_BACKOFF_S", 0.01)
    real_unlink = type(tmp_path).unlink
    locked = {"n": 3}

    def flaky(self, *a, **kw):
        if self.suffix == ".md" and locked["n"] > 0:
            locked["n"] -= 1
            raise PermissionError("sharing violation")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(type(tmp_path), "unlink", flaky)
    mgr = AgentPluginManager(tmp_path / "p")
    s = _Session(mgr)
    await mgr.spawn(s, "x")  # returns at once; delete continues in background
    assert len(mgr.transient_files()) == 1
    await mgr.aclose()
    assert mgr.transient_files() == []

    # Locked for good: background retries give up, the sweep removes it later.
    locked["n"] = 10**6
    s2 = _Session(mgr)
    await mgr.spawn(s2, "x")
    await mgr.aclose()
    assert len(mgr.transient_files()) == 1
    locked["n"] = 0
    assert mgr.sweep() == 1  # own undeleted file: no age gate
    assert mgr.transient_files() == []


def test_replace_retries_on_sharing_violation(tmp_path, monkeypatch):
    """Windows: os.replace onto a file another process holds open →
    PermissionError. Transient ones are retried; the content still lands."""
    import app.agent_config as ac

    target = tmp_path / "plugin.json"
    target.write_text("old", encoding="utf-8")
    real = os.replace
    fails = iter([True, True])

    def flaky(src, dst):
        if next(fails, False):
            raise PermissionError("sharing violation")
        return real(src, dst)

    monkeypatch.setattr(ac.os, "replace", flaky)
    monkeypatch.setattr(ac, "_REPLACE_BACKOFF_S", 0.0)
    AgentPluginManager._write_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["plugin.json"]  # no temp left


def test_custom_agent_name_and_manifest_follow_the_dir(tmp_path):
    mgr = AgentPluginManager(tmp_path / "hermes-bridge-bench-8800", agent_name="hermes-bench-8800")
    assert mgr.available()
    assert mgr.agent_file.name == "hermes-bench-8800.md"
    assert json.loads((mgr.path / "plugin.json").read_text())["name"] == "hermes-bridge-bench-8800"
    assert "name: hermes-bench-8800\n" in render_agent("x", name="hermes-bench-8800")


def test_bench_server_uses_its_own_plugin_and_agent(monkeypatch):
    """scripts/bench.py starts a second server on the same HOME: it must not
    share the agent directory (or the agent name) with the live one."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_under_test", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen.update(kw["env"])
            self.stderr = iter(())

        def poll(self):
            return 0

    monkeypatch.setattr(bench.subprocess, "Popen", FakePopen)
    srv = bench.ServerHandle(8811, None, None, None)
    srv.start()
    assert seen["AGI_AGENT_PLUGIN_PATH"].endswith("hermes-bridge-bench-8811")
    assert seen["AGI_AGENT_NAME"] == "hermes-bench-8811"
    assert Path(seen["AGI_AGENT_PLUGIN_PATH"]) != Path("~/.gemini/config/plugins/hermes-bridge").expanduser()
