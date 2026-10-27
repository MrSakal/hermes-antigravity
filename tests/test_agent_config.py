"""The single managed agent file: frontmatter shape for agy 1.4.8 (list-form
tools allowlist or no field, strict MCP list), write-if-changed, legacy
cleanup, toolset-hash naming, and the spawn lock held until the process has
read the file."""

from __future__ import annotations

import asyncio
import json
import os

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
    """Stand-in for AgySession: 'prints' its first line after ``delay``."""

    def __init__(self, mgr: AgentPluginManager, delay: float = 0.0, starts: bool = True) -> None:
        self.mgr, self.delay, self.starts = mgr, delay, starts
        self.agent = None
        self.read: str | None = None
        self.closed_while_locked: list[bool] = []

    async def start(self) -> None:
        pass

    async def wait_started(self, timeout: float) -> bool:
        if not self.starts:
            await asyncio.sleep(timeout)
            return False
        await asyncio.sleep(self.delay)
        # What the real process would have read at boot:
        self.read = self.mgr.agent_file.read_text(encoding="utf-8")
        return True

    async def aclose(self) -> None:
        self.closed_while_locked.append(self.mgr._lock.locked())


async def test_spawn_writes_single_file_and_manifest(tmp_path):
    mgr = AgentPluginManager(tmp_path / "hermes-bridge")
    assert mgr.available()
    s = _Session(mgr)
    await mgr.spawn(s, render_agent("You are a pirate."))
    assert s.agent == AGENT_NAME and "You are a pirate." in s.read
    manifest = json.loads((tmp_path / "hermes-bridge" / "plugin.json").read_text())
    assert manifest["name"] == "hermes-bridge"
    await mgr.spawn(_Session(mgr), render_agent("Other."))
    assert sorted(p.name for p in mgr.agents_dir.iterdir()) == [".hermes.md.lock", "hermes.md"]
    assert "Other." in mgr.agent_file.read_text(encoding="utf-8")


async def test_unchanged_content_is_not_rewritten(tmp_path, monkeypatch):
    mgr = AgentPluginManager(tmp_path / "p")
    md = render_agent("same")
    await mgr.spawn(_Session(mgr), md)
    writes = []
    monkeypatch.setattr(mgr, "_write_atomic", lambda *a: writes.append(a))
    await mgr.spawn(_Session(mgr), md)
    assert writes == []


async def test_externally_changed_file_is_put_back(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    md = render_agent("mine")
    await mgr.spawn(_Session(mgr), md)
    mgr.agent_file.write_text("tampered", encoding="utf-8")
    await mgr.spawn(_Session(mgr), md)
    assert mgr.agent_file.read_text(encoding="utf-8") == md


async def test_unwritable_path(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the plugin dir should be")
    mgr = AgentPluginManager(blocker)
    assert not mgr.available()
    with pytest.raises(AgentUnavailable):
        await mgr.spawn(_Session(mgr), "x")


async def test_lock_held_until_process_loaded(tmp_path):
    """A second spawn may not overwrite hermes.md before the first process
    printed its first line (i.e. read the file)."""
    mgr = AgentPluginManager(tmp_path / "p")
    first, second = _Session(mgr, delay=0.3), _Session(mgr)
    await asyncio.gather(mgr.spawn(first, render_agent("first")),
                         mgr.spawn(second, render_agent("second")))
    assert "first" in first.read and "second" in second.read


async def test_start_timeout_kills_before_release(tmp_path):
    """No proof agy read hermes.md: kill it while the file is still held, then
    fail the spawn — never release the file to a still-starting agy."""
    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=0.05)
    s = _Session(mgr, starts=False)
    with pytest.raises(SpawnTimeout):
        await mgr.spawn(s, "x")
    assert s.closed_while_locked == [True]
    assert not mgr._lock.locked()


async def test_cancelled_spawn_kills_before_release(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    s = _Session(mgr, delay=10)
    task = asyncio.create_task(mgr.spawn(s, "x"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert s.closed_while_locked == [True]
    assert not mgr._lock.locked()


def test_cleanup_legacy_only_touches_old_slugs(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    mgr.agents_dir.mkdir(parents=True)
    (mgr.agents_dir / "hermes-0123456789ab.md").write_text("old")
    (mgr.agents_dir / "hermes-fedcba987654.md").write_text("old")
    (mgr.agents_dir / "hermes.md").write_text("current")
    (mgr.agents_dir / "my-own-agent.md").write_text("hands off")
    assert mgr.cleanup_legacy() == 2
    assert sorted(p.name for p in mgr.agents_dir.iterdir()) == ["hermes.md", "my-own-agent.md"]
    assert AgentPluginManager(tmp_path / "missing").cleanup_legacy() == 0


def test_replace_retries_on_sharing_violation(tmp_path, monkeypatch):
    """Windows: os.replace onto a file agy holds open → PermissionError.
    Transient ones are retried; the content still lands."""
    import app.agent_config as ac

    target = tmp_path / "hermes.md"
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
    assert [p.name for p in tmp_path.iterdir()] == ["hermes.md"]  # no temp left


# ── cross-process lock, priority, boot failure ──────────────────────────────


async def test_two_managers_sharing_a_dir_serialize_via_the_file_lock(tmp_path):
    """Two servers on one HOME (two managers, separate in-process locks, one
    plugin dir): the OS file lock still keeps the second write out of the
    first process's write -> first-stdout window."""
    a, b = AgentPluginManager(tmp_path / "p"), AgentPluginManager(tmp_path / "p")
    first, second = _Session(a, delay=0.3), _Session(b)
    t1 = asyncio.create_task(a.spawn(first, render_agent("first")))
    await asyncio.sleep(0.05)  # a holds both locks now
    t2 = asyncio.create_task(b.spawn(second, render_agent("second")))
    await asyncio.gather(t1, t2)
    assert "first" in first.read and "second" in second.read
    assert a.lock_file.exists() and a.lock_file.name == ".hermes.md.lock"


async def test_file_lock_held_elsewhere_fails_busy_within_budget(tmp_path):
    from app.agent_config import SpawnBusy, _try_lock

    holder = AgentPluginManager(tmp_path / "p")
    holder.agents_dir.mkdir(parents=True)
    fd = os.open(str(holder.lock_file), os.O_RDWR | os.O_CREAT)
    try:
        assert _try_lock(fd)
        mgr = AgentPluginManager(tmp_path / "p")
        with pytest.raises(SpawnBusy):
            await mgr.spawn(_Session(mgr), "x", budget_s=0.2)
        assert not mgr._lock.locked()
    finally:
        holder._file_unlock(fd)


async def test_request_spawns_jump_ahead_of_background_spawns(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    order: list[str] = []

    class Tagged(_Session):
        def __init__(self, tag, delay=0.0):
            super().__init__(mgr, delay=delay)
            self.tag = tag

        async def start(self):
            order.append(self.tag)

    hold = asyncio.create_task(mgr.spawn(Tagged("holder", delay=0.2), "h"))
    await asyncio.sleep(0.02)
    refill = asyncio.create_task(mgr.spawn(Tagged("refill"), "r", priority=False))
    await asyncio.sleep(0.02)
    request = asyncio.create_task(mgr.spawn(Tagged("request"), "q"))
    await asyncio.gather(hold, refill, request)
    assert order == ["holder", "request", "refill"]


async def test_queued_request_fails_fast_when_queue_outlasts_budget(tmp_path):
    from app.agent_config import SpawnBusy

    mgr = AgentPluginManager(tmp_path / "p", start_timeout_s=5)
    hold = asyncio.create_task(mgr.spawn(_Session(mgr, delay=0.3), "h"))
    await asyncio.sleep(0.02)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(SpawnBusy):  # worst case 5 s ahead > 1 s budget
        await mgr.spawn(_Session(mgr), "x", budget_s=1.0)
    assert loop.time() - t0 < 0.1  # failed fast, did not wait
    await hold
    await mgr.spawn(_Session(mgr), "x", budget_s=1.0)  # queue empty: fine


async def test_boot_eof_fails_spawn_with_stderr_before_release(tmp_path):
    from app.agent_config import SpawnFailed

    mgr = AgentPluginManager(tmp_path / "p")

    class Dead(_Session):
        boot_failed = True

        async def exit_diagnostics(self):
            return "stderr: Error: not signed in; exit code 1"

    s = Dead(mgr)
    with pytest.raises(SpawnFailed, match="not signed in"):
        await mgr.spawn(s, "x")
    assert s.closed_while_locked == [True] and not mgr._lock.locked()


def test_custom_agent_name_and_manifest_follow_the_dir(tmp_path):
    mgr = AgentPluginManager(tmp_path / "hermes-bridge-bench-8800", agent_name="hermes-bench-8800")
    assert mgr.available()
    assert mgr.agent_file.name == "hermes-bench-8800.md"
    assert json.loads((mgr.path / "plugin.json").read_text())["name"] == "hermes-bridge-bench-8800"
    assert "name: hermes-bench-8800\n" in render_agent("x", name="hermes-bench-8800")


def test_bench_server_uses_its_own_plugin_and_agent(monkeypatch):
    """scripts/bench.py starts a second server on the same HOME: it must not
    share hermes.md (or the agent name) with the live one."""
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
