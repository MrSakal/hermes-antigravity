"""Managed agent plugin — ONE agent file (``hermes.md``) carrying each spawn's
system prompt, builtin-tool allowlist and strict MCP server list.

agy has no ``--system-prompt`` and no ``--mcp-config`` flag, but it loads
custom agents from plugin directories (``~/.gemini/config/plugins/<name>/``
with ``plugin.json`` + ``agents/*.md``), selectable per spawn with ``--agent``.
Verified live on 1.4.8:

* The ``.md`` body is the system prompt; it REPLACES the CLI's built-in
  Antigravity persona. An empty body is accepted but leaves the model with no
  persona at all, so a neutral one-liner is used when the request has none.
* ``tools:`` must be a YAML **list** and acts as an allowlist of builtins
  (``[view_file]`` → view_file + manage_task; ``[]`` → manage_task only). The
  scalar ``"*"`` older builds wrote is rejected (the agent is skipped and every
  turn fails "unknown agent") and ``["*"]`` fails at runtime — so the "all"
  mode simply omits the field, which grants every builtin incl.
  ``call_mcp_tool``.
* Frontmatter ``mcpServers`` (LIST form — the map form is rejected) gives the
  agent its own MCP servers; ``inheritMcp: false`` hides the user's servers
  from the shared ``mcp_config.json`` (builtins stay). agy contacts them only
  at the first user message, never during boot.
* The file is read at process BOOT, always before agy prints its first stdout
  line (the ``init`` event, printed even when no prompt was sent yet).
  Rewriting it after that line does not affect the running process.

That last point is what makes a single shared file safe: :meth:`spawn`
serializes every spawn (cold, resumed and warm-pool refills alike) under one
lock, writes the file only when its bytes differ, spawns, and holds the lock
until the process printed its first stdout line — message-free proof it has
read the file — before the next spawn may overwrite it. The same window is
also held under an advisory OS file lock (``agents/.hermes.md.lock``;
``flock`` on POSIX, ``msvcrt.locking`` on Windows, polled so the event loop
never blocks), so a second server on the same HOME cannot rewrite the file
under our boot either. Request-driven spawns take precedence over warm-pool
refills, and an agy that exits before its first line fails the spawn
(:class:`SpawnFailed`, carrying its stderr tail) instead of handing out a
dead process.

agy caches tool schemas per server NAME in a shared folder
(``~/.gemini/antigravity-cli/mcp/<name>/``), so the server name is unique per
distinct toolset: ``hermes-<sha8 of the canonical tools JSON>``. The
conversation itself is routed by the URL (``/mcp/<conv_id>``), not the name.

Only this plugin directory is ever written. On startup :meth:`cleanup_legacy`
deletes the per-prompt ``hermes-<12hex>.md`` files older builds generated.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("agi.agentcfg")

AGENT_NAME = "hermes"
_LEGACY_RE = re.compile(r"^hermes-[0-9a-f]{12}\.md$")

_PLUGIN_JSON = {
    "name": "hermes-bridge",
    "version": "0.3.0",
    "description": "System prompt + strict per-conversation MCP agent managed by antigravity-interface.",
    "author": {"name": "antigravity-interface"},
}

# tools-mode name → frontmatter `tools:` line (None = omit the field). See
# Settings.plain_chat_tools and the module docstring for the 1.4.8 rules.
TOOLS_BY_MODE: dict[str, Optional[str]] = {
    "all": None,
    "readonly": "[view_file]",
    "none": "[]",
}

# Body used when a request carries no system message (see module docstring).
# Neutral: does not fight the client's own instructions in the user turn.
NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

# os.replace onto a file another process holds open fails on Windows with
# PermissionError (sharing violation) — agy may be reading hermes.md right
# then. Retry briefly before giving up.
_REPLACE_RETRIES = 5
_REPLACE_BACKOFF_S = 0.05

# Poll interval of the cross-process agent-file lock (never a blocking call on
# the event loop).
_FILE_LOCK_POLL_S = 0.05

if os.name == "nt":  # pragma: no cover - exercised on Windows
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, PermissionError):
            return False

    def _unlock(fd: int) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)


class AgentUnavailable(Exception):
    """The managed plugin directory cannot be written, so the spawn cannot get
    its system prompt / tool allowlist / MCP server. Callers fail closed."""


class SpawnTimeout(Exception):
    """agy printed nothing within ``start_timeout_s`` of spawning, so there is
    no proof it has read the agent file. The process is killed and the spawn
    fails: releasing the lock instead would let a slow agy read the NEXT
    spawn's file (wrong system prompt, wrong tools, wrong MCP URL)."""


class SpawnFailed(Exception):
    """agy exited during boot without printing a single stdout line (not
    signed in, unknown model, rejected agent frontmatter, ...). ``str()``
    carries agy's stderr tail / exit code."""


class SpawnBusy(Exception):
    """The spawn could not get the agent-file lock within its budget — the
    queue ahead of it (in this process or in another server sharing the
    plugin directory) would outlast the request. Rendered as 503."""


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    url: str


def default_plugin_path() -> Path:
    return Path("~/.gemini/config/plugins/hermes-bridge").expanduser()


def toolset_hash(tools: Optional[list[Any]]) -> str:
    """sha256 (hex) of the canonical JSON of an OpenAI ``tools`` list."""
    canon = [
        t.model_dump(mode="json") if hasattr(t, "model_dump") else t
        for t in (tools or [])
    ]
    payload = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def server_name_for(tools: Optional[list[Any]]) -> str:
    """MCP server name for a toolset — unique per distinct schema set because
    agy caches schemas per server name (see module docstring)."""
    return "hermes-" + toolset_hash(tools)[:8]


def agent_digest(content: str) -> str:
    """Identity of a rendered agent file (warm-pool signature, cache namespace)."""
    return hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _yaml_str(s: str) -> str:
    # JSON string syntax is valid YAML double-quoted scalar syntax.
    return json.dumps(s, ensure_ascii=False)


def render_agent(
    system_prompt: Optional[str],
    *,
    tools_mode: str = "all",
    servers: Optional[list[McpServerSpec]] = None,
    inherit_mcp: bool = False,
    name: str = AGENT_NAME,
) -> str:
    """The full ``hermes.md`` content for one spawn (``name`` must match the
    managing :attr:`AgentPluginManager.agent_name`)."""
    if tools_mode not in TOOLS_BY_MODE:
        raise ValueError(f"unknown tools mode {tools_mode!r}")
    lines = [
        "---",
        f"name: {name}",
        f"description: Hermes provider agent managed by antigravity-interface (tools={tools_mode}).",
    ]
    tools = TOOLS_BY_MODE[tools_mode]
    if tools is not None:
        lines.append(f"tools: {tools}")
    lines.append(f"inheritMcp: {'true' if inherit_mcp else 'false'}")
    if servers:
        lines.append("mcpServers:")
        for s in servers:
            lines.append(f"  - name: {s.name}")
            lines.append(f"    serverUrl: {_yaml_str(s.url)}")
    lines.append("---")
    body = (system_prompt or "").strip() or NEUTRAL_SYSTEM_PROMPT
    return "\n".join(lines) + "\n\n" + body + "\n"


class AgentPluginManager:
    """Owns the managed plugin directory and its single agent file."""

    def __init__(self, path: Optional[Path] = None, *, start_timeout_s: float = 30.0,
                 agent_name: str = AGENT_NAME) -> None:
        self.path = Path(path).expanduser() if path else default_plugin_path()
        # A second server on the same HOME (scripts/bench.py) uses its own
        # plugin dir AND agent name: two plugins defining one agent name would
        # make ``--agent`` ambiguous.
        self.agent_name = agent_name
        self.start_timeout_s = start_timeout_s
        self._lock = asyncio.Lock()
        # Request-driven spawns waiting for (or holding) _lock. Background
        # spawns (warm-pool refills) only take the lock while this is 0.
        self._urgent = 0
        self._no_urgent = asyncio.Event()
        self._no_urgent.set()
        # available() is on every request's path; once the directory and
        # manifest exist, skip the filesystem round trip (a spawn re-ensures
        # both under the lock anyway, see _write_if_changed).
        self._ensured = False

    @property
    def agents_dir(self) -> Path:
        return self.path / "agents"

    @property
    def agent_file(self) -> Path:
        return self.agents_dir / f"{self.agent_name}.md"

    @property
    def lock_file(self) -> Path:
        """Cross-process advisory lock next to ``hermes.md`` (not ``*.md``, so
        agy never mistakes it for an agent)."""
        return self.agents_dir / f".{self.agent_name}.md.lock"

    # ── public API ───────────────────────────────────────────────────────────

    def available(self) -> bool:
        """Create the plugin dir + manifest if needed; False when unwritable.

        Callers check this BEFORE building the prompt: without an agent the
        system message must ride as a preamble instead (or the request fails
        closed when it needs a tool allowlist / MCP server)."""
        if self._ensured:
            return True
        try:
            self._ensure_plugin()
            self._ensured = True
            return True
        except OSError:
            logger.warning("agent-plugin: cannot manage %s", self.path, exc_info=True)
            return False

    async def spawn(self, session: Any, content: str, *, priority: bool = True,
                    budget_s: Optional[float] = None) -> None:
        """Start ``session`` with ``--agent hermes`` reading exactly ``content``.

        Holds the in-process lock AND the cross-process file lock from the
        file write through the process's first stdout line. Any failure or
        cancellation in between kills and reaps the process BEFORE the locks
        are released.

        ``priority`` (request-driven spawns) jumps ahead of background spawns
        (``priority=False``: warm-pool refills), which only take the lock while
        no request spawn is queued. ``budget_s`` bounds the time spent waiting
        for the locks; a queue that would outlast it fails fast.

        Raises :class:`AgentUnavailable` (file not writable),
        :class:`SpawnBusy` (lock budget exceeded), :class:`SpawnTimeout` (no
        output) or :class:`SpawnFailed` (agy exited during boot).
        """
        loop = asyncio.get_running_loop()
        deadline = None if budget_s is None else loop.time() + max(0.0, budget_s)
        await self._acquire(priority, deadline)
        try:
            lock_budget = (self.start_timeout_s * 2 + 5.0 if deadline is None
                           else deadline - loop.time())
            fd = await self._file_lock(lock_budget)
            try:
                await self._spawn_locked(session, content)
            finally:
                self._file_unlock(fd)
        finally:
            self._lock.release()

    async def _spawn_locked(self, session: Any, content: str) -> None:
        # Synchronous on purpose: a write running in a worker thread could
        # outlive a cancelled spawn and land after the NEXT spawn's write.
        try:
            self._write_if_changed(content)
        except OSError as e:
            self._ensured = False
            logger.warning("agent-plugin: cannot write %s", self.agent_file, exc_info=True)
            raise AgentUnavailable(str(e)) from e
        session.agent = self.agent_name
        try:
            await session.start()
            loaded = await session.wait_started(self.start_timeout_s)
        except BaseException:
            await session.aclose()
            raise
        if not loaded:
            logger.warning(
                "agent-plugin: no output from agy within %.0fs; killing it before "
                "the agent file is released", self.start_timeout_s,
            )
            await session.aclose()
            raise SpawnTimeout(f"agy did not start within {self.start_timeout_s:.0f}s")
        if getattr(session, "boot_failed", False):
            try:
                diag = await session.exit_diagnostics()
            finally:
                await session.aclose()
            raise SpawnFailed(f"agy exited during startup ({diag})")

    # ── locking ──────────────────────────────────────────────────────────────

    async def _acquire(self, priority: bool, deadline: Optional[float]) -> None:
        """Take ``_lock``: request spawns first, background spawns only while
        none is queued. A request spawn whose queue would outlast its budget
        (one ``start_timeout_s`` per spawn ahead, the worst case) fails fast."""
        loop = asyncio.get_running_loop()
        if not priority:
            while True:
                await self._no_urgent.wait()
                if deadline is None:
                    await self._lock.acquire()
                else:
                    try:
                        async with asyncio.timeout_at(deadline):
                            await self._lock.acquire()
                    except TimeoutError:
                        raise SpawnBusy("agent spawn queue is busy") from None
                if self._urgent == 0:
                    return
                self._lock.release()  # a request spawn queued meanwhile: yield
        if deadline is not None and self._lock.locked():
            ahead = self._urgent + 1  # queued request spawns + the current holder
            if ahead * self.start_timeout_s >= deadline - loop.time():
                # The worst case of the queue ahead (each holder bounded by
                # start_timeout_s) would outlast this request.
                raise SpawnBusy(f"{ahead} agy spawn(s) queued ahead; retry later")
        self._urgent += 1
        self._no_urgent.clear()
        try:
            if deadline is None:
                await self._lock.acquire()
            else:
                try:
                    async with asyncio.timeout_at(deadline):
                        await self._lock.acquire()
                except TimeoutError:
                    raise SpawnBusy("timed out waiting for the agent spawn queue") from None
        finally:
            self._urgent -= 1
            if self._urgent == 0:
                self._no_urgent.set()

    async def _file_lock(self, budget_s: float) -> Optional[int]:
        """Advisory OS lock on :attr:`lock_file`, held for the same write →
        first-stdout window as ``_lock``: a second server sharing this plugin
        directory (same HOME) must not rewrite ``hermes.md`` under our boot.
        Polled non-blocking so the event loop never blocks. Returns the fd, or
        None when the lock file cannot be created (the in-process lock still
        holds; :meth:`_write_if_changed` then reports the real error)."""
        try:
            self.agents_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.lock_file), os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            logger.warning("agent-plugin: cannot open %s; cross-process lock skipped",
                           self.lock_file, exc_info=True)
            return None
        loop = asyncio.get_running_loop()
        end = loop.time() + max(0.0, budget_s)
        try:
            while not _try_lock(fd):
                if loop.time() >= end:
                    raise SpawnBusy("another server holds the agent file lock")
                await asyncio.sleep(_FILE_LOCK_POLL_S)
        except BaseException:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def _file_unlock(fd: Optional[int]) -> None:
        if fd is None:
            return
        try:
            _unlock(fd)
        finally:
            os.close(fd)

    def cleanup_legacy(self) -> int:
        """Delete per-prompt ``hermes-<12hex>.md`` files from older builds
        (ours only — foreign agents are never touched). Returns the count."""
        removed = 0
        try:
            entries = list(self.agents_dir.iterdir())
        except OSError:
            return 0
        for p in entries:
            if _LEGACY_RE.match(p.name):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    logger.warning("agent-plugin: could not delete legacy %s", p.name)
        if removed:
            logger.info("agent-plugin: removed %d legacy per-prompt agent file(s)", removed)
        return removed

    # ── internals ────────────────────────────────────────────────────────────

    def _write_if_changed(self, content: str) -> None:
        self._ensure_plugin()
        target = self.agent_file
        want = content.encode("utf-8", "replace")
        try:
            if target.read_bytes() == want:
                return
        except OSError:
            pass
        self._write_atomic(target, content)
        logger.debug("agent-plugin: wrote %s", target)

    def _ensure_plugin(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "plugin.json"
        manifest_json = dict(_PLUGIN_JSON, name=self.path.name or _PLUGIN_JSON["name"])
        want = json.dumps(manifest_json, indent=2) + "\n"
        try:
            have = manifest.read_text(encoding="utf-8")
        except OSError:
            have = None
        if have != want:
            self._write_atomic(manifest, want)
            logger.info("agent-plugin: wrote manifest %s", manifest)

    @staticmethod
    def _write_atomic(target: Path, content: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            # errors="replace": a lone surrogate in a client's system prompt
            # must not make the file unwritable.
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace", newline="\n") as f:
                f.write(content)
            for attempt in range(_REPLACE_RETRIES):
                try:
                    os.replace(tmp, str(target))
                    break
                except PermissionError:
                    if attempt == _REPLACE_RETRIES - 1:
                        raise
                    time.sleep(_REPLACE_BACKOFF_S)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
