"""Managed agent plugin — one TRANSIENT agent file per spawn carrying that
spawn's system prompt, builtin-tool allowlist and strict MCP server list.

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
  Deleting it after that line changes nothing for the running process: the
  persona applies and the MCP URL from the deleted file is still connected at
  the first message (both checked live, with the message sent before AND
  after the deletion).
* agy snapshots the agent (persona + MCP servers) into the conversation it
  creates; a ``--conversation <id>`` resume ignores ``--agent`` entirely and
  reconnects to the ORIGINAL MCP URL. The server never resumes (an exited
  process' conversation is deleted, see app.conversation).

So every spawn gets its own file, ``agents/<agent_name>-<12 hex>.md`` with a
matching frontmatter ``name``, written right before the spawn and deleted as
soon as the process printed its first stdout line (or right after the kill on
timeout / boot failure / cancellation). There is no lock: request spawns,
and warm-pool refills all boot in parallel, and a second
server sharing the directory cannot collide either (random names). An agy that
exits before its first line fails the spawn (:class:`SpawnFailed`, carrying
its stderr tail) instead of handing out a dead process.

Housekeeping: a delete refused with ``PermissionError`` (Windows: agy may
still hold the handle for a moment) is retried in the background and
otherwise left to :meth:`AgentPluginManager.sweep`, which removes our
transient files that no in-flight spawn owns once they are older than any boot
can take (at startup and from the GC loop). The startup sweep also removes the
per-prompt ``hermes-<12hex>.md`` files of older builds and the single
``<agent_name>.md`` + ``.<agent_name>.md.lock`` of the previous build. Nothing
else in the directory is ever touched.

agy caches tool schemas per server NAME in a shared folder
(``~/.gemini/antigravity-cli/mcp/<name>/``), so the server name is unique per
distinct toolset: ``hermes-<sha8 of the canonical tools JSON>``. The
conversation itself is routed by the URL (``/mcp/<conv_id>``), not the name.
Warm-pool / reuse signatures are digests of the rendered TEMPLATE (base agent
name), never of the random per-spawn file name.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
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
    "version": "0.4.0",
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

# os.replace / os.unlink on a file another process holds open fails on Windows
# with PermissionError (sharing violation). Retry briefly before giving up.
_REPLACE_RETRIES = 5
_REPLACE_BACKOFF_S = 0.05
# Background delete retries of a transient file agy still holds (Windows);
# whatever survives them is left to the sweep.
_DELETE_RETRIES = 8
_DELETE_BACKOFF_S = 0.25


class AgentUnavailable(Exception):
    """The managed plugin directory cannot be written, so the spawn cannot get
    its system prompt / tool allowlist / MCP server. Callers fail closed."""


class SpawnTimeout(Exception):
    """agy printed nothing within ``start_timeout_s`` of spawning. The process
    is killed (then its agent file deleted) and the spawn fails."""


class SpawnFailed(Exception):
    """agy exited during boot without printing a single stdout line (not
    signed in, unknown model, rejected agent frontmatter, ...). ``str()``
    carries agy's stderr tail / exit code."""


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
    """Identity of a rendered agent template (warm-pool signature, cache namespace)."""
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
    """The agent TEMPLATE for one spawn (``name`` must be the managing
    :attr:`AgentPluginManager.agent_name`; :meth:`AgentPluginManager.spawn`
    swaps in the per-spawn file name)."""
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
    """Owns the managed plugin directory and its transient per-spawn agents."""

    def __init__(self, path: Optional[Path] = None, *, start_timeout_s: float = 30.0,
                 agent_name: str = AGENT_NAME) -> None:
        self.path = Path(path).expanduser() if path else default_plugin_path()
        # A second server on the same HOME (scripts/bench.py) uses its own
        # plugin dir AND agent name, so neither sweeps the other's files.
        self.agent_name = agent_name
        self.start_timeout_s = start_timeout_s
        self._transient_re = re.compile(rf"^{re.escape(agent_name)}-[0-9a-f]{{12}}\.md$")
        # Transient files of spawns still booting (never swept).
        self._booting: set[Path] = set()
        # Background delete retries (strong refs) and our files they have not
        # removed yet.
        self._deleters: set[asyncio.Task] = set()
        self._undeleted: set[Path] = set()
        # available() is on every request's path; once the directory and
        # manifest exist, skip the filesystem round trip.
        self._ensured = False

    @property
    def agents_dir(self) -> Path:
        return self.path / "agents"

    @property
    def agent_file(self) -> Path:
        """The single shared file of the previous build (removed at startup)."""
        return self.agents_dir / f"{self.agent_name}.md"

    @property
    def lock_file(self) -> Path:
        """The previous build's cross-process lock file (removed at startup)."""
        return self.agents_dir / f".{self.agent_name}.md.lock"

    @property
    def booting(self) -> int:
        """Spawns between file write and first stdout line (metrics / tests)."""
        return len(self._booting)

    def transient_files(self) -> list[Path]:
        """Our per-spawn agent files currently on disk."""
        try:
            return sorted(p for p in self.agents_dir.iterdir() if self._transient_re.match(p.name))
        except OSError:
            return []

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

    async def spawn(self, session: Any, content: str) -> None:
        """Start ``session`` with ``--agent <agent_name>-<hex>``: a fresh file
        holding exactly ``content`` (a :func:`render_agent` template; its
        ``name:`` line is rewritten to the per-spawn name).

        The file is deleted once the process printed its first stdout line —
        agy has read it by then — or, after the kill, on any failure or
        cancellation in between. Never waits on another spawn.

        Raises :class:`AgentUnavailable` (file not writable),
        :class:`SpawnTimeout` (no output within ``start_timeout_s``) or
        :class:`SpawnFailed` (agy exited during boot).
        """
        name = f"{self.agent_name}-{secrets.token_hex(6)}"
        path = self.agents_dir / f"{name}.md"
        try:
            self._ensure_plugin()
            self._write_atomic(path, self._personalize(content, name))
        except OSError as e:
            self._ensured = False
            logger.warning("agent-plugin: cannot write %s", path, exc_info=True)
            raise AgentUnavailable(str(e)) from e
        self._booting.add(path)
        try:
            session.agent = name
            try:
                await session.start()
                loaded = await session.wait_started(self.start_timeout_s)
            except BaseException:
                await session.aclose()
                raise
            if not loaded:
                logger.warning("agent-plugin: no output from agy within %.0fs; killing it",
                               self.start_timeout_s)
                await session.aclose()
                raise SpawnTimeout(f"agy did not start within {self.start_timeout_s:.0f}s")
            if getattr(session, "boot_failed", False):
                try:
                    diag = await session.exit_diagnostics()
                finally:
                    await session.aclose()
                raise SpawnFailed(f"agy exited during startup ({diag})")
        finally:
            self._booting.discard(path)
            self._delete(path)

    def startup_cleanup(self) -> int:
        """Startup housekeeping: the previous build's single agent file + lock
        file, legacy per-prompt files and leftover transient files (a crash
        between write and delete). Returns the count removed.

        Transient files younger than one boot window are kept: they may belong
        to another server booting right now on the same directory; the
        periodic :meth:`sweep` removes them once stale."""
        removed = 0
        for p in (self.agent_file, self.lock_file):
            if self._unlink_quiet(p):
                removed += 1
        try:
            entries = list(self.agents_dir.iterdir())
        except OSError:
            entries = []
        for p in entries:
            if _LEGACY_RE.match(p.name) and not self._transient_re.match(p.name):
                if self._unlink_quiet(p):
                    removed += 1
        removed += self.sweep(min_age_s=self.start_timeout_s + 5.0)
        if removed:
            logger.info("agent-plugin: startup cleanup removed %d file(s)", removed)
        return removed

    # Older name (pre-transient builds).
    cleanup_legacy = startup_cleanup

    def sweep(self, *, min_age_s: Optional[float] = None) -> int:
        """Delete our transient agent files that no in-flight spawn owns and
        that are older than ``min_age_s`` (default: 2x ``start_timeout_s``,
        longer than any boot may take). Returns the count removed."""
        if min_age_s is None:
            min_age_s = 2 * self.start_timeout_s
        horizon = time.time() - max(0.0, min_age_s)
        removed = 0
        # Our own files whose delete failed earlier: no age gate needed.
        for p in list(self._undeleted - self._booting):
            if self._unlink_quiet(p):
                removed += 1
            if not p.exists():
                self._undeleted.discard(p)
        for p in self.transient_files():
            if p in self._booting:
                continue
            try:
                if p.stat().st_mtime > horizon:
                    continue
            except OSError:
                continue
            if self._unlink_quiet(p):
                removed += 1
                self._undeleted.discard(p)
        if removed:
            logger.info("agent-plugin: swept %d stale transient agent file(s)", removed)
        return removed

    async def aclose(self) -> None:
        """Shutdown: finish background deletes, then one last attempt on our
        own files they could not remove (never another server's)."""
        if self._deleters:
            await asyncio.gather(*list(self._deleters), return_exceptions=True)
        for p in list(self._undeleted):
            if self._unlink_quiet(p) or not p.exists():
                self._undeleted.discard(p)

    # ── internals ────────────────────────────────────────────────────────────

    def _personalize(self, content: str, name: str) -> str:
        head = f"---\nname: {self.agent_name}\n"
        if content.startswith(head):
            return f"---\nname: {name}\n" + content[len(head):]
        return content

    def _delete(self, path: Path) -> None:
        """Delete a transient file now; on PermissionError (Windows: agy still
        holds it) retry in the background, then leave it to :meth:`sweep`."""
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            pass
        except OSError:
            self._undeleted.add(path)
            logger.warning("agent-plugin: could not delete %s; left to the sweep", path.name)
            return
        self._undeleted.add(path)
        try:
            task = asyncio.get_running_loop().create_task(self._delete_later(path))
        except RuntimeError:  # no loop (sync caller): the sweep handles it
            return
        self._deleters.add(task)
        task.add_done_callback(self._deleters.discard)

    async def _delete_later(self, path: Path) -> None:
        for _ in range(_DELETE_RETRIES):
            await asyncio.sleep(_DELETE_BACKOFF_S)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                continue
            except OSError:
                break
            self._undeleted.discard(path)
            return
        logger.info("agent-plugin: %s still locked; left to the sweep", path.name)

    @staticmethod
    def _unlink_quiet(p: Path) -> bool:
        try:
            p.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:
            logger.warning("agent-plugin: could not delete %s", p.name)
            return False

    def _ensure_plugin(self) -> None:
        if self._ensured and self.agents_dir.is_dir():
            return
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
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".agi-", suffix=".tmp")
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
