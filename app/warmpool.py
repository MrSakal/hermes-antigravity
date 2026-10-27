"""Warm subprocess pool — pre-booted, idle ``agy`` procs ready to adopt.

A fresh conversation pays the CLI boot (exec → ``init`` ≈1–3 s on agy 1.4.8,
~3.5 s on 1.1.27) before agy even reads its first message. The pool pre-pays
it: a small number of procs are spawned ahead of time (no prompt, see
:meth:`~app.agy_session.AgySession.send`), so an adopting request only pays
the model latency. Cross-turn reuse (app.conversation) covers follow-up turns
of a live conversation; the pool covers the FIRST turn of a new one.

Ported from schmarta/claude-code-openai-server's ``warmpool.py``. agy fixes
``--model`` / ``--agent`` / ``--add-dir`` at spawn and reads the agent FILE
(system prompt, builtin allowlist, MCP server list) at boot, so a pooled proc
is reusable only by a request with an equal :class:`Signature`. Each entry:

* pre-mints its own ``conv_id`` and, for tool turns, registers a bridge
  carrying the target toolset under ``/mcp/<conv_id>``;
* boots with that URL baked into its agent file (``mcpServers`` entry
  ``hermes-<toolset sha8>``) — every pooled boot goes through
  :meth:`~app.agent_config.AgentPluginManager.spawn`, which holds the
  agent-file lock until agy's first stdout line;
* on adoption the request's tools are late-bound into that bridge before the
  first stdin message. Safe because agy contacts MCP servers lazily, at the
  first message — verified live on 1.4.8 (zero requests during a 12 s idle
  boot).

**Server naming.** agy caches MCP tool schemas per server NAME
(``~/.gemini/antigravity-cli/mcp/<name>/``). The pool therefore keeps the
content-addressed name ``hermes-<toolset sha8>`` and makes the toolset hash part
of the signature: a pooled tool proc is only adopted by a request with exactly
that toolset ("pool only for a previously-seen toolset"). The cache stays
correct by construction (one name ⇔ one schema set), nothing has to be cleaned
up per conversation, and hermes' toolset is stable across a session, so the
steady-state hit rate is unaffected. The first request of a new toolset is a
cold start that re-targets the pool.

**Targets.** Every fresh (non-reused) turn records its signature in a window
of the last 16 acquires. The pool keeps up to ``max_signatures``
(``AGI_WARM_POOL_SIGNATURES``, default 2) of the hottest signatures in that
window warm, splitting ``size`` (``AGI_WARM_POOL_SIZE``, default 2) between
them in rank order: signatures seen at least twice in the window, most recently
used first; a signature seen only once fills a FREE slot but never displaces a
warm one (hysteresis), so mixed traffic — e.g. hermes' main
loop interleaved with its side calls — keeps both warm instead of thrashing
the pool on every switch.

**Priority and failures.** Refills spawn with ``priority=False``: they take the
agent-file lock only while no request-driven spawn is queued. A boot that
fails (agy exits before its first line: logged out, bad model, rejected agent)
or a pooled proc that dies while idle puts the pool into exponential backoff
(1 s doubling to 60 s, reset on the next good boot) — it never respawns in a
tight loop.

``--print-timeout`` is disabled for persistent procs, so an idle pooled proc
does not expire on its own; ``max_idle_s`` recycles it anyway (stale auth /
model list). Each idle proc costs ≈175–200 MB RSS and leaves one (empty)
conversation DB in ``~/.gemini/antigravity-cli/conversations``. Set
``AGI_WARM_POOL_SIZE=0`` to disable. Pooled procs are NOT counted against
``max_sessions`` until adopted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.agent_config import AgentPluginManager, SpawnFailed
from app.agy_session import AgySession
from app.config import Settings
from app.mcp_bridge import ConversationBridge

logger = logging.getLogger("agi.warmpool")

# Recent acquires (signatures) the target ranking looks at.
_WINDOW = 16
# Refill backoff after a failed boot / a pooled proc dying while idle.
_BACKOFF_MIN_S = 1.0
_BACKOFF_MAX_S = 60.0


@dataclass(frozen=True)
class Signature:
    """Everything agy fixes at spawn — the identity a pooled or parked
    process must share with the request that adopts it.

    ``agent`` is the digest of the rendered agent TEMPLATE (the per-conversation
    MCP URL excluded; None = no ``--agent``) and ``tools_key`` the toolset hash
    ("" = no MCP server). ``extra`` carries prompt-side identity that is not in
    the agent file (a preamble-delivered system prompt). The remaining fields
    are what is needed to spawn and are excluded from equality."""

    model: str
    workdir: str
    agent: Optional[str]
    agent_md: Optional[str] = field(default=None, compare=False, repr=False)
    tools_key: str = ""
    extra: str = ""
    # conv_id -> agent file content (tool turns: carries /mcp/<conv_id>).
    render: Optional[Callable[[str], str]] = field(default=None, compare=False, repr=False)
    tools: tuple = field(default=(), compare=False, repr=False)
    server: Optional[str] = field(default=None, compare=False)

    def agent_for(self, conv_id: str) -> Optional[str]:
        if self.render is not None:
            return self.render(conv_id)
        return self.agent_md

    @property
    def key(self) -> str:
        """Stable string form (history-key namespace for cross-turn reuse)."""
        return json.dumps([self.model, self.workdir, self.agent, self.tools_key, self.extra])


@dataclass
class PooledEntry:
    conv_id: str
    session: AgySession
    bridge: Any  # ConversationBridge registered under /mcp/<conv_id>, or None
    signature: Signature


def new_conv_id(prefix: str, counter: int) -> str:
    # uuid4 entropy: the id is the only selector on the unauthenticated
    # /mcp/<conv_id> mount (defense in depth on top of the loopback gate).
    return f"{prefix}{counter}-{uuid.uuid4().hex}"


class WarmPool:
    def __init__(self, settings: Settings, size: int, *, max_idle_s: float = 600.0,
                 agents: Optional[AgentPluginManager] = None, mcp: Any = None,
                 mcp_url: Optional[Callable[[str], str]] = None,
                 max_signatures: Optional[int] = None) -> None:
        self.settings = settings
        self.agents = agents
        self.mcp = mcp
        self.mcp_url = mcp_url
        self.size = max(0, size)
        self.max_idle_s = max_idle_s
        if max_signatures is None:
            max_signatures = getattr(settings, "warm_pool_signatures", 1)
        self.max_signatures = max(1, max_signatures)
        self._entries: list[PooledEntry] = []
        self._lock = asyncio.Lock()
        # Ranked target signatures (hottest first) and the acquire window
        # they are derived from (see _retarget).
        self._targets: list[Signature] = []
        self._window: deque[Signature] = deque(maxlen=_WINDOW)
        self._refill_task: Optional[asyncio.Task[None]] = None
        # (signature, done-future) of the pooled boot in flight, if any.
        self._booting: Optional[tuple[Signature, asyncio.Future]] = None
        self._closed = False
        self._counter = 0
        self._backoff = 0.0
        # Teardowns of discarded entries (strong refs; they run to completion
        # even when the refill that started them is cancelled).
        self._teardowns: set[asyncio.Task] = set()
        self.hits = 0
        self.misses = 0
        self.tool_hits = 0
        self.spawn_failures = 0

    # ── lifecycle ─────────────────────────────────────────────────────────—

    @property
    def enabled(self) -> bool:
        return self.size > 0 and not self._closed

    @property
    def idle(self) -> int:
        return len(self._entries)

    @property
    def _target(self) -> Optional[Signature]:
        """Hottest target (single-signature view, metrics / tests)."""
        return self._targets[0] if self._targets else None

    async def stop(self) -> None:
        async with self._lock:
            self._closed = True
            entries, self._entries = self._entries, []
        if self._refill_task is not None:
            self._refill_task.cancel()
            await asyncio.gather(self._refill_task, return_exceptions=True)
        for entry in entries:
            await self._discard(entry)
        if self._teardowns:
            await asyncio.gather(*list(self._teardowns), return_exceptions=True)
        if entries:
            logger.info("warmpool: stopped (%d procs drained)", len(entries))

    async def _discard(self, entry: PooledEntry) -> None:
        """Tear ``entry`` down in its own task: a cancellation of the caller
        (the refill, cancelled by stop()) must not leave the process or its
        bridge behind."""
        task = asyncio.create_task(self._discard_impl(entry), name="agi-warmpool-discard")
        self._teardowns.add(task)
        task.add_done_callback(self._teardowns.discard)
        await asyncio.shield(task)

    async def _discard_impl(self, entry: PooledEntry) -> None:
        try:
            await entry.session.aclose(graceful=True)
        finally:
            if entry.bridge is not None and self.mcp is not None:
                entry.bridge.fail_all("pooled process discarded")
                await self.mcp.close_conversation(entry.conv_id)

    def _session(self, sig: Signature) -> AgySession:
        s = self.settings
        return AgySession(
            agy_bin=s.agy_bin, model=sig.model, workdir=sig.workdir, prompt=None,
            # --agent is set by AgentPluginManager.spawn.
            agent=None, print_timeout_s=0,
            disable_slash_commands=s.disable_slash_commands,
            timing_log=s.timing_log, timing_label="pool",
            event_queue_size=s.event_queue_size, stdin_timeout_s=s.stdin_timeout_s,
        )

    async def _spawn_entry(self, sig: Signature) -> PooledEntry:
        """Boot one pooled proc for ``sig``. Raises on any failure — a proc
        that died during boot is never returned (it would be dropped and
        respawned at once, forever)."""
        self._counter += 1
        conv_id = new_conv_id("pool", self._counter)
        bridge = None
        if sig.tools_key:
            if self.mcp is None or self.mcp_url is None:
                raise RuntimeError("tool-turn pooling needs the MCP bridge")
            # Registered WITH the target tools: should agy ever list them early
            # it must cache the right schemas under this server name.
            bridge = ConversationBridge(conv_id, list(sig.tools),
                                        max_pending_tool_calls=self.settings.max_pending_tool_calls)
            self.mcp.register(bridge)
        session = self._session(sig)
        try:
            content = sig.agent_for(conv_id)
            if content is not None and self.agents is not None:
                # Background spawn: yields the agent-file lock to requests.
                await self.agents.spawn(session, content, priority=False)
            else:
                await session.start()
                if not await session.wait_started(self.settings.agent_start_timeout_s):
                    raise RuntimeError("pooled agy did not start")
                if getattr(session, "boot_failed", False):
                    diag = await session.exit_diagnostics()
                    raise SpawnFailed(f"agy exited during startup ({diag})")
        except BaseException:
            try:
                await session.aclose()
            finally:
                if bridge is not None:
                    await self.mcp.close_conversation(conv_id)
            raise
        return PooledEntry(conv_id, session, bridge, sig)

    # ── targets ───────────────────────────────────────────────────────────—

    def _retarget(self) -> None:
        """Pick up to ``min(max_signatures, size)`` targets from the recent
        acquire window: signatures seen at least twice in it ("hot"), most
        recently used first (LRU among the hot ones). Free slots are filled
        with seen-once signatures, current targets first — so a single stray
        miss never displaces a warm signature (no thrash), while a signature
        that recurs takes its place at its second sighting."""
        counts = Counter(self._window)
        last = {sig: i for i, sig in enumerate(self._window)}
        k = max(1, min(self.max_signatures, self.size))
        new = sorted((sig for sig in counts if counts[sig] >= 2),
                     key=lambda sig: last[sig], reverse=True)[:k]
        if len(new) < k:
            incumbents = [t for t in self._targets if t in counts and t not in new]
            others = sorted((sig for sig in counts if sig not in new and sig not in incumbents),
                            key=lambda sig: last[sig], reverse=True)
            new += (incumbents + others)[:k - len(new)]
        if new != self._targets:
            logger.info("warmpool: targets -> %s",
                        [(t.model, bool(t.tools_key), counts[t]) for t in new])
        self._targets = new

    def _quota(self) -> dict[Signature, int]:
        """Per-target share of ``size``, dealt round-robin in rank order."""
        quota: dict[Signature, int] = {t: 0 for t in self._targets}
        for i in range(self.size if self._targets else 0):
            quota[self._targets[i % len(self._targets)]] += 1
        return quota

    # ── refill ────────────────────────────────────────────────────────────—

    def _kick_refill(self) -> None:
        if self.enabled and (self._refill_task is None or self._refill_task.done()):
            self._refill_task = asyncio.create_task(self._refill(), name="agi-warmpool")

    def _fail(self, why: str) -> float:
        """Advance the backoff one step and log it (once per step)."""
        self.spawn_failures += 1
        self._backoff = min(_BACKOFF_MAX_S, max(_BACKOFF_MIN_S, self._backoff * 2))
        logger.warning("warmpool: %s; next boot in %.0fs", why, self._backoff)
        return self._backoff

    async def _refill(self) -> None:
        """Top every target up to its quota, discarding dead, stale-signature
        and over-aged entries; back off exponentially on boot failures."""
        while self.enabled:
            async with self._lock:
                quota = self._quota()
                keep, drop, died = [], [], 0
                for entry in self._entries:
                    alive = entry.session.running
                    if (alive and quota.get(entry.signature, 0) > 0
                            and entry.session.idle_s < self.max_idle_s):
                        quota[entry.signature] -= 1
                        keep.append(entry)
                    else:
                        died += not alive
                        drop.append(entry)
                self._entries = keep
                # Most-needed target first (stable sort keeps rank order on ties).
                need = sorted((t for t in self._targets if quota[t] > 0), key=lambda t: -quota[t])
                target = need[0] if need else None
            for entry in drop:
                await self._discard(entry)
            if died:
                # Booted fine, then died idle (agy exiting right after init):
                # respawning at once would loop just like a failed boot.
                await asyncio.sleep(self._fail(f"{died} pooled proc(s) died while idle"))
                continue
            if target is None:
                return
            self._booting = (target, asyncio.get_running_loop().create_future())
            try:
                entry = await self._spawn_entry(target)
            except Exception as e:
                delay = self._fail(f"pooled spawn failed: {type(e).__name__}: {e}")
                await asyncio.sleep(delay)
                continue
            finally:
                _, booted = self._booting
                self._booting = None
                if not booted.done():
                    booted.set_result(None)
            self._backoff = 0.0
            # Filed synchronously — no await between the boot returning and
            # the append, so a cancellation cannot lose the proc. Every _lock
            # section is await-free, so this is consistent with them.
            if not self._closed and target in self._targets:
                self._entries.append(entry)
                continue
            await self._discard(entry)
            if self._closed:
                return

    # ── acquire ───────────────────────────────────────────────────────────—

    async def acquire(self, sig: Signature) -> Optional[PooledEntry]:
        """Pop a live idle entry for ``sig``, or ``None`` to cold-spawn.

        Records ``sig`` as recently used (which may re-rank the targets) and
        kicks a background refill either way."""
        if not self.enabled:
            return None
        popped = await self._pop(sig)
        booting = self._booting
        if popped is None and booting is not None and booting[0] == sig:
            # A pooled boot for exactly this signature is already under way:
            # it started earlier than a cold spawn would, so waiting for it is
            # never slower (bounded by the boot timeout either way).
            try:
                async with asyncio.timeout(self.settings.agent_start_timeout_s):
                    await asyncio.shield(booting[1])
                await asyncio.sleep(0)  # let the refill file the entry
            except TimeoutError:
                pass
            popped = await self._take(sig)
        self._kick_refill()
        if popped is None:
            self.misses += 1
            return None
        self.hits += 1
        if sig.tools_key:
            self.tool_hits += 1
        logger.info("warmpool: hit (model=%s, tools=%s, idle %.1fs)", sig.model,
                    bool(sig.tools_key), popped.session.idle_s)
        return popped

    async def _pop(self, sig: Signature) -> Optional[PooledEntry]:
        async with self._lock:
            self._window.append(sig)
            self._retarget()
        return await self._take(sig)

    async def _take(self, sig: Signature) -> Optional[PooledEntry]:
        async with self._lock:
            for i, entry in enumerate(self._entries):
                if (entry.signature == sig and entry.session.running
                        and entry.session.idle_s < self.max_idle_s):
                    return self._entries.pop(i)
        return None

    async def gc(self) -> None:
        """Periodic hook: drop dead / over-aged procs and refill."""
        if self.enabled and self._targets:
            self._kick_refill()
