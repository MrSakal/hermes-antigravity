"""Warm subprocess pool — pre-spawned, idle ``agy`` procs ready to adopt.

Every fresh autonomous turn pays the CLI boot: measured on this host against
agy 1.1.27, the process needs ~3.5 s from exec to its ``init`` event before it
drains stdin, then answers a trivial prompt ~1.5 s later. The boot is the bulk
of the ~5.5 s TTFT p50 and is independent of the prompt. The pool pre-pays it:
a small number of procs are spawned ahead of time (prompt deferred, see
:meth:`~app.agy_session.AgySession.send`), so an adopting request only pays the
model latency.

Ported from schmarta/claude-code-openai-server's ``warmpool.py`` and narrowed to
what agy can do:

* agy fixes ``--model`` / ``--agent`` / ``--add-dir`` at spawn, so a pooled proc
  is only reusable by a request with the same :class:`Signature`. The pool holds
  one signature at a time and re-targets to live traffic.
* Only **autonomous** turns are pooled. Tool turns need a per-conversation MCP
  slot in the shared config before spawn, and conversation-cache resumes need
  ``--conversation <id>`` — both are spawn-time argv, so neither can adopt.
* ``--print-timeout`` starts counting at the first prompt, not at boot
  (verified: an idle proc outlived a 15 s timeout), so an idle pooled proc does
  not expire on its own. ``max_idle_s`` recycles it anyway: agy creates the
  conversation DB at boot and a very old idle proc is a stale-auth / stale-model
  risk, not a saving.

Each idle proc is ~175 MB RSS (measured) and leaves one conversation DB in
``~/.gemini/antigravity-cli/conversations`` whether or not it is ever used —
so ``AGI_WARM_POOL_SIZE`` defaults to 0 and 1–2 is the sane range for one user.
Pooled procs are counted against ``max_sessions`` only once adopted.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.agy_session import AgySession
from app.config import Settings

logger = logging.getLogger("agi.warmpool")


@dataclass(frozen=True)
class Signature:
    """Everything agy fixes at spawn for an autonomous turn."""

    model: str
    workdir: str
    agent: Optional[str]


class WarmPool:
    def __init__(self, settings: Settings, size: int, *, max_idle_s: float = 600.0) -> None:
        self.settings = settings
        self.size = max(0, size)
        self.max_idle_s = max_idle_s
        self._entries: list[tuple[Signature, AgySession]] = []
        self._lock = asyncio.Lock()
        self._target: Optional[Signature] = None
        self._refill_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self.hits = 0
        self.misses = 0

    # ── lifecycle ─────────────────────────────────────────────────────────—

    @property
    def enabled(self) -> bool:
        return self.size > 0 and not self._closed

    @property
    def idle(self) -> int:
        return len(self._entries)

    async def stop(self) -> None:
        async with self._lock:
            self._closed = True
            entries, self._entries = self._entries, []
        if self._refill_task is not None:
            self._refill_task.cancel()
            await asyncio.gather(self._refill_task, return_exceptions=True)
        for _, session in entries:
            await session.aclose()
        if entries:
            logger.info("warmpool: stopped (%d procs drained)", len(entries))

    def _spawn(self, sig: Signature) -> AgySession:
        s = self.settings
        return AgySession(
            agy_bin=s.agy_bin, model=sig.model, workdir=sig.workdir, prompt=None,
            agent=sig.agent, print_timeout_s=s.request_timeout_s,
            disable_slash_commands=s.disable_slash_commands,
            timing_log=s.timing_log, timing_label="pool",
            event_queue_size=s.event_queue_size, stdin_timeout_s=s.stdin_timeout_s,
        )

    # ── refill ────────────────────────────────────────────────────────────—

    def _kick_refill(self) -> None:
        if self.enabled and (self._refill_task is None or self._refill_task.done()):
            self._refill_task = asyncio.create_task(self._refill(), name="agi-warmpool")

    async def _refill(self) -> None:
        """Top the pool up to ``size`` procs of the current target signature,
        discarding dead, stale-signature and over-aged entries."""
        while self.enabled:
            async with self._lock:
                target = self._target
                keep, drop = [], []
                for sig, session in self._entries:
                    fresh = session.running and sig == target and session.idle_s < self.max_idle_s
                    (keep if fresh else drop).append((sig, session))
                self._entries = keep
                need = self.size - len(keep) if target is not None else 0
            for _, session in drop:
                await session.aclose()
            if need <= 0:
                return
            session = self._spawn(target)
            try:
                await session.start()
            except Exception:
                logger.exception("warmpool: spawn failed; aborting refill")
                return
            async with self._lock:
                if self._closed or target != self._target:
                    await session.aclose()
                    if self._closed:
                        return
                else:
                    self._entries.append((target, session))

    # ── acquire ───────────────────────────────────────────────────────────—

    async def acquire(self, sig: Signature) -> Optional[AgySession]:
        """Pop a live idle proc for ``sig``, or ``None`` to cold-spawn.

        Re-targets the pool to ``sig`` and kicks a background refill either way,
        so the first request of a new signature is cold and steady-state traffic
        of that signature is warm."""
        if not self.enabled:
            return None
        popped: Optional[AgySession] = None
        async with self._lock:
            self._target = sig
            for i, (s, session) in enumerate(self._entries):
                if s == sig and session.running and session.idle_s < self.max_idle_s:
                    _, popped = self._entries.pop(i)
                    break
        self._kick_refill()
        if popped is None:
            self.misses += 1
            return None
        self.hits += 1
        logger.info("warmpool: hit (model=%s, idle %.1fs)", sig.model, popped.idle_s)
        return popped

    async def gc(self) -> None:
        """Periodic hook: drop over-aged procs and refill (called by the app GC loop)."""
        if self.enabled and self._entries:
            self._kick_refill()
