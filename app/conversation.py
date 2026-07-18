"""ConversationManager — reconciles stateless OpenAI requests with stateful,
suspendable agy turns for the tool-passthrough path.

A *conversation* owns one live ``agy`` subprocess (the current turn), its
:class:`~app.mcp_bridge.ConversationBridge`, and a server *slot* — its own entry
(``hermes``, ``hermes2``, …) in the shared Antigravity MCP config pointing at
``/mcp/<conv_id>``. It moves through:

    RUNNING ── tool calls ──▶ SUSPENDED ──(next request resolves results)──▶ RUNNING
            └────────────── clean result ──────────────▶ CLOSED

Request classification (same as hermes-claude):

* **Continuation** — the message list ends with one or more ``tool`` results,
  matched to a suspended conversation by ``tool_call_id``. Resolving the pending
  Futures lets the agy subprocess — still blocked inside our MCP ``call_tool``
  — resume; no new turn is spawned.
* **Fresh turn** — otherwise: allocate conv_id + slot, register the bridge, add
  the config entry, spawn ``agy -p`` with the folded history.

agy difference vs claude: tool calls surface on our MCP endpoint (the bridge
queue), not as a distinct stdout event we must pair with the queue — so
``run_turn`` waits on the stdout event queue and the bridge's incoming queue
simultaneously and treats a PendingCall arrival as the suspend trigger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from app.agy_session import STREAM_CLOSED, AgySession
from app.config import Settings
from app.events import Error, TextDelta, ToolStep, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge, PendingCall
from app.mcp_config import McpConfigManager
from app.openai_models import ChatCompletionRequest, ChatMessage
from app.translate import build_prompt, message_text, split_system, usage_from_turn

logger = logging.getLogger("agi.conv")

RUNNING = "running"
SUSPENDED = "suspended"
CLOSED = "closed"


# ── turn-loop chunks (rendered to SSE / JSON by the chat route) ─────────────—


@dataclass
class TextChunk:
    text: str


@dataclass
class ToolCallsChunk:
    calls: list[PendingCall]


@dataclass
class DoneChunk:
    finish_reason: str
    usage: dict


@dataclass
class ErrorChunk:
    message: str
    status_code: int = 502


@dataclass
class ToolBoundaryChunk:
    """Marks an internal (builtin) tool use between two assistant text
    segments. Carries no text; the route uses it to insert a blank-line seam."""


TurnChunk = Union[TextChunk, ToolCallsChunk, DoneChunk, ErrorChunk, ToolBoundaryChunk]


# ── conversation ────────────────────────────────────────────────────────────


@dataclass
class Conversation:
    conv_id: str
    slot: str  # server name in the shared MCP config ("hermes", "hermes2", …)
    session: AgySession
    bridge: ConversationBridge
    model: str
    state: str = RUNNING
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class ExpiredContinuation(Exception):
    """A continuation whose tool_call_ids match no suspended conversation."""


class ConversationManager:
    def __init__(self, mcp: McpBridge, mcp_config: McpConfigManager, settings: Settings) -> None:
        self.mcp = mcp
        self.mcp_config = mcp_config
        self.settings = settings
        self._conversations: dict[str, Conversation] = {}
        self._pending_index: dict[str, str] = {}  # tool_call_id -> conv_id
        self._lock = asyncio.Lock()
        self._counter = 0

    # ── classification ───────────────────────────────────────────────────—

    @staticmethod
    def is_continuation(req: ChatCompletionRequest) -> bool:
        for m in reversed(req.messages):
            if m.role == "tool":
                return True
            return False
        return False

    @staticmethod
    def _trailing_tool_messages(req: ChatCompletionRequest) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        for m in reversed(req.messages):
            if m.role == "tool":
                out.append(m)
            else:
                break
        out.reverse()
        return out

    # ── creation (fresh turn) ─────────────────────────────────────────────—

    def _next_conv_id(self) -> str:
        self._counter += 1
        return f"conv{self._counter}-{int(time.time())}"

    def _mcp_url(self, conv_id: str) -> str:
        return f"http://127.0.0.1:{self.settings.port}/mcp/{conv_id}"

    async def create(
        self,
        req: ChatCompletionRequest,
        *,
        model: str,
        workdir: Path,
    ) -> Conversation:
        convo, system = split_system(req.messages)
        content = build_prompt(
            convo, system, inject_system=self.settings.system_prompt_injection
        )

        async with self._lock:
            conv_id = self._next_conv_id()
        bridge = ConversationBridge(conv_id, req.tools or [])
        self.mcp.register(bridge)
        slot = await self.mcp_config.add_conversation(conv_id, self._mcp_url(conv_id))

        # Nudge the model toward its slot: the schema cache the CLI injects
        # names the server, but an explicit hint removes ambiguity.
        tool_names = ", ".join(t.function.name for t in (req.tools or []))
        hint = (
            f"\n\n[External functions are available on the MCP server "
            f"'{slot}' via the call_mcp_tool tool: {tool_names}. Use them when needed.]"
        )
        prompt = content + (hint if tool_names else "")

        session = AgySession(
            agy_bin=self.settings.agy_bin,
            model=model,
            workdir=workdir,
            prompt=prompt,
            # The proc must survive suspensions (blocked in our MCP call while
            # hermes runs the tool), so its own ceiling exceeds one request's.
            print_timeout_s=self.settings.request_timeout_s + self.settings.suspended_ttl_s,
            timing_log=self.settings.timing_log,
            timing_label="tool",
        )
        try:
            await session.start()
        except Exception:
            self.mcp.unregister(conv_id)
            await self.mcp_config.remove_conversation(slot)
            raise
        conv = Conversation(conv_id=conv_id, slot=slot, session=session,
                            bridge=bridge, model=model)
        async with self._lock:
            self._conversations[conv_id] = conv
        logger.info("conv=%s created (slot=%s, model=%s, %d tools)",
                    conv_id, slot, model, len(req.tools or []))
        return conv

    # ── resume (continuation) ─────────────────────────────────────────────—

    async def resume(self, req: ChatCompletionRequest) -> Conversation:
        tool_msgs = self._trailing_tool_messages(req)
        if not tool_msgs:
            raise ExpiredContinuation("no trailing tool results in continuation request")
        ids = [m.tool_call_id for m in tool_msgs if m.tool_call_id]

        # Locate AND claim atomically (SUSPENDED→RUNNING under one lock) so a
        # concurrent duplicate continuation is rejected instead of double-driving
        # the one agy subprocess.
        async with self._lock:
            conv_id = next((self._pending_index[i] for i in ids if i in self._pending_index), None)
            conv = self._conversations.get(conv_id) if conv_id else None
            if conv is None or conv.state != SUSPENDED:
                raise ExpiredContinuation(
                    "tool results reference an expired or unknown conversation; retry the turn"
                )
            conv.state = RUNNING
            conv.touch()
            for tid in ids:
                self._pending_index.pop(tid, None)

        outstanding = set(conv.bridge.pending_ids)
        answered: set[str] = set()
        resolved = 0
        for m in tool_msgs:
            if not m.tool_call_id:
                continue
            answered.add(m.tool_call_id)
            if conv.bridge.resolve(m.tool_call_id, message_text(m)):
                resolved += 1

        missing = outstanding - answered
        if missing:
            logger.warning("conv=%s partial continuation: %d of %d tool calls unanswered",
                           conv.conv_id, len(missing), len(outstanding))
            conv.bridge.fail_all("continuation did not supply all tool results")

        logger.info("conv=%s resumed (%d/%d tool results)", conv.conv_id, resolved, len(tool_msgs))
        return conv

    # ── turn loop ─────────────────────────────────────────────────────────—

    async def run_turn(self, conv: Conversation) -> AsyncIterator[TurnChunk]:
        """Yield chunks until the turn suspends on tool calls or completes.

        Waits on BOTH the subprocess stdout events and the bridge's incoming
        PendingCall queue: with agy, a hermes tool call manifests as an HTTP
        ``call_tool`` on our bridge (the stdout ``tool`` step for it may arrive
        before, after, or not at all while blocked), so the queue is the
        authoritative suspend trigger.
        """
        timeout = self.settings.request_timeout_s
        try:
            while True:
                ev = await self._next_event_or_call(conv, timeout)
                if ev is None:
                    await self._close(conv)
                    yield ErrorChunk("upstream timeout", status_code=504)
                    return
                if isinstance(ev, PendingCall):
                    batch = [ev]
                    batch.extend(await conv.bridge.drain_incoming(grace_s=0.15))
                    async with self._lock:
                        for pc in batch:
                            self._pending_index[pc.id] = conv.conv_id
                        conv.state = SUSPENDED
                    conv.touch()
                    logger.info("conv=%s suspended on %d tool call(s)", conv.conv_id, len(batch))
                    yield ToolCallsChunk(batch)
                    return  # park SUSPENDED; subprocess blocked in call_tool
                if ev is STREAM_CLOSED:
                    # Close BEFORE yielding: the consumer may break out of its
                    # async-for on a terminal chunk, and code after that yield
                    # would never run (GeneratorExit).
                    await self._close(conv)
                    yield DoneChunk("stop", {})
                    return
                if isinstance(ev, TextDelta):
                    yield TextChunk(ev.text)
                elif isinstance(ev, ToolStep):
                    if ev.matches_server(conv.slot):
                        # Companion stdout event of a bridge call; the queue
                        # handles suspension — nothing to surface.
                        continue
                    if ev.state == "ACTIVE":
                        logger.debug("conv=%s internal tool: %s", conv.conv_id, ev.tool_name)
                        yield ToolBoundaryChunk()
                    continue
                elif isinstance(ev, TurnDone):
                    usage = usage_from_turn(ev)
                    await self._close(conv)
                    yield DoneChunk("stop", usage)
                    return
                elif isinstance(ev, Error):
                    message = ev.message
                    await self._close(conv)
                    yield ErrorChunk(message, status_code=502)
                    return
                # Init / others: ignore.
        except asyncio.CancelledError:
            if conv.state != SUSPENDED:
                await self._close(conv)
            raise

    async def _next_event_or_call(
        self, conv: Conversation, timeout: float
    ) -> Union[object, PendingCall, None]:
        """Race the stdout event queue against the bridge's incoming calls."""
        ev_task = asyncio.create_task(conv.session.next_event(timeout=timeout))
        call_task = asyncio.create_task(conv.bridge.next_incoming(timeout=timeout))
        done, pending = await asyncio.wait(
            {ev_task, call_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        # Prefer a PendingCall when both completed together.
        if call_task in done:
            call = call_task.result()
            if call is not None:
                # Don't lose a simultaneously-completed stdout event.
                if ev_task in done:
                    ev = ev_task.result()
                    if ev is not None:
                        conv.session.push_back(ev)
                return call
        if ev_task in done:
            return ev_task.result()
        return None

    # ── teardown ──────────────────────────────────────────────────────────—

    async def _close(self, conv: Conversation) -> None:
        if conv.state == CLOSED:
            return
        conv.state = CLOSED
        conv.bridge.fail_all("conversation closed")
        async with self._lock:
            self._conversations.pop(conv.conv_id, None)
            stale = [tid for tid, cid in self._pending_index.items() if cid == conv.conv_id]
            for tid in stale:
                self._pending_index.pop(tid, None)
        self.mcp.unregister(conv.conv_id)
        await self.mcp_config.remove_conversation(conv.slot)
        await conv.session.aclose()
        logger.info("conv=%s closed", conv.conv_id)

    async def close_all(self) -> None:
        for conv in list(self._conversations.values()):
            await self._close(conv)

    # ── garbage collection ────────────────────────────────────────────────—

    async def gc_once(self) -> int:
        """Reap conversations whose TTL elapsed. Returns the count closed."""
        now = time.monotonic()
        susp_ttl = self.settings.suspended_ttl_s
        idle_ttl = self.settings.idle_session_ttl_s
        victims: list[Conversation] = []
        async with self._lock:
            for conv in list(self._conversations.values()):
                age = now - conv.last_activity
                if conv.state == SUSPENDED and age > susp_ttl:
                    victims.append(conv)
                elif age > idle_ttl:
                    victims.append(conv)
        for conv in victims:
            logger.info("conv=%s GC (state=%s)", conv.conv_id, conv.state)
            await self._close(conv)
        return len(victims)

    async def gc_loop(self) -> None:
        interval = max(1, self.settings.gc_interval_s)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.gc_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("GC loop iteration failed")
