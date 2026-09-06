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
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import anyio

from app.agent_config import AgentPluginManager
from app.agy_session import STREAM_CLOSED, AgySession
from app.config import PromptTooLarge, Settings, check_prompt_size
from app.errors import OpenAIError
from app.events import Error, Init, TextDelta, ToolStep, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge, PendingCall, tool_signature
from app.mcp_config import McpConfigManager
from app.openai_models import ChatCompletionRequest, ChatMessage
from app.thinking import TurnThinking, default_brain_path
from app.translate import (
    build_prompt,
    materialize_messages,
    message_text,
    split_system,
    usage_from_turn,
)

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


@dataclass
class ReasoningChunk:
    """One whole thinking block (see app/thinking.py), rendered by the route
    as a ``delta.reasoning_content`` chunk / ``message.reasoning_content``."""

    text: str


TurnChunk = Union[
    TextChunk, ToolCallsChunk, DoneChunk, ErrorChunk, ToolBoundaryChunk, ReasoningChunk
]


# ── conversation ────────────────────────────────────────────────────────────


@dataclass
class Conversation:
    conv_id: str
    slot: str  # server name in the shared MCP config ("hermes", "hermes2", …)
    session: AgySession
    bridge: ConversationBridge
    model: str
    state: str = RUNNING
    thinking: Optional[TurnThinking] = None
    last_activity: float = field(default_factory=time.monotonic)
    workdir: Optional[Path] = None
    release_resources: Callable[[], None] | None = None
    _close_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class ExpiredContinuation(Exception):
    """A continuation whose tool_call_ids match no suspended conversation."""


class ConversationManager:
    def __init__(
        self,
        mcp: McpBridge,
        mcp_config: McpConfigManager,
        settings: Settings,
        agents: Optional[AgentPluginManager] = None,
    ) -> None:
        self.mcp = mcp
        self.mcp_config = mcp_config
        self.settings = settings
        self.agents = agents
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
        self._check_size(build_prompt(convo, system, inject_system=True))
        agent = None
        if self.settings.system_prompt_via_agent and self.agents is not None:
            # Tool turns need call_mcp_tool, which only `tools: "*"` grants
            # (see agent_config) — hence tool_turn_tools, not plain_chat_tools.
            agent = await self.agents.ensure_agent(
                system, tools_mode=self.settings.tool_turn_tools
            )
        if self.settings.image_input:
            convo = materialize_messages(convo, workdir)
        # No agent (mode off, or plugin dir unwritable) → automatic fallback:
        # the system message rides as an instruction preamble instead.
        content = build_prompt(convo, system, inject_system=agent is None)
        if not content.strip():
            raise OpenAIError("no user content in messages", status_code=400, param="messages")

        # Nudge the model toward its slot: the schema cache the CLI injects
        # names the server, but an explicit hint removes ambiguity. Built before
        # the slot is allocated so a size rejection leaves no side effects; the
        # slot name is a short token, so the cap is checked against the body.
        tool_names = ", ".join(t.function.name for t in (req.tools or []))
        try:
            check_prompt_size(content, self.settings)
        except PromptTooLarge as e:
            raise OpenAIError(
                f"prompt is {e.nbytes} UTF-8 bytes; the server cap is {e.cap} "
                "(AGI_MAX_PROMPT_BYTES). Shorten the conversation.",
                status_code=400, type="invalid_request_error", code="context_length_exceeded",
            )

        async with self._lock:
            conv_id = self._next_conv_id()
        bridge = ConversationBridge(
            conv_id, req.tools or [],
            max_pending_tool_calls=self.settings.max_pending_tool_calls,
        )
        slot = None
        session = None
        allocation = None
        try:
            self.mcp.register(bridge)
            # Finish slot allocation even if the caller is cancelled, so rollback
            # always knows which owned slot must be removed.
            allocation = asyncio.create_task(
                self.mcp_config.add_conversation(conv_id, self._mcp_url(conv_id))
            )
            slot = await asyncio.shield(allocation)
            hint = (
                f"\n\n[External functions are available on the MCP server "
                f"'{slot}' via the call_mcp_tool tool: {tool_names}. Use them when needed.]"
            )
            session = AgySession(
                agy_bin=self.settings.agy_bin, model=model, workdir=workdir,
                prompt=content + (hint if tool_names else ""), agent=agent,
                print_timeout_s=self.settings.request_timeout_s + self.settings.suspended_ttl_s,
                disable_slash_commands=self.settings.disable_slash_commands,
                timing_log=self.settings.timing_log, timing_label="tool",
                event_queue_size=self.settings.event_queue_size,
                stdin_timeout_s=self.settings.stdin_timeout_s,
            )
            await session.start()
            brain = self.settings.agy_brain_path or default_brain_path()
            conv = Conversation(conv_id=conv_id, slot=slot, session=session,
                                bridge=bridge, model=model, workdir=workdir.resolve(),
                                thinking=TurnThinking(brain))
            async with self._lock:
                self._conversations[conv_id] = conv
        except BaseException:
            async def rollback():
                nonlocal slot
                try:
                    if allocation is not None:
                        try:
                            slot = await allocation
                        except BaseException:
                            pass
                    bridge.fail_all("conversation creation failed")
                    self._conversations.pop(conv_id, None)
                    await self.mcp.close_conversation(conv_id)
                finally:
                    try:
                        if slot is not None:
                            await self.mcp_config.remove_conversation(slot)
                    finally:
                        if session is not None:
                            await session.aclose()
            try:
                await self._finish_cleanup(rollback())
            except BaseException:
                logger.exception("conv=%s rollback failed", conv_id)
            raise
        logger.info("conv=%s created (slot=%s, model=%s, %d tools)",
                    conv_id, slot, model, len(req.tools or []))
        return conv

    # ── resume (continuation) ─────────────────────────────────────────────—

    def _check_size(self, content: str) -> None:
        try:
            check_prompt_size(content, self.settings)
        except PromptTooLarge as exc:
            raise OpenAIError(str(exc), status_code=400, code="context_length_exceeded") from exc

    async def resume(
        self, req: ChatCompletionRequest, *, model: Optional[str] = None,
        workdir: Optional[Path] = None,
    ) -> Conversation:
        tool_msgs = self._trailing_tool_messages(req)
        results = [(m.tool_call_id, message_text(m)) for m in tool_msgs]
        self._check_size("\n".join(result for _, result in results))
        ids = [tid for tid, _ in results]
        if not ids or any(not tid for tid in ids) or len(set(ids)) != len(ids):
            raise ExpiredContinuation("continuation requires unique, nonempty tool_call_ids")

        async with self._lock:
            owners = {self._pending_index.get(tid) for tid in ids}
            if len(owners) != 1 or None in owners:
                raise ExpiredContinuation("tool results must belong to one suspended conversation")
            conv = self._conversations.get(next(iter(owners)))
            if conv is None or conv.state != SUSPENDED:
                raise ExpiredContinuation("tool results reference an expired or unknown conversation")
            indexed = {tid for tid, cid in self._pending_index.items() if cid == conv.conv_id}
            if set(ids) != indexed or set(ids) != set(conv.bridge.pending_ids):
                raise ExpiredContinuation("continuation must supply every pending tool result exactly once")
            if req.model is not None and (model if model is not None else req.model) != conv.model:
                raise ExpiredContinuation("continuation model does not match the suspended conversation")
            if req.workdir is not None:
                candidate = Path(workdir if workdir is not None else req.workdir).expanduser().resolve()
                if conv.workdir is None or candidate != conv.workdir:
                    raise ExpiredContinuation("continuation workdir does not match the suspended conversation")
            if req.tools is not None and tool_signature(req.tools) != conv.bridge.signature:
                raise ExpiredContinuation("continuation tools schema does not match the suspended conversation")
            # All validation and result preparation precedes the first mutation.
            conv.state = RUNNING
            conv.touch()
            for tid, result in results:
                self._pending_index.pop(tid)
                conv.bridge.resolve(tid, result)
        logger.info("conv=%s resumed (%d tool results)", conv.conv_id, len(results))
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
        deadline = asyncio.get_running_loop().time() + self.settings.request_timeout_s
        saw_text = False
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                ev = await asyncio.wait_for(self._next_event_or_call(conv, remaining), remaining)
                if conv.thinking is not None:
                    if isinstance(ev, Init):
                        conv.thinking.on_init(ev.conversation_id)
                    for block in conv.thinking.poll():
                        yield ReasoningChunk(block)
                if ev is None:
                    await self._close(conv)
                    yield ErrorChunk("upstream timeout", status_code=504)
                    return
                if isinstance(ev, PendingCall):
                    batch = [ev]
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    batch.extend(await asyncio.wait_for(conv.bridge.drain_incoming(grace_s=0.15), remaining))
                    batch = [pc for pc in batch if not pc.future.done()]
                    if not batch:
                        continue
                    async with asyncio.timeout_at(deadline):
                        async with self._lock:
                            at_capacity = sum(c.state == SUSPENDED for c in self._conversations.values()) >= self.settings.max_suspended_sessions
                            if not at_capacity:
                                for pc in batch:
                                    self._pending_index[pc.id] = conv.conv_id
                                conv.state = SUSPENDED
                    if at_capacity:
                        await self._close(conv)
                        yield ErrorChunk("suspended conversation capacity exceeded", status_code=503)
                        return
                    conv.touch()
                    logger.info("conv=%s suspended on %d tool call(s)", conv.conv_id, len(batch))
                    yield ToolCallsChunk(batch)
                    return  # park SUSPENDED; subprocess blocked in call_tool
                if ev is STREAM_CLOSED:
                    # Close BEFORE yielding: the consumer may break out of its
                    # async-for on a terminal chunk, and code after that yield
                    # would never run (GeneratorExit).
                    await self._close(conv)
                    yield ErrorChunk("upstream closed without TurnDone", status_code=502)
                    return
                if isinstance(ev, TextDelta):
                    saw_text = saw_text or bool(ev.text)
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
                    if not saw_text and ev.response_text:
                        yield TextChunk(ev.response_text)
                    yield DoneChunk("stop", usage)
                    return
                elif isinstance(ev, Error):
                    message = ev.message
                    await self._close(conv)
                    yield ErrorChunk(message, status_code=502)
                    return
                # Init / others: ignore.
        except TimeoutError:
            await self._close(conv)
            yield ErrorChunk("upstream timeout", status_code=504)
        except BaseException:
            if conv.state != SUSPENDED:
                await self._close(conv)
            raise

    async def _next_event_or_call(
        self, conv: Conversation, timeout: float
    ) -> Union[object, PendingCall, None]:
        """Race the stdout event queue against the bridge's incoming calls."""
        ev_task = asyncio.create_task(conv.session.next_event(timeout=timeout))
        call_task = asyncio.create_task(conv.bridge.next_incoming(timeout=timeout))
        try:
            done, pending = await asyncio.wait(
                {ev_task, call_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (ev_task, call_task):
                if not task.done():
                    task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(ev_task, call_task, return_exceptions=True)
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

    @staticmethod
    async def _finish_cleanup(awaitable):
        # AnyIO scopes alone do not shield direct asyncio.Task.cancel().
        with anyio.CancelScope(shield=True):
            task = asyncio.ensure_future(awaitable)
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.done():
                        break
            return task.result()

    async def _close(self, conv: Conversation) -> None:
        if conv._close_task is None:
            conv._close_task = asyncio.create_task(self._close_with_resources(conv))
        await self._finish_cleanup(conv._close_task)

    async def _close_with_resources(self, conv: Conversation) -> None:
        try:
            await self._close_impl(conv)
        finally:
            release, conv.release_resources = conv.release_resources, None
            if release is not None:
                release()

    async def _close_impl(self, conv: Conversation) -> None:
        if conv.state == CLOSED:
            return
        conv.state = CLOSED
        try:
            conv.bridge.fail_all("conversation closed")
        finally:
            try:
                async with self._lock:
                    self._conversations.pop(conv.conv_id, None)
                    stale = [tid for tid, cid in self._pending_index.items() if cid == conv.conv_id]
                    for tid in stale:
                        self._pending_index.pop(tid, None)
                await self.mcp.close_conversation(conv.conv_id)
            finally:
                try:
                    await self.mcp_config.remove_conversation(conv.slot)
                finally:
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
