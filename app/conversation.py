"""ConversationManager — maps stateless OpenAI requests onto PERSISTENT agy
processes: one session pool for every chat request, with or without tools.

A *conversation* owns one live ``agy`` stdin-mode process (see
:mod:`app.agy_session`) and its :class:`~app.mcp_bridge.ConversationBridge`.
When the request has tools, its MCP server is declared in the managed agent
file the process boots with (``inheritMcp: false`` + one ``mcpServers`` entry
``hermes-<toolset sha8>`` → ``/mcp/<conv_id>``, see :mod:`app.agent_config`);
the shared ``~/.gemini/config/mcp_config.json`` is never written. States:

    RUNNING ── hermes tool calls ──▶ SUSPENDED ──(tool results)──▶ RUNNING
       │
       ├── final text ──▶ IDLE ──(next user message matches)──▶ RUNNING
       └── error / timeout / disconnect / process death ──▶ CLOSED

Request classification:

* **Continuation** — the message list ends with ``tool`` results, matched to a
  SUSPENDED conversation by ``tool_call_id``. Resolving the pending Futures
  lets the agy process — still blocked inside our MCP ``call_tool`` — resume.
  A dead suspended process is a plain expiry (rebuildable from the transcript
  the continuation carries); a live conversation that is not waiting on these
  ids, or ids whose results were already delivered (a bounded memory of
  recently completed ``tool_call_id``s), is a :class:`DuplicateContinuation`
  — 409, never rebuilt, because a rebuild re-runs the turn and its effects.
* **New turn** — otherwise. Cross-turn reuse: after a turn ends with final
  text its process is parked IDLE under ``key = H(spawn signature +
  normalized history incl. the assistant text exactly as returned to the
  client)``. A request whose ``messages[:-1]`` hash to an IDLE key adopts that
  process and sends ONLY the newest user message. Anything else starts
  fresh — adopting a pre-booted warm-pool process when one matches — with the
  whole history folded into the first turn. A miss is always safe; it only
  costs a boot.

There is no resume of an exited process: once a process is gone (idle TTL,
crash, eviction, shutdown) nothing here references its agy conversation, so
the housekeeper deletes its state at once (see :mod:`app.housekeeping`). The
client resends the whole history every request, so a follow-up simply takes
the fresh path — which, unlike a ``--conversation`` resume (a boot argument),
can adopt a warm-pool process.

The spawn signature is (model, workdir, agent template digest, toolset hash,
preamble-system hash): a process may only serve a request that would have
spawned an identical one, because model, agent body and MCP server are fixed
per process. agy has no per-turn cancel, so a turn abandoned mid-flight
(client disconnect, request timeout) kills its process — a process whose
history diverged from what the client saw is never parked for reuse.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import anyio

from app.agent_config import (
    AgentPluginManager,
    AgentUnavailable,
    McpServerSpec,
    SpawnFailed,
    agent_digest,
    render_agent,
    server_name_for,
    toolset_hash,
)
from app.agy_session import STREAM_CLOSED, AgySession
from app.config import PromptTooLarge, Settings, check_prompt_size
from app.errors import OpenAIError
from app.events import Error, Init, TextDelta, ToolStep, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge, PendingCall, tool_signature
from app.openai_models import ChatCompletionRequest, ChatMessage
from app.textfilter import OutputFilter
from app.thinking import TurnThinking, default_brain_path
from app.translate import (
    build_prompt,
    materialize_messages,
    message_text,
    message_text_with_images,
    split_system,
    usage_from_turn,
)
from app.warmpool import Signature, WarmPool, new_conv_id

logger = logging.getLogger("agi.conv")

RUNNING = "running"
SUSPENDED = "suspended"
IDLE = "idle"  # turn complete, process parked alive for cross-turn reuse
CLOSING = "closing"  # detached from every index, teardown pending
CLOSED = "closed"

# A RUNNING conversation whose turn loop never started (the response was never
# consumed — e.g. the client vanished before the first byte) is reaped by GC
# after this long instead of the full request timeout.
_UNDRIVEN_TTL_S = 30.0

# A conversation detached for teardown (CLOSING) whose teardown never started
# — its owner was cancelled first — is reaped by GC after this long.
_CLOSING_GRACE_S = 10.0

# Parallel tool calls of one assistant step arrive within this window.
_BATCH_GRACE_S = 0.15

# The thinking transcript is polled (a stat() per poll) at most this often
# while text streams; turn boundaries always poll.
_THINKING_POLL_S = 0.25


# ── turn-loop chunks (rendered to SSE / JSON by the chat route) ─────────────—


@dataclass
class TextChunk:
    """Assistant text, already post-processed (segment seams, table
    flattening) — exactly what the client receives."""

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
    """Legacy marker (seams are now applied inside the turn loop)."""


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
    slot: Optional[str]  # MCP server name in the agent file; None = no tools
    session: Any  # AgySession (or a test double)
    bridge: ConversationBridge
    model: str
    state: str = RUNNING
    thinking: Optional[TurnThinking] = None
    last_activity: float = field(default_factory=time.monotonic)
    workdir: Optional[Path] = None
    release_resources: Callable[[], None] | None = None
    signature: str = ""
    # The message list of the request currently being served — with the final
    # assistant text it becomes the history key when the turn parks IDLE.
    current_msgs: list[ChatMessage] = field(default_factory=list)
    reuse_key: Optional[str] = None
    # Set when run_turn starts consuming the current turn (see _UNDRIVEN_TTL_S).
    driven: bool = False
    # Bumped on every claim (fresh / reuse / continuation): request-scoped cleanup
    # only touches the turn it started, never a later one on the same process.
    turn_token: int = 0
    # Monotonic time it was detached for teardown (state CLOSING).
    closing_since: Optional[float] = None
    _close_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # One event stream for the turn loop: bridge calls join stdout events.
        inject = getattr(self.session, "inject", None)
        if inject is not None:
            self.bridge.attach(inject)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class ExpiredContinuation(Exception):
    """A continuation whose tool_call_ids match no live suspended conversation."""


class DuplicateContinuation(ExpiredContinuation):
    """A second copy of a continuation: its conversation is alive but not
    waiting on these ids, or their results were already delivered. Never
    rebuilt — the first copy is (or was) answering it, and a rebuild would
    re-run the turn and its side effects. Still renders as 409."""


# ── history keys ────────────────────────────────────────────────────────────


def _norm_message(m: ChatMessage) -> list[Any]:
    """One message, normalized for history matching.

    User/tool content is compared verbatim; list content (images included) is
    serialized whole, so two histories differing only in an attached image
    never collide. Assistant text is compared whitespace-stripped (clients
    commonly trim it) and tool calls by id + name only (argument JSON gets
    re-serialized by clients)."""
    if m.role == "assistant":
        calls = [[tc.id or "", tc.function.name or ""] for tc in (m.tool_calls or [])]
        return ["assistant", message_text(m).strip(), calls]
    c = m.content
    content = json.dumps(c, sort_keys=True, ensure_ascii=False) if isinstance(c, list) else (c or "")
    return [m.role, content, m.tool_call_id or "", m.name or ""]


def history_key(signature: str, messages: list[ChatMessage]) -> str:
    """Reuse key: spawn signature + normalized non-system history."""
    norm = [_norm_message(m) for m in messages if m.role != "system"]
    payload = json.dumps([signature, norm], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def _has_images(messages: list[ChatMessage]) -> bool:
    return any(isinstance(m.content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in m.content)
        for m in messages)


# ── manager ─────────────────────────────────────────────────────────────────


class ConversationManager:
    def __init__(
        self,
        mcp: McpBridge,
        settings: Settings,
        agents: Optional[AgentPluginManager] = None,
        pool: Optional[WarmPool] = None,
        housekeeper: Any = None,
    ) -> None:
        self.mcp = mcp
        self.settings = settings
        self.agents = agents
        self.pool = pool
        # app.housekeeping.Housekeeper (or None): deletes an agy conversation
        # once nothing here references it any more (see referenced_agy_ids).
        self.housekeeper = housekeeper
        if housekeeper is not None:
            housekeeper.references = self.referenced_agy_ids
            housekeeper.servers_in_use = self.servers_in_use
        root = getattr(settings, "agy_state_root", None)
        self._brain = settings.agy_brain_path or (Path(root).expanduser() / "brain" if root
                                                  else default_brain_path())
        self._conversations: dict[str, Conversation] = {}
        self._pending_index: dict[str, str] = {}  # tool_call_id -> conv_id
        self._idle: dict[str, str] = {}  # history key -> conv_id (IDLE convs)
        # tool_call_id -> monotonic time its result was delivered.
        self._completed: "OrderedDict[str, float]" = OrderedDict()
        self._lock = asyncio.Lock()
        self._counter = 0
        self._tokens = 0
        # Teardowns detached from a cancelled caller (strong refs).
        self._background: set[asyncio.Task] = set()
        # reuse_hits / reuse_misses / pool_adopted / duplicates_rejected
        self.stats: Counter = Counter()

    # ── classification ───────────────────────────────────────────────────—

    # ── ownership (housekeeping) ──────────────────────────────────────────—

    def referenced_agy_ids(self) -> set[str]:
        """agy conversations that may still be used: those of live / parked /
        closing processes. Everything else of ours — an exited process'
        conversation included — is deleted by the housekeeper."""
        refs: set[str] = set()
        for conv in self._conversations.values():
            agy_id = getattr(conv.session, "conversation_id", None)
            if agy_id:
                refs.add(agy_id)
        return refs

    def servers_in_use(self) -> set[str]:
        names = {c.slot for c in self._conversations.values() if c.slot}
        if self.pool is not None:
            names.update(e.signature.server for e in self.pool._entries if e.signature.server)
        return names

    def _poke(self) -> None:
        if self.housekeeper is not None:
            self.housekeeper.poke()

    @staticmethod
    def is_continuation(req: ChatCompletionRequest) -> bool:
        return bool(req.messages) and req.messages[-1].role == "tool"

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

    # ── ids / urls ────────────────────────────────────────────────────────—

    def _next_conv_id(self) -> str:
        self._counter += 1
        return new_conv_id("conv", self._counter)

    def _next_token(self) -> int:
        self._tokens += 1
        return self._tokens

    def _mcp_url(self, conv_id: str) -> str:
        return f"http://127.0.0.1:{self.settings.port}/mcp/{conv_id}"

    # ── turn planning ─────────────────────────────────────────────────────—

    def _agent_unavailable(self) -> OpenAIError:
        return OpenAIError(
            "cannot install the managed agent that carries the system prompt / "
            "tool allowlist / tool server", status_code=503, code="agent_unavailable",
        )

    def plan(self, req: ChatCompletionRequest, *, model: str, workdir: Path
             ) -> tuple[Signature, list[ChatMessage], Optional[str], bool]:
        """(signature, non-system messages, system prompt, inject_system)."""
        s = self.settings
        convo, system = split_system(req.messages)
        tools = list(req.tools or [])
        via_agent = s.system_prompt_via_agent

        def agents_ok() -> bool:  # lazy: "no agent" configs never touch the dir
            return self.agents is not None and self.agents.available()
        render = None
        agent_md: Optional[str] = None
        server: Optional[str] = None
        if tools:
            # The MCP server reaches agy only through the agent file, so a
            # tool turn without a writable plugin directory fails closed.
            if not agents_ok():
                raise self._agent_unavailable()
            server = server_name_for(tools)
            body = system if via_agent else None

            def render(conv_id: str, _body=body, _server=server) -> str:
                return render_agent(
                    _body, tools_mode=s.tool_turn_tools, name=self.agents.agent_name,
                    servers=[McpServerSpec(_server, self._mcp_url(conv_id))],
                    inherit_mcp=s.inherit_user_mcp,
                )
            template = agent_digest(render("<conv>"))
        else:
            if via_agent or s.plain_chat_tools != "all":
                if agents_ok():
                    agent_md = render_agent(
                        system if via_agent else None, tools_mode=s.plain_chat_tools,
                        name=self.agents.agent_name,
                        inherit_mcp=s.inherit_user_mcp,
                    )
                elif s.plain_chat_tools != "all":
                    raise self._agent_unavailable()
            template = agent_digest(agent_md) if agent_md is not None else None
        # No agent (mode off, or plugin dir unwritable) → the system message
        # rides as an instruction preamble instead.
        inject_system = bool(system) and (template is None or not via_agent)
        extra = hashlib.sha256((system or "").encode("utf-8", "surrogatepass")).hexdigest()[:16] \
            if inject_system else ""
        if s.image_input:
            extra += ":img"
        sig = Signature(
            model=model, workdir=str(workdir.resolve()), agent=template, agent_md=agent_md,
            tools_key=toolset_hash(tools) if tools else "", extra=extra, render=render,
            tools=tuple(tools), server=server,
        )
        return sig, convo, system, inject_system

    def _check_size(self, content: str) -> None:
        try:
            check_prompt_size(content, self.settings)
        except PromptTooLarge as e:
            raise OpenAIError(
                f"prompt is {e.nbytes} UTF-8 bytes; the server cap is {e.cap} "
                "(AGI_MAX_PROMPT_BYTES). Shorten the conversation.",
                status_code=400, type="invalid_request_error", code="context_length_exceeded",
            ) from e

    async def _turn_text(self, msg: ChatMessage, workdir: Path) -> str:
        if self.settings.image_input and _has_images([msg]):
            return await asyncio.to_thread(message_text_with_images, msg, workdir)
        return message_text(msg)

    # ── new turn (reuse / resume / fresh) ─────────────────────────────────—

    async def start_turn(
        self, req: ChatCompletionRequest, *, model: str, workdir: Path,
    ) -> Conversation:
        """Start the turn for a non-continuation request (or a rebuilt one);
        returns the conversation, claimed RUNNING, whose :meth:`run_turn`
        yields the answer. Never refused for capacity: every new turn gets its
        own process (the route's request deadline is the only bound)."""
        if not req.messages:
            raise OpenAIError("messages must not be empty", status_code=400, param="messages")
        sig, convo, system, inject_system = self.plan(req, model=model, workdir=workdir)
        key_sig = sig.key

        last = req.messages[-1]
        if last.role == "user" and len(convo) > 1:
            newest = await self._turn_text(last, workdir)
            # Validate BEFORE claiming: a claimed IDLE process must get its turn.
            if newest.strip():
                self._check_size(newest)
                key = history_key(key_sig, req.messages[:-1])
                conv = await self._claim_idle(key)
                if conv is not None:
                    try:
                        conv.session.send(newest)
                    except RuntimeError:
                        # Died between the liveness check and the write:
                        # fall through to a full fold.
                        await self._abort(conv)
                    except BaseException:
                        await self._abort(conv)
                        raise
                    else:
                        conv.current_msgs = list(req.messages)
                        self.stats["reuse_hits"] += 1
                        logger.info("conv=%s REUSED (sent newest message only, %d prior msgs)",
                                    conv.conv_id, len(req.messages) - 1)
                        return conv
                self.stats["reuse_misses"] += 1

        if self.settings.image_input and _has_images(convo):
            convo = await asyncio.to_thread(materialize_messages, convo, workdir)
        content = build_prompt(convo, system, inject_system=inject_system)
        if not content.strip():
            raise OpenAIError("no user content in messages", status_code=400, param="messages")
        if req.tools:
            names = ", ".join(t.function.name for t in req.tools)
            # Nudge the model toward its server: the schema cache the CLI
            # injects names the server, an explicit hint removes ambiguity.
            content += (f"\n\n[External functions are available on the MCP server "
                        f"'{sig.server}' via the call_mcp_tool tool: {names}. "
                        "Use them when needed.]")
        self._check_size(content)
        return await self._spawn(req, sig, text=content)

    # Older name of the fresh-turn entry point (tool turns only, pre-WP2).
    create = start_turn

    async def _claim_idle(self, key: str) -> Optional[Conversation]:
        """Claim the IDLE conversation parked under ``key`` (→ RUNNING), or
        ``None`` (the caller then starts fresh).

        The claim happens under the lock — a dead IDLE conversation is
        detached right there — so two concurrent identical requests cannot
        both adopt the same process."""
        dead: Optional[Conversation] = None
        async with self._lock:
            cid = self._idle.get(key)
            conv = self._conversations.get(cid) if cid else None
            if conv is not None and conv.state == IDLE:
                if getattr(conv.session, "running", False):
                    self._idle.pop(key, None)
                    conv.reuse_key = None
                    conv.state = RUNNING
                    conv.driven = False
                    conv.turn_token = self._next_token()
                    conv.touch()
                    return conv
                dead = conv  # crashed while parked
                self._retire_locked(dead)
        if dead is not None:
            await self._close(dead)
        return None

    async def _spawn(
        self, req: ChatCompletionRequest, sig: Signature, *, text: str,
    ) -> Conversation:
        """Start ``text`` on a new process: a warm-pool adoption or a cold
        spawn."""
        tools = list(req.tools or [])
        s = self.settings
        entry = None
        session = None
        bridge: Optional[ConversationBridge] = None
        conv_id = ""
        conv: Optional[Conversation] = None
        if self.pool is not None:
            entry = await self.pool.acquire(sig)
        if entry is not None:
            conv_id, session = entry.conv_id, entry.session
            bridge = entry.bridge
            if bridge is None:
                bridge = ConversationBridge(conv_id, tools,
                                            max_pending_tool_calls=s.max_pending_tool_calls)
            else:
                # Late-bind before the first message (agy lists tools at
                # its lazy MCP connect, which that message triggers).
                bridge.bind_tools(tools)
        else:
            conv_id = self._next_conv_id()
            bridge = ConversationBridge(conv_id, tools,
                                        max_pending_tool_calls=s.max_pending_tool_calls)
            session = AgySession(
                agy_bin=s.agy_bin, model=sig.model, workdir=sig.workdir,
                prompt=text, print_timeout_s=0,
                disable_slash_commands=s.disable_slash_commands,
                timing_log=s.timing_log,
                timing_label="tool" if tools else "fresh",
                event_queue_size=s.event_queue_size, stdin_timeout_s=s.stdin_timeout_s,
                observer=self.housekeeper,
            )
        try:
            if entry is None:
                if tools:
                    self.mcp.register(bridge)
                content = sig.agent_for(conv_id)
                if content is not None:
                    # Own transient agent file, no lock: boots in parallel
                    # with every other spawn.
                    try:
                        await self.agents.spawn(session, content)
                    except AgentUnavailable as e:
                        raise self._agent_unavailable() from e
                    except SpawnFailed as e:
                        raise OpenAIError(str(e), status_code=502, type="upstream_error",
                                          code="upstream_start_failed") from e
                else:
                    await session.start()
            conv = Conversation(
                conv_id=conv_id, slot=sig.server, session=session, bridge=bridge,
                model=sig.model, workdir=Path(sig.workdir), signature=sig.key,
                current_msgs=list(req.messages), turn_token=self._next_token(),
            )
            async with self._lock:
                self._conversations[conv_id] = conv
            if entry is not None:
                try:
                    session.send(text)
                except RuntimeError as e:
                    # The adopted process died before its first
                    # message: an upstream failure, not a server bug.
                    raise OpenAIError(f"agy exited before the turn started ({e})",
                                      status_code=502, type="upstream_error",
                                      code="upstream_exited") from e
        except BaseException:
            if conv is not None:
                async with self._lock:
                    self._conversations.pop(conv_id, None)
            try:
                await self._finish_cleanup(self._rollback(conv_id, session, bridge))
            except BaseException:
                logger.exception("conv=%s rollback failed", conv_id)
            raise
        if entry is not None:
            self.stats["pool_adopted"] += 1
        logger.info("conv=%s %s (model=%s, %d tools)", conv_id,
                    "adopted from warm pool" if entry is not None else "spawned",
                    sig.model, len(tools))
        return conv

    async def _rollback(self, conv_id: str, session: Any, bridge: Optional[ConversationBridge]) -> None:
        try:
            if bridge is not None:
                bridge.fail_all("conversation creation failed")
            await self.mcp.close_conversation(conv_id)
        finally:
            if session is not None:
                await session.aclose()

    # ── resume (continuation) ─────────────────────────────────────────────—

    def _prune_completed_locked(self) -> None:
        horizon = time.monotonic() - self.settings.completed_call_memory_s
        while self._completed and next(iter(self._completed.values())) < horizon:
            self._completed.popitem(last=False)

    def _remember_completed_locked(self, ids: list[str]) -> None:
        cap = self.settings.completed_call_memory_max
        if cap <= 0:
            return
        now = time.monotonic()
        for tid in ids:
            self._completed[tid] = now
            self._completed.move_to_end(tid)
        while len(self._completed) > cap:
            self._completed.popitem(last=False)

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

        dead: Optional[Conversation] = None
        async with self._lock:
            self._prune_completed_locked()
            if (any(self._pending_index.get(tid) is None for tid in ids)
                    and any(tid in self._completed for tid in ids)):
                self.stats["duplicates_rejected"] += 1
                raise DuplicateContinuation(
                    "these tool results were already delivered; the turn is not re-run")
            owners = {self._pending_index.get(tid) for tid in ids}
            if len(owners) != 1 or None in owners:
                if any(self._pending_index.get(tid) for tid in ids):
                    raise DuplicateContinuation("tool results span or partially match a live conversation")
                raise ExpiredContinuation("tool results must belong to one suspended conversation")
            conv = self._conversations.get(next(iter(owners)))
            if conv is None or conv.state != SUSPENDED:
                if conv is not None:
                    raise DuplicateContinuation("conversation is not waiting on these tool results")
                raise ExpiredContinuation("tool results reference an expired or unknown conversation")
            if not getattr(conv.session, "running", True):
                # Died while waiting for these results: nobody is answering
                # this turn, so it is a plain expiry (rebuildable).
                dead = conv
                self._retire_locked(conv)
            else:
                indexed = {tid for tid, cid in self._pending_index.items() if cid == conv.conv_id}
                if set(ids) != indexed or set(ids) != set(conv.bridge.pending_ids):
                    raise DuplicateContinuation("continuation must supply every pending tool result exactly once")
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
                conv.driven = False
                conv.turn_token = self._next_token()
                conv.current_msgs = list(req.messages)
                conv.touch()
                for tid, result in results:
                    self._pending_index.pop(tid)
                    conv.bridge.resolve(tid, result)
                self._remember_completed_locked(ids)
        if dead is not None:
            logger.info("conv=%s died while suspended", dead.conv_id)
            await self._close(dead)
            raise ExpiredContinuation(
                "the agy process waiting for these tool results has exited")
        logger.info("conv=%s resumed (%d tool results)", conv.conv_id, len(results))
        return conv

    async def resume_or_rebuild(
        self, req: ChatCompletionRequest, *, model: str, workdir: Path,
        before_create: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Conversation:
        """:meth:`resume`, or — when the suspended conversation is gone and
        ``settings.rebuild_on_expiry`` — a fresh turn from the full history the
        stateless client resent. The transcript already carries the
        assistant's tool_calls and the tool results, so the model produces the
        next turn as it would have; only the 409 the client could not retry is
        avoided. Duplicates are never rebuilt.

        ``before_create`` runs only on the rebuild path — the route uses it to
        take ownership of the request's resources (attachment pin)."""
        try:
            return await self.resume(req, model=model, workdir=workdir)
        except DuplicateContinuation:
            raise
        except ExpiredContinuation:
            if not self.settings.rebuild_on_expiry:
                raise
            logger.info("continuation expired; rebuilding a fresh %s turn from %d messages",
                        model, len(req.messages))
            if before_create is not None:
                await before_create()
            return await self.start_turn(req, model=model, workdir=workdir)

    # ── turn loop ─────────────────────────────────────────────────────────—

    async def run_turn(self, conv: Conversation) -> AsyncIterator[TurnChunk]:
        """Yield chunks until the turn suspends on tool calls, completes, or fails.

        One stream carries agy's stdout events AND the bridge's PendingCalls
        (the bridge injects into the session), so a hermes tool call — an HTTP
        ``call_tool`` on our bridge — is just another item in arrival order; no
        per-event task is created.

        Terminal bookkeeping (park IDLE / SUSPENDED, or close) happens BEFORE
        the terminal chunk is yielded: the consumer stops iterating on it, and
        code after that yield would only run at generator finalization."""
        session = conv.session
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.request_timeout_s
        filt = OutputFilter(flatten_tables=self.settings.flatten_markdown_tables)
        thinking = TurnThinking(self._brain, getattr(session, "conversation_id", None))
        emitted: list[str] = []
        last_poll = 0.0
        conv.driven = True
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                ev = await session.next_event(timeout=remaining)
                conv.touch()
                if isinstance(ev, Init):
                    self._learn_id(session, ev.conversation_id)
                    thinking.on_init(ev.conversation_id)
                now = loop.time()
                if (ev is None or ev is STREAM_CLOSED or now - last_poll >= _THINKING_POLL_S
                        or isinstance(ev, (Init, TurnDone, PendingCall, Error))):
                    last_poll = now
                    for block in thinking.poll():
                        yield ReasoningChunk(block)

                if ev is None:
                    raise TimeoutError
                if isinstance(ev, PendingCall):
                    if ev.future.done():
                        continue  # failed / answered while queued — stale
                    batch = [ev] + await self._drain_batch(conv, deadline)
                    batch = [pc for pc in batch if not pc.future.done()]
                    if loop.time() >= deadline:
                        raise TimeoutError
                    if not batch:
                        continue
                    async with self._lock:
                        if conv.state == RUNNING:
                            for pc in batch:
                                self._pending_index[pc.id] = conv.conv_id
                            conv.state = SUSPENDED
                    conv.touch()
                    tail = filt.flush()
                    logger.info("conv=%s suspended on %d tool call(s)", conv.conv_id, len(batch))
                    if tail:
                        yield TextChunk(tail)
                    yield ToolCallsChunk(batch)
                    return  # parked SUSPENDED; process blocked in call_tool
                if ev is STREAM_CLOSED:
                    # stdin mode: EOF means the process died mid-turn — there is
                    # no result line, so whatever text arrived is a TRUNCATED
                    # answer. Flush it (a streaming client already holds part of
                    # it) but report the failure: "stop" here would pass a
                    # cut-off reply off as complete.
                    tail = filt.flush()
                    diag_fn = getattr(session, "diagnostics", None)
                    diag = diag_fn() if diag_fn else "no diagnostics captured"
                    await self._close(conv)
                    if tail:
                        yield TextChunk(tail)
                    if ("".join(emitted) + tail).strip():
                        yield ErrorChunk(f"agy exited mid-turn ({diag})", status_code=502)
                    else:
                        yield ErrorChunk(f"agy produced no result ({diag})", status_code=502)
                    return
                if isinstance(ev, TextDelta):
                    out = filt.feed(ev.text)
                    if out:
                        emitted.append(out)
                        yield TextChunk(out)
                elif isinstance(ev, ToolStep):
                    if conv.slot and ev.matches_server(conv.slot):
                        # Companion stdout event of a bridge call; the queue
                        # item handles suspension — nothing to surface.
                        continue
                    if ev.state == "ACTIVE":
                        logger.debug("conv=%s internal tool: %s", conv.conv_id, ev.tool_name)
                        filt.tool_boundary()
                elif isinstance(ev, TurnDone):
                    self._learn_id(session, ev.conversation_id)
                    tail = filt.flush()
                    if not ("".join(emitted) + tail).strip() and ev.response_text:
                        # No deltas captured: the result line carries the text.
                        tail += filt.feed(ev.response_text) + filt.flush()
                    if tail:
                        emitted.append(tail)
                    await self._park_or_close(conv, "".join(emitted))
                    if tail:
                        yield TextChunk(tail)
                    yield DoneChunk("stop", usage_from_turn(ev))
                    return
                elif isinstance(ev, Error):
                    await self._close(conv)
                    yield ErrorChunk(ev.message, status_code=502)
                    return
                # Init / others: ignore.
        except TimeoutError:
            await self._close(conv)
            yield ErrorChunk("upstream timeout", status_code=504)
        except BaseException:
            # Abandoned mid-turn (client disconnect / cancellation) or failed on
            # a bug: agy has no per-turn cancel and this process' history now
            # diverges from what the client saw — kill it (a no-op once the
            # turn parked or closed).
            if conv.state == RUNNING:
                await self._abort(conv)
            raise

    @staticmethod
    def _learn_id(session: Any, conversation_id: Optional[str]) -> None:
        if conversation_id and getattr(session, "conversation_id", None) is None:
            session.conversation_id = conversation_id

    async def _drain_batch(self, conv: Conversation, deadline: float) -> list[PendingCall]:
        """Collect further calls of a parallel batch arriving within
        ``_BATCH_GRACE_S`` (never past the turn deadline). Non-call items seen
        meanwhile are pushed back, in order, for the turn loop."""
        loop = asyncio.get_running_loop()
        end = min(loop.time() + _BATCH_GRACE_S, deadline)
        batch: list[PendingCall] = []
        others: list[Any] = []
        while True:
            remaining = end - loop.time()
            if remaining <= 0:
                break
            item = await conv.session.next_event(timeout=remaining)
            if item is None:
                break
            if isinstance(item, PendingCall):
                batch.append(item)
                continue
            others.append(item)
            if item is STREAM_CLOSED:
                break
        for item in others:
            conv.session.push_back(item)
        return batch

    # ── park for reuse ────────────────────────────────────────────────────—

    async def _park_or_close(self, conv: Conversation, final_text: str) -> None:
        """After a clean turn end: park the live process IDLE under the history
        it has now processed (request messages + this reply as the client got
        it), so a matching next turn can adopt it. Otherwise close (its agy
        state is then deleted by the housekeeper)."""
        s = self.settings
        session = conv.session
        key = None
        if final_text.strip():
            key = history_key(conv.signature, list(conv.current_msgs) + [
                ChatMessage(role="assistant", content=final_text)])
        if (s.session_reuse and key is not None and getattr(session, "running", False)
                and not conv.bridge.pending_ids):
            replaced: Optional[Conversation] = None
            async with self._lock:
                if conv.state == RUNNING:
                    prev = self._idle.get(key)
                    if prev and prev != conv.conv_id:
                        replaced = self._conversations.get(prev)
                        if replaced is not None:
                            self._retire_locked(replaced)
                    conv.state = IDLE
                    conv.reuse_key = key
                    self._idle[key] = conv.conv_id
                    for tid in [t for t, c in self._pending_index.items() if c == conv.conv_id]:
                        self._pending_index.pop(tid, None)
            if conv.state == IDLE:
                conv.touch()
                release, conv.release_resources = conv.release_resources, None
                if release is not None:
                    release()
                logger.info("conv=%s parked IDLE for reuse", conv.conv_id)
            if replaced is not None:
                await self._close(replaced, graceful=True)
            if conv.state == IDLE:
                return
        await self._close(conv, graceful=bool(getattr(session, "running", False)))

    # ── teardown ──────────────────────────────────────────────────────────—

    def _retire_locked(self, conv: Conversation) -> None:
        """Detach ``conv`` from the reuse index (caller holds the lock) so no
        request can claim it while its teardown is pending."""
        key = conv.reuse_key
        if key and self._idle.get(key) == conv.conv_id:
            self._idle.pop(key, None)
        conv.reuse_key = None
        if conv.state != CLOSED:
            conv.state = CLOSING
            if conv.closing_since is None:
                conv.closing_since = time.monotonic()

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

    async def _close(self, conv: Conversation, *, graceful: bool = False) -> None:
        if conv._close_task is None:
            conv._close_task = asyncio.create_task(self._close_with_resources(conv, graceful))
        await self._finish_cleanup(conv._close_task)

    async def _close_with_resources(self, conv: Conversation, graceful: bool) -> None:
        try:
            await self._close_impl(conv, graceful)
        finally:
            release, conv.release_resources = conv.release_resources, None
            if release is not None:
                release()

    async def _close_impl(self, conv: Conversation, graceful: bool) -> None:
        if conv.state == CLOSED:
            return
        conv.state = CLOSED
        try:
            conv.bridge.fail_all("conversation closed")
        finally:
            try:
                async with self._lock:
                    if self._conversations.get(conv.conv_id) is conv:
                        self._conversations.pop(conv.conv_id, None)
                    if conv.reuse_key and self._idle.get(conv.reuse_key) == conv.conv_id:
                        self._idle.pop(conv.reuse_key, None)
                    stale = [tid for tid, cid in self._pending_index.items() if cid == conv.conv_id]
                    for tid in stale:
                        self._pending_index.pop(tid, None)
                await self.mcp.close_conversation(conv.conv_id)
            finally:
                try:
                    if graceful:
                        await conv.session.aclose(graceful=True)
                    else:
                        await conv.session.aclose()
                finally:
                    # Unreferenced now: its agy state goes at once. The
                    # session's own exit hook pokes too, but a process that
                    # died earlier (while still referenced) relies on this.
                    self._poke()
        logger.info("conv=%s closed", conv.conv_id)

    async def _abort(self, conv: Conversation) -> None:
        """Kill-close that survives cancellation of the caller: the kill is
        issued synchronously before the first await, and bookkeeping + teardown
        run shielded, so a second cancel (or a disconnect landing mid-teardown)
        cannot leave a live process behind."""
        kill = getattr(conv.session, "kill_now", None)
        if kill is not None:
            kill()
        await self._close(conv)

    async def abandon(self, conv: Conversation, token: Optional[int] = None) -> None:
        """The response for ``conv``'s current turn will never be consumed
        (client gone, cancellation, a failure after the claim): kill it unless
        it already reached a parked state. ``token`` pins the turn — a later
        turn already claimed on the same process is never touched."""
        if conv.state == RUNNING and (token is None or token == conv.turn_token):
            await self._abort(conv)

    async def close_all(self) -> None:
        for conv in list(self._conversations.values()):
            try:
                await self._close(conv, graceful=conv.state == IDLE)
            except Exception:  # keep going: one stuck process must not leak the rest
                logger.exception("conv=%s close failed", conv.conv_id)
        if self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def evict_idle(self) -> int:
        """Shut down every IDLE process now (its agy state is deleted)."""
        async with self._lock:
            victims = [c for c in self._conversations.values() if c.state == IDLE]
            for v in victims:
                self._retire_locked(v)
        for v in victims:
            await self._close(v, graceful=True)
        return len(victims)

    # ── introspection (metrics / tests) ───────────────────────────────────—

    def count(self, state: Optional[str] = None) -> int:
        if state is None:
            return sum(c.state not in (CLOSED, CLOSING) for c in self._conversations.values())
        return sum(c.state == state for c in self._conversations.values())

    # ── garbage collection ────────────────────────────────────────────────—

    async def gc_once(self) -> int:
        """Reap conversations whose TTL elapsed. Returns the count closed.

        * SUSPENDED past ``suspended_ttl_s`` (the client never returned its tool
          results): killed.
        * IDLE past ``idle_session_ttl_s`` (or dead): shut down gracefully; a
          late follow-up starts fresh with the full history folded.
        * RUNNING untouched for longer than a request may take — or whose turn
          loop never started within ``_UNDRIVEN_TTL_S`` (claimed, but the
          response was never consumed): killed."""
        now = time.monotonic()
        s = self.settings
        victims: list[tuple[Conversation, bool]] = []
        async with self._lock:
            for conv in list(self._conversations.values()):
                age = now - conv.last_activity
                if conv.state == SUSPENDED and age > s.suspended_ttl_s:
                    victims.append((conv, False))
                elif conv.state == IDLE and (age > s.idle_session_ttl_s
                                             or not getattr(conv.session, "running", False)):
                    self._retire_locked(conv)
                    victims.append((conv, True))
                elif conv.state == RUNNING and (
                        age > s.request_timeout_s + 60
                        or (not conv.driven and age > _UNDRIVEN_TTL_S)):
                    victims.append((conv, False))
                elif (conv.state == CLOSING and conv._close_task is None
                      and conv.closing_since is not None
                      and now - conv.closing_since > _CLOSING_GRACE_S):
                    # Detached for teardown by a caller that was cancelled
                    # before it could close it: nobody else will.
                    victims.append((conv, True))
            self._prune_completed_locked()
        for conv, idle in victims:
            logger.info("conv=%s GC (%s)", conv.conv_id,
                        "idle" if idle and conv.state != CLOSING else conv.state)
            try:
                await self._close(conv, graceful=idle)
            except Exception:  # pragma: no cover - defensive
                logger.exception("conv=%s GC teardown failed", conv.conv_id)
        return len(victims)

    async def gc_loop(self) -> None:
        interval = max(1, self.settings.gc_interval_s)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.gc_once()
                if self.agents is not None:
                    # Transient agent files a failed delete left behind.
                    self.agents.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("GC loop iteration failed")
