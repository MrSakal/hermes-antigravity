"""In-process MCP server that exposes hermes' OpenAI functions to the agy CLI.

Ported from hermes-claude's bridge; the mechanics are identical, only the
client differs. Architecture:

* A **single** low-level :class:`mcp.server.lowlevel.Server` and a **single**
  :class:`StreamableHTTPSessionManager` serve every conversation. Each
  conversation gets its own entry in the shared Antigravity MCP config (see
  :mod:`app.mcp_config`) whose ``serverUrl`` is ``/mcp/<conv_id>``; an ASGI
  dispatcher reads ``conv_id`` from the path and stashes it in a
  :class:`ContextVar` *before* delegating to ``handle_request``. The manager
  spawns the per-session server loop via ``task_group.start()``, which copies
  the current contextvar context into that task — so ``list_tools`` /
  ``call_tool`` see the right ``conv_id`` with no cross-talk.

* agy reaches these tools through its generic ``call_mcp_tool`` builtin
  (``ServerName=<slot>, ToolName=<fn>, Arguments={...}``). Under
  ``--dangerously-skip-permissions`` the call lands on ``call_tool`` directly.
  The handler mints an OpenAI ``tool_call_id``, registers a
  :class:`PendingCall`, pushes it onto the conversation's queue, and **awaits
  its Future** — blocking the agy subprocess inside the tool call until hermes
  returns the result on a later HTTP request (verified live: agy tolerates 90s+
  inside a tool call; the ceiling is its ``--print-timeout``).

* agy's client sends a nonstandard ``server/discover`` request before
  ``initialize``; the MCP SDK answers method-not-found, which agy tolerates.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

import mcp.types as mtypes
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.types import Receive, Scope, Send

from app.openai_models import ToolDef

logger = logging.getLogger("agi.mcp")

# Set by the ASGI dispatcher per request; read by list_tools / call_tool, which
# run in a task that inherits a copy of this context (see module docstring).
current_conv_id: ContextVar[str] = ContextVar("current_conv_id", default="")


class ToolResultError(Exception):
    """Raised inside ``call_tool`` when a pending call is cancelled/expired so the
    error propagates back to Claude instead of hanging."""


@dataclass
class PendingCall:
    """One in-flight hermes function call, blocking the subprocess on its Future."""

    id: str  # OpenAI tool_call_id ("call_…")
    name: str  # bare function name (e.g. "get_weather")
    arguments: dict[str, Any]
    future: "asyncio.Future[str]"

    def arguments_json(self) -> str:
        try:
            return json.dumps(self.arguments)
        except (TypeError, ValueError):
            return "{}"

    def openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_json()},
        }


class ConversationBridge:
    """Per-conversation tool registry + pending-call plumbing."""

    def __init__(self, conv_id: str, tools: list[ToolDef]) -> None:
        self.conv_id = conv_id
        self.tools = tools
        self._incoming: asyncio.Queue[PendingCall] = asyncio.Queue()
        self._pending: dict[str, PendingCall] = {}

    # ── advertised to Claude ────────────────────────────────────────────────

    def mcp_tools(self) -> list[mtypes.Tool]:
        out: list[mtypes.Tool] = []
        for t in self.tools:
            fn = t.function
            schema = fn.parameters or {"type": "object", "properties": {}}
            if not isinstance(schema, dict) or "type" not in schema:
                schema = {"type": "object", "properties": {}}
            out.append(
                mtypes.Tool(
                    name=fn.name,
                    description=fn.description or "",
                    inputSchema=schema,
                )
            )
        return out

    # ── call_tool side (blocks the subprocess) ───────────────────────────────

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        call = PendingCall(
            id="call_" + uuid.uuid4().hex,
            name=name,
            arguments=arguments or {},
            future=loop.create_future(),
        )
        self._pending[call.id] = call
        await self._incoming.put(call)
        logger.debug("conv=%s dispatch %s id=%s — suspending", self.conv_id, name, call.id)
        try:
            return await call.future
        finally:
            self._pending.pop(call.id, None)

    # ── chat-loop side ────────────────────────────────────────────────────—

    async def next_incoming(self, timeout: Optional[float] = None) -> Optional[PendingCall]:
        """Await the next pending call; ``None`` when ``timeout`` elapses."""
        try:
            if timeout is None:
                return await self._incoming.get()
            return await asyncio.wait_for(self._incoming.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def drain_incoming(self, *, grace_s: float = 0.15) -> list[PendingCall]:
        """Collect any additional calls that arrive within a short grace window
        (a parallel batch issued by one assistant step)."""
        batch: list[PendingCall] = []
        while True:
            try:
                batch.append(await asyncio.wait_for(self._incoming.get(), timeout=grace_s))
            except asyncio.TimeoutError:
                return batch

    def resolve(self, tool_call_id: str, result: str) -> bool:
        """Deliver a hermes tool result by id. Returns True if a call was waiting."""
        call = self._pending.get(tool_call_id)
        if call is None or call.future.done():
            return False
        call.future.set_result(result)
        return True

    def fail_all(self, message: str) -> None:
        """Error out every pending call (GC/disconnect) so Claude unblocks."""
        for call in list(self._pending.values()):
            if not call.future.done():
                call.future.set_exception(ToolResultError(message))

    @property
    def pending_ids(self) -> list[str]:
        return list(self._pending.keys())


class McpBridge:
    """Owns the single MCP server, session manager, and conversation registry."""

    def __init__(self) -> None:
        self._registry: dict[str, ConversationBridge] = {}
        self.server: Server = Server("hermes")
        self._install_handlers()
        self.session_manager = StreamableHTTPSessionManager(
            app=self.server, json_response=False, stateless=False
        )

    def _install_handlers(self) -> None:
        @self.server.list_tools()
        async def _list_tools() -> list[mtypes.Tool]:
            bridge = self._current_bridge()
            if bridge is None:
                return []
            return bridge.mcp_tools()

        @self.server.call_tool(validate_input=False)
        async def _call_tool(name: str, arguments: dict):
            bridge = self._current_bridge()
            if bridge is None:
                raise ToolResultError(f"no active conversation for tool {name}")
            result = await bridge.dispatch(name, arguments or {})
            return [mtypes.TextContent(type="text", text=result)]

    def _current_bridge(self) -> Optional[ConversationBridge]:
        conv_id = current_conv_id.get()
        bridge = self._registry.get(conv_id)
        if bridge is None:
            logger.warning("no bridge registered for conv_id=%r", conv_id)
        return bridge

    # ── registry ──────────────────────────────────────────────────────────—

    def register(self, bridge: ConversationBridge) -> None:
        self._registry[bridge.conv_id] = bridge

    def unregister(self, conv_id: str) -> None:
        self._registry.pop(conv_id, None)

    def get(self, conv_id: str) -> Optional[ConversationBridge]:
        return self._registry.get(conv_id)

    # ── ASGI / lifespan ───────────────────────────────────────────────────—

    @contextlib.asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        async with self.session_manager.run():
            yield

    def asgi_app(self):
        manager = self.session_manager

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                return
            path = scope.get("path", "") or ""
            segments = [s for s in path.split("/") if s]
            # URL is /<prefix>/<conv_id>; the conv_id is the last segment whether
            # or not the Mount stripped the prefix.
            conv_id = segments[-1] if segments else ""
            current_conv_id.set(conv_id)
            new_scope = dict(scope)
            new_scope["path"] = "/"
            new_scope["raw_path"] = b"/"
            await manager.handle_request(new_scope, receive, send)

        return app
