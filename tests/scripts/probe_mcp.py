#!/usr/bin/env python3
"""Probe the MCP bridge end-to-end against the live `agy` CLI.

Stands up a real in-process MCP server (one tool `get_weather`, NON-suspending —
returns canned JSON immediately) mounted at /mcp/<conv_id> via a single global
StreamableHTTPSessionManager, declares it as the only MCP server of the managed
agent file (``inheritMcp: false`` + ``mcpServers``, see app/agent_config.py —
agy has no --mcp-config flag), then launches `agy --agent hermes` and asks it to
use the tool.

Confirms:
  #1  the tool call arrives at our handler via agy's generic `call_mcp_tool`
      builtin (ServerName=<slot>, ToolName=get_weather) with no permission stop
      under --dangerously-skip-permissions
  #2  the agent-file mcpServers entry is picked up at boot (the shared
      ~/.gemini/config/mcp_config.json is never written)
  #3  /mcp/<conv_id> routing through one mounted manager works

The managed agent file is left in place (the server rewrites it per spawn).

Env knobs:
  PROBE_PORT=8799
"""
import asyncio
import contextlib
import os
import sys
import tempfile
from contextvars import ContextVar
from pathlib import Path

import mcp.types as mtypes
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount

from app.agy_session import STREAM_CLOSED, AgySession
from app.events import Error, Init, TextDelta, ToolStep, TurnDone
from app.agent_config import AgentPluginManager, McpServerSpec, render_agent

PORT = int(os.environ.get("PROBE_PORT", "8799"))
CONV_ID = "probeconv"
MODEL = "Gemini 3.8 Flash (Medium)"
WORKDIR = Path(tempfile.gettempdir()) / "agi_probe_ws"

# Shared probe state
HANDLER_CALLS: list[tuple[str, dict]] = []
current_conv_id: ContextVar[str] = ContextVar("current_conv_id", default="")

# ── MCP server ───────────────────────────────────────────────────────────────
server: Server = Server("hermes")


@server.list_tools()
async def list_tools() -> list[mtypes.Tool]:
    print(f"[mcp] list_tools (conv={current_conv_id.get()})", flush=True)
    return [
        mtypes.Tool(
            name="get_weather",
            description="Get the current weather for a city.",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        )
    ]


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict):
    conv = current_conv_id.get()
    print(f"[mcp] *** call_tool name={name} args={arguments} conv={conv} ***", flush=True)
    HANDLER_CALLS.append((name, arguments))
    return [mtypes.TextContent(type="text", text='{"temp_c": 21, "summary": "sunny"}')]


session_manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)


class McpDispatcher:
    """ASGI app mounted at /mcp; routes /mcp/<conv_id> to the one manager,
    setting the conv_id contextvar (captured into the server task)."""

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope.get("path", "")  # already stripped of /mcp by Mount
        conv_id = path.strip("/").split("/")[0] if path.strip("/") else ""
        current_conv_id.set(conv_id)
        # Rewrite to root so the transport sees its own base path.
        new_scope = dict(scope)
        new_scope["path"] = "/"
        new_scope["raw_path"] = b"/"
        await session_manager.handle_request(new_scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app):
    async with session_manager.run():
        yield


def build_app() -> Starlette:
    async def health(request):
        return JSONResponse({"ok": True})

    return Starlette(
        lifespan=lifespan,
        routes=[Mount("/mcp", app=McpDispatcher()), Mount("/health", app=health)],
    )


# ── driver ───────────────────────────────────────────────────────────────────


async def drive_agy(slot: str) -> bool:
    sess = AgySession(
        agy_bin="agy",
        model=MODEL,
        workdir=WORKDIR,
        prompt=(
            f"Use the get_weather MCP tool (server {slot!r}) to get the weather "
            "for Paris, then tell me the result in one sentence."
        ),
    )
    md = render_agent(None, servers=[McpServerSpec(slot, f"http://127.0.0.1:{PORT}/mcp/{CONV_ID}")])
    await AgentPluginManager().spawn(sess, md)
    saw_slot_toolstep = False
    final_text = ""
    done: TurnDone | None = None
    try:
        while True:
            ev = await sess.next_event(timeout=180)
            if ev is None:
                print("[probe] TIMEOUT", flush=True)
                print("[probe] diagnostics:", sess.diagnostics(), flush=True)
                break
            if ev is STREAM_CLOSED:
                print("[probe] stream closed", flush=True)
                break
            if isinstance(ev, Init):
                print(f"[agy] init conv={ev.conversation_id}", flush=True)
            elif isinstance(ev, TextDelta):
                final_text += ev.text
            elif isinstance(ev, ToolStep):
                print(f"[agy] tool step: {ev.tool_name} server={ev.mcp_server} "
                      f"fn={ev.hermes_function_name} ({ev.state})", flush=True)
                if ev.matches_server(slot):
                    saw_slot_toolstep = True
            elif isinstance(ev, TurnDone):
                done = ev
                print(f"[agy] TURN DONE conv={ev.conversation_id}", flush=True)
                break
            elif isinstance(ev, Error):
                print(f"[agy] ERROR: {ev.message}", flush=True)
                break
    finally:
        await sess.aclose()

    if not final_text and done and done.response_text:
        final_text = done.response_text

    print("\n========== PROBE RESULTS ==========", flush=True)
    print(f"handler invoked:         {len(HANDLER_CALLS)} time(s) -> {HANDLER_CALLS}", flush=True)
    print(f"call_mcp_tool to {slot!r}: {saw_slot_toolstep}", flush=True)
    print(f"final text:              {final_text!r}", flush=True)
    ok = len(HANDLER_CALLS) >= 1 and "21" in final_text
    print("PROBE:", "PASS" if ok else "FAIL", flush=True)
    return ok


async def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    config = uvicorn.Config(build_app(), host="127.0.0.1", port=PORT, log_level="warning")
    server_obj = uvicorn.Server(config)
    serve_task = asyncio.create_task(server_obj.serve())
    for _ in range(100):
        if server_obj.started:
            break
        await asyncio.sleep(0.05)
    print(f"[probe] MCP server up on :{PORT}", flush=True)

    slot = "hermes-probe"
    try:
        ok = await drive_agy(slot)
    finally:
        server_obj.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(serve_task, timeout=5)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
