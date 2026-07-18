"""Semantic events distilled from the `agy` CLI ``--output-format stream-json``
stream.

The wire format (verified live against agy 1.1.4) is one JSON object per line:

* ``{"event":"init","conversation_id":...,"init":{"cwd":...,"tools":[...],
  "permission_mode":...}}`` — once per turn, first line.
* ``{"event":"step_update","step_update":{"conversation_id":...,"step_index":N,
  "state":"ACTIVE"|"DONE"|"ERROR","step_type":...,...}}`` — step progress.
  ``step_type`` values seen: ``user_input``, ``agent_response`` (carries
  incremental ``text_delta`` on ACTIVE and the final chunk on DONE),
  ``tool`` (``tool_name`` + ``tool_info{name,parameters,output}``),
  ``checkpoint``, ``subagent``, ``unknown``.
* ``{"event":"result","result":{"conversation_id":...,"status":"SUCCESS"|"ERROR",
  "response":...,"error":...,"num_turns":N,"duration_seconds":...,
  "usage":{"input_tokens","output_tokens","thinking_tokens","total_tokens"}}}``
  — last line, the turn boundary.

Parsing is deliberately **defensive and total**: a blank line, non-JSON noise, or
a renamed/missing field in a future CLI version degrades to ``None`` ("ignore
this line") rather than raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Union

# The generic builtin through which agy reaches every MCP tool. Calls whose
# ``ServerName`` equals our bridge's server name are hermes function calls;
# every other tool step is an Antigravity builtin running internally.
MCP_TOOL_NAME = "call_mcp_tool"
HERMES_SERVER_NAME = "hermes"


# ── Event types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Init:
    conversation_id: Optional[str]
    cwd: Optional[str]
    tools: list[str] = field(default_factory=list)
    permission_mode: Optional[str] = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolStep:
    """One ``step_type=tool`` update (ACTIVE = call started, DONE = finished)."""

    state: str  # "ACTIVE" | "DONE" | "ERROR" | ...
    step_index: Optional[int]
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None

    @property
    def is_mcp(self) -> bool:
        return self.tool_name == MCP_TOOL_NAME

    @property
    def mcp_server(self) -> Optional[str]:
        v = self.parameters.get("ServerName")
        return v if isinstance(v, str) else None

    @property
    def is_hermes(self) -> bool:
        return self.is_mcp and self.mcp_server == HERMES_SERVER_NAME

    @property
    def hermes_function_name(self) -> str:
        """The bare OpenAI function name of a hermes ``call_mcp_tool`` call."""
        v = self.parameters.get("ToolName")
        return v if isinstance(v, str) else self.tool_name

    @property
    def hermes_arguments(self) -> dict[str, Any]:
        v = self.parameters.get("Arguments")
        return v if isinstance(v, dict) else {}


@dataclass(frozen=True)
class TurnDone:
    """A clean ``result`` line (status SUCCESS) ending the turn."""

    response_text: Optional[str] = None
    num_turns: Optional[int] = None
    usage: dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass(frozen=True)
class Error:
    """A ``result`` line whose status is not SUCCESS."""

    message: str
    conversation_id: Optional[str] = None


ChatEvent = Union[Init, TextDelta, ToolStep, TurnDone, Error]


# ── Parser ──────────────────────────────────────────────────────────────────


def parse_line(line: str) -> Optional[ChatEvent]:
    """Parse one raw JSONL line from the CLI into a :data:`ChatEvent`, or ``None``.

    Total and panic-free: blank/non-JSON/unknown lines and missing fields all
    yield ``None``.
    """
    line = line.strip()
    if not line:
        return None
    try:
        v = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(v, dict):
        return None

    ev = v.get("event")
    if ev == "init":
        return _parse_init(v)
    if ev == "step_update":
        return _parse_step_update(v)
    if ev == "result":
        return _parse_result(v)
    return None


def _parse_init(v: dict[str, Any]) -> Optional[ChatEvent]:
    init = v.get("init")
    init = init if isinstance(init, dict) else {}
    tools = init.get("tools")
    return Init(
        conversation_id=_as_str(v.get("conversation_id")),
        cwd=_as_str(init.get("cwd")),
        tools=[t for t in tools if isinstance(t, str)] if isinstance(tools, list) else [],
        permission_mode=_as_str(init.get("permission_mode")),
    )


def _parse_step_update(v: dict[str, Any]) -> Optional[ChatEvent]:
    su = v.get("step_update")
    if not isinstance(su, dict):
        return None
    step_type = su.get("step_type")

    if step_type == "agent_response":
        text = su.get("text_delta")
        if isinstance(text, str) and text:
            return TextDelta(text)
        return None

    if step_type == "tool":
        info = su.get("tool_info")
        info = info if isinstance(info, dict) else {}
        params = info.get("parameters")
        return ToolStep(
            state=_as_str(su.get("state")) or "",
            step_index=_as_int(su.get("step_index")),
            tool_name=_as_str(su.get("tool_name")) or _as_str(info.get("name")) or "tool",
            parameters=params if isinstance(params, dict) else {},
            output=_as_str(info.get("output")),
        )

    # user_input / checkpoint / subagent / unknown: nothing to surface.
    return None


def _parse_result(v: dict[str, Any]) -> Optional[ChatEvent]:
    r = v.get("result")
    if not isinstance(r, dict):
        return None
    conv_id = _as_str(r.get("conversation_id"))
    status = _as_str(r.get("status"))
    if status != "SUCCESS":
        msg = _as_str(r.get("error")) or _as_str(r.get("response")) or "the request failed"
        return Error(message=msg, conversation_id=conv_id)
    usage = r.get("usage")
    return TurnDone(
        response_text=_as_str(r.get("response")),
        num_turns=_as_int(r.get("num_turns")),
        usage=usage if isinstance(usage, dict) else {},
        conversation_id=conv_id,
        duration_seconds=_as_float(r.get("duration_seconds")),
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def _as_str(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None


def _as_int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None
