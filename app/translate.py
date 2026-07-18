"""Translation between the OpenAI Chat Completions wire format and agy.

Two directions:

* **OpenAI → agy**: split out ``system`` messages, fold the conversation into a
  single prompt string (agy has no system-prompt flag, so the system message is
  injected as an instruction preamble), or — on a conversation-cache hit — pass
  only the newest user message alongside ``--conversation <id>``.
* **agy → OpenAI**: build SSE chunks and the non-streaming response body, and
  derive ``usage`` from the ``result`` line (input/output/thinking tokens).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from app.events import TurnDone
from app.openai_models import ChatMessage

# ── ids / timestamps ─────────────────────────────────────────────────────────


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def now() -> int:
    return int(time.time())


# ── OpenAI message helpers ─────────────────────────────────────────────────—


def message_text(msg: ChatMessage) -> str:
    """Flatten an OpenAI message's ``content`` to plain text.

    Handles the string form and the content-parts list form
    (``[{"type":"text","text":...}, ...]``); non-text parts (images) are skipped.
    """
    c = msg.content
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for part in c:
        if isinstance(part, dict):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


def split_system(messages: list[ChatMessage]) -> tuple[list[ChatMessage], Optional[str]]:
    """Partition into (non-system messages, joined system prompt or None)."""
    system_parts: list[str] = []
    convo: list[ChatMessage] = []
    for m in messages:
        if m.role == "system":
            t = message_text(m)
            if t:
                system_parts.append(t)
        else:
            convo.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return convo, system


def _role_label(role: str) -> str:
    return {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role, role.capitalize())


def fold_conversation(convo: list[ChatMessage]) -> str:
    """Fold a (system-stripped) OpenAI conversation into one agy prompt.

    A lone trailing user message is sent as-is. Otherwise prior turns become a
    transcript preamble so a fresh, stateless conversation still has the context.
    """
    if not convo:
        return ""
    if len(convo) == 1 and convo[0].role == "user":
        return message_text(convo[0])

    lines: list[str] = []
    for m in convo[:-1]:
        text = message_text(m)
        if m.tool_calls:
            calls = ", ".join(
                f"{tc.function.name}({tc.function.arguments or ''})" for tc in m.tool_calls
            )
            text = (text + " " if text else "") + f"[called tools: {calls}]"
        if text:
            lines.append(f"{_role_label(m.role)}: {text}")
    last = convo[-1]
    last_line = f"{_role_label(last.role)}: {message_text(last)}"
    if lines:
        return (
            "Conversation so far:\n"
            + "\n".join(lines)
            + "\n\n"
            + last_line
        )
    return message_text(last)


def build_prompt(convo: list[ChatMessage], system: Optional[str], *, inject_system: bool) -> str:
    """The full first-turn prompt: optional system preamble + folded history.

    agy has no system-prompt flag (verified against 1.1.4), so the request's
    system message rides at the top of the prompt as explicit instructions.
    """
    folded = fold_conversation(convo)
    if system and inject_system:
        return (
            "<system_instructions>\n"
            f"{system}\n"
            "</system_instructions>\n\n"
            f"{folded}"
        )
    return folded


# ── usage mapping ──────────────────────────────────────────────────────────—


def usage_from_turn(turn: TurnDone) -> dict[str, Any]:
    """Derive an OpenAI ``usage`` object from an agy ``result`` line.

    agy reports ``input_tokens`` / ``output_tokens`` / ``thinking_tokens`` /
    ``total_tokens`` (output already includes thinking, observed live).
    Thinking is surfaced under the standard ``completion_tokens_details``
    key. Never fabricates — unknown values stay 0.
    """
    u = turn.usage or {}
    prompt = _int(u.get("input_tokens"))
    completion = _int(u.get("output_tokens"))
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": _int(u.get("total_tokens")) or (prompt + completion),
    }
    thinking = _int(u.get("thinking_tokens"))
    if thinking:
        usage["completion_tokens_details"] = {"reasoning_tokens": thinking}
    return usage


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


# ── SSE chunk builders ─────────────────────────────────────────────────────—

DONE = "data: [DONE]\n\n"


def sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(
    cid: str,
    model: str,
    created: int,
    *,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def role_chunk(cid: str, model: str, created: int) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"role": "assistant", "content": ""})


def text_chunk(cid: str, model: str, created: int, text: str) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"content": text})


def tool_calls_chunk(
    cid: str, model: str, created: int, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"tool_calls": tool_calls})


def finish_chunk(
    cid: str,
    model: str,
    created: int,
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={}, finish_reason=finish_reason, usage=usage)


# ── non-streaming response ─────────────────────────────────────────────────—


def completion_response(
    cid: str,
    model: str,
    created: int,
    *,
    content: Optional[str],
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    obj: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj
