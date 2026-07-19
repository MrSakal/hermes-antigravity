"""Translation between the OpenAI Chat Completions wire format and agy.

Two directions:

* **OpenAI → agy**: split out ``system`` messages (delivered via a generated
  ``--agent`` definition, see :mod:`app.agent_config`; injected as an
  instruction preamble only as the fallback), fold the conversation into a
  single prompt string, or — on a conversation-cache hit — pass only the
  newest user message alongside ``--conversation <id>``.
* **agy → OpenAI**: build SSE chunks and the non-streaming response body, and
  derive ``usage`` from the ``result`` line (input/output/thinking tokens).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from app.events import TurnDone
from app.openai_models import ChatMessage

logger = logging.getLogger("agi.translate")

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


# ── image materialization ────────────────────────────────────────────────—
#
# agy has no image flag and no stdin channel (the prompt is argv), but its
# builtin file tools can view local images and the models are multimodal
# (verified live: a PNG dropped in the workdir is correctly described). So
# data-URL images from OpenAI `image_url` content parts are written to
# <workdir>/attachments/img-<sha1>.<ext> and the prompt references the file.

ATTACHMENTS_DIR = "attachments"

_IMAGE_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
_MAX_IMAGE_BASE64_BYTES = 20 * 1024 * 1024


def _materialize_image_part(part: dict[str, Any], workdir: Path) -> Optional[str]:
    """Write one ``image_url`` data-URL part to disk; return the relative path.

    Returns ``None`` (part is skipped, matching the old behavior) for remote
    URLs, unknown media types, invalid base64, or oversized payloads.
    """
    image_url = part.get("image_url")
    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
    header, sep, data = str(url).partition(",")
    media_type = header[5:].split(";", 1)[0].lower() if header.startswith("data:") else ""
    ext = _IMAGE_EXT.get(media_type)
    if (
        not sep
        or ";base64" not in header.lower()
        or ext is None
        or not data
        or len(data) > _MAX_IMAGE_BASE64_BYTES
    ):
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        return None
    name = "img-" + hashlib.sha1(raw).hexdigest()[:16] + ext
    rel = f"{ATTACHMENTS_DIR}/{name}"
    target = workdir / ATTACHMENTS_DIR / name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():  # content-addressed: identical bytes, same file
            target.write_bytes(raw)
    except OSError:
        logger.warning("could not write attachment %s", target, exc_info=True)
        return None
    return rel


def message_text_with_images(msg: ChatMessage, workdir: Path) -> str:
    """Like :func:`message_text`, but data-URL images become file references."""
    c = msg.content
    if not isinstance(c, list):
        return message_text(msg)
    parts: list[str] = []
    for part in c:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            rel = _materialize_image_part(part, workdir)
            if rel:
                parts.append(
                    f"\n[Attached image file: {rel} — open it with your "
                    "image/file viewing tool to see it.]\n"
                )
    return "".join(parts)


def materialize_messages(messages: list[ChatMessage], workdir: Path) -> list[ChatMessage]:
    """Replace image parts with file references throughout a message list.

    Messages without image parts pass through unchanged; the input list is
    never mutated (the conversation cache keys on the original request).
    """
    out: list[ChatMessage] = []
    for m in messages:
        has_image = isinstance(m.content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in m.content
        )
        if has_image:
            out.append(m.model_copy(update={"content": message_text_with_images(m, workdir)}))
        else:
            out.append(m)
    return out


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

    ``inject_system=True`` is the FALLBACK delivery for the system message
    (callers pass it when no ``--agent`` definition could be used): it rides at
    the top of the prompt as explicit instructions.
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
