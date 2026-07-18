"""``POST /v1/chat/completions`` — autonomous (no-tool) path.

Phase 1 handles requests with no external ``tools``: the OpenAI conversation is
either resumed as an existing agy conversation (conversation-cache hit → only
the newest user message is sent with ``--conversation <id>``) or folded into one
prompt for a fresh conversation. The streamed text becomes SSE or a single JSON
completion. The tool-passthrough path (suspend/resume via the MCP bridge) is
layered on in Phase 2; any ``tools`` field is currently rejected as unsupported.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agy_session import STREAM_CLOSED, AgySession
from app.config import ensure_workdir, get_settings, resolve_model
from app.convcache import ConversationCache
from app.errors import OpenAIError, error_envelope
from app.events import Error, TextDelta, ToolStep, TurnDone
from app.openai_models import ChatCompletionRequest
from app.textfilter import OutputFilter
from app.timing import TurnTimer
from app.translate import (
    DONE,
    build_prompt,
    completion_response,
    finish_chunk,
    new_completion_id,
    now,
    role_chunk,
    split_system,
    sse,
    text_chunk,
    usage_from_turn,
)

logger = logging.getLogger("agi.chat")

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    settings = get_settings()
    model = resolve_model(req.model, settings, effort=req.effort or settings.default_effort)
    try:
        workdir = settings.resolve_workdir(req.workdir)
    except ValueError as e:
        raise OpenAIError(str(e), status_code=400, param="workdir")
    ensure_workdir(workdir)

    if req.tools:
        raise OpenAIError(
            "tool calling is not supported yet (Phase 2)",
            status_code=400,
            type="invalid_request_error",
            param="tools",
        )

    cache: ConversationCache = request.app.state.conv_cache

    # ── conversation-cache hit: resume with only the newest user message ─────
    hit = cache.lookup(req.messages)
    if hit is not None:
        prompt = hit.new_user_text
        conversation_id = hit.conversation_id
        logger.info("conv-cache hit → resuming agy conversation %s", conversation_id)
    else:
        convo, system = split_system(req.messages)
        prompt = build_prompt(convo, system, inject_system=settings.system_prompt_injection)
        conversation_id = None

    if not prompt.strip():
        raise OpenAIError("no user content in messages", status_code=400, param="messages")
    if len(prompt) > settings.max_prompt_chars:
        raise OpenAIError(
            f"prompt is {len(prompt)} chars; the agy CLI accepts at most "
            f"{settings.max_prompt_chars} (argv limit). Shorten the conversation "
            "or let the server resume it via the conversation cache.",
            status_code=400,
            type="invalid_request_error",
            code="context_length_exceeded",
        )

    sess = AgySession(
        agy_bin=settings.agy_bin,
        model=model,
        workdir=workdir,
        prompt=prompt,
        conversation_id=conversation_id,
        print_timeout_s=settings.request_timeout_s,
        timing_log=settings.timing_log,
        timing_label="autonomous",
    )
    await sess.start()

    cid = new_completion_id()
    created = now()
    timeout = settings.request_timeout_s

    flatten = settings.flatten_markdown_tables
    if req.stream:
        return StreamingResponse(
            _stream(sess, cache, req, cid, model, created, timeout, flatten,
                    settings.timing_log),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    body = await _collect(sess, cache, req, cid, model, created, timeout, flatten,
                          settings.timing_log)
    return JSONResponse(content=body)


async def _stream(
    sess: AgySession, cache: ConversationCache, req: ChatCompletionRequest,
    cid: str, model: str, created: int, timeout: float,
    flatten_tables: bool = True, timing: bool = False,
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    joiner = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "autonomous")
    completion_tokens = 0
    text_parts: list[str] = []
    try:
        yield sse(role_chunk(cid, model, created))
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                yield sse(error_envelope("upstream timeout", type="timeout_error"))
                break
            ev = await sess.next_event(timeout=remaining)
            if ev is None:
                yield sse(error_envelope("upstream timeout", type="timeout_error"))
                break
            if ev is STREAM_CLOSED:
                tail = joiner.flush()
                if tail:
                    text_parts.append(tail)
                    yield sse(text_chunk(cid, model, created, tail))
                if not text_parts:
                    yield sse(error_envelope(
                        f"agy produced no result ({sess.diagnostics()})",
                        type="upstream_error",
                    ))
                    break
                yield sse(finish_chunk(cid, model, created, "stop"))
                break
            if isinstance(ev, TextDelta):
                out = joiner.feed(ev.text)
                if out:
                    timer.first_token()
                    text_parts.append(out)
                    yield sse(text_chunk(cid, model, created, out))
            elif isinstance(ev, TurnDone):
                tail = joiner.flush()
                if tail:
                    text_parts.append(tail)
                    yield sse(text_chunk(cid, model, created, tail))
                usage = usage_from_turn(ev)
                completion_tokens = usage.get("completion_tokens", 0)
                _store(cache, req, text_parts, ev)
                yield sse(finish_chunk(cid, model, created, "stop", usage))
                break
            elif isinstance(ev, Error):
                yield sse(error_envelope(ev.message, type="upstream_error"))
                break
            elif isinstance(ev, ToolStep):
                # Builtin tool runs internally; mark the seam between two text
                # segments so the next one starts on a fresh blank line.
                if ev.state == "ACTIVE":
                    logger.debug("internal tool use: %s", ev.tool_name)
                    joiner.tool_boundary()
                continue
            # Init / others: ignore.
        yield DONE
    finally:
        timer.done(completion_tokens)
        await sess.aclose()


async def _collect(
    sess: AgySession, cache: ConversationCache, req: ChatCompletionRequest,
    cid: str, model: str, created: int, timeout: float,
    flatten_tables: bool = True, timing: bool = False,
) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    joiner = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "autonomous")
    completion_tokens = 0
    chunks: list[str] = []
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise OpenAIError("upstream timeout", status_code=504, type="timeout_error")
            ev = await sess.next_event(timeout=remaining)
            if ev is None:
                raise OpenAIError("upstream timeout", status_code=504, type="timeout_error")
            if ev is STREAM_CLOSED:
                chunks.append(joiner.flush())
                text = "".join(chunks)
                if not text.strip():
                    raise OpenAIError(
                        f"agy produced no result ({sess.diagnostics()})",
                        status_code=502, type="upstream_error",
                    )
                return completion_response(
                    cid, model, created, content=text, finish_reason="stop"
                )
            if isinstance(ev, TextDelta):
                timer.first_token()
                chunks.append(joiner.feed(ev.text))
            elif isinstance(ev, TurnDone):
                chunks.append(joiner.flush())
                # Prefer the delta-assembled text (it carries tool-boundary
                # seams); fall back to `response` when no deltas were captured.
                assembled = "".join(chunks)
                text = assembled if assembled.strip() else (ev.response_text or "")
                usage = usage_from_turn(ev)
                completion_tokens = usage.get("completion_tokens", 0)
                _store(cache, req, [text], ev)
                return completion_response(
                    cid, model, created,
                    content=text,
                    finish_reason="stop",
                    usage=usage,
                )
            elif isinstance(ev, Error):
                raise OpenAIError(ev.message, status_code=502, type="upstream_error")
            elif isinstance(ev, ToolStep):
                if ev.state == "ACTIVE":
                    logger.debug("internal tool use: %s", ev.tool_name)
                    joiner.tool_boundary()
                continue
    finally:
        timer.done(completion_tokens)
        await sess.aclose()


def _store(cache: ConversationCache, req: ChatCompletionRequest,
           text_parts: list[str], turn: TurnDone) -> None:
    """Record the finished turn in the conversation cache (best-effort)."""
    conv_id = turn.conversation_id
    text = "".join(text_parts) or (turn.response_text or "")
    if conv_id and text.strip():
        cache.store(req.messages, text, conv_id)
