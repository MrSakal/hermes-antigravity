"""``POST /v1/chat/completions``.

Two paths:

* **Autonomous** (no ``tools``): the OpenAI conversation is either resumed as
  an existing agy conversation (conversation-cache hit → only the newest user
  message is sent with ``--conversation <id>``) or folded into one prompt for a
  fresh conversation. The streamed text becomes SSE or a single JSON completion.
  The spawned agent carries the ``plain_chat_tools`` allowlist (default: read
  only) — this path answers hermes' side calls on untrusted content.
* **Tool passthrough** (``tools`` present, or a continuation carrying tool
  results): suspend/resume via the MCP bridge, see :mod:`app.conversation`.

Client disconnects: a streaming response is cancelled by Starlette when the
client goes away (the ``finally`` closes the subprocess); a non-streaming
request is not, so ``_collect`` polls ``request.is_disconnected()`` and kills
the subprocess instead of letting it run to ``request_timeout_s`` for nobody.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agy_session import STREAM_CLOSED, AgySession
from app.config import (
    PromptTooLarge,
    Settings,
    UnknownModel,
    ensure_workdir,
    get_settings,
    resolve_model,
)
from app.config import check_prompt_size as _check_prompt_size
from app.convcache import ConversationCache
from app.conversation import (
    ConversationManager,
    DoneChunk,
    ErrorChunk,
    ExpiredContinuation,
    ReasoningChunk,
    TextChunk,
    ToolBoundaryChunk,
    ToolCallsChunk,
)
from app.errors import OpenAIError, error_envelope
from app.events import Error, Init, TextDelta, ToolStep, TurnDone
from app.model_catalog import ModelCatalog
from app.openai_models import ChatCompletionRequest
from app.textfilter import OutputFilter
from app.thinking import TurnThinking, default_brain_path
from app.timing import TurnTimer
from app.translate import (
    DONE,
    build_prompt,
    completion_response,
    finish_chunk,
    materialize_messages,
    message_text_with_images,
    new_completion_id,
    now,
    reasoning_chunk,
    role_chunk,
    split_system,
    sse,
    text_chunk,
    tool_calls_chunk,
    usage_from_turn,
)

logger = logging.getLogger("agi.chat")

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ClientDisconnected(Exception):
    """The HTTP client went away while the turn was still running."""


def resolve_request_model(
    req: ChatCompletionRequest, settings: Settings, catalog: Optional[ModelCatalog]
) -> str:
    """``resolve_model`` against the live catalog, mapped to an OpenAI error."""
    known = catalog.display_names() if catalog and catalog.loaded else None
    fams = catalog.families() if catalog and catalog.loaded else None
    try:
        return resolve_model(
            req.model, settings, effort=req.effort or settings.default_effort,
            known_models=known, families=fams,
        )
    except UnknownModel as e:
        raise OpenAIError(
            f"model {str(e)!r} is not available on this Antigravity install; "
            f"see GET /v1/models",
            status_code=400, type="invalid_request_error", param="model",
            code="model_not_found",
        )


def check_prompt_size(prompt: str, settings: Settings) -> int:
    """Return the UTF-8 byte size; raise 400 when over the sanity cap."""
    try:
        return _check_prompt_size(prompt, settings)
    except PromptTooLarge as e:
        raise OpenAIError(
            f"prompt is {e.nbytes} UTF-8 bytes; the server cap is {e.cap} "
            "(AGI_MAX_PROMPT_BYTES). Shorten the conversation or let the server "
            "resume it via the conversation cache.",
            status_code=400, type="invalid_request_error", code="context_length_exceeded",
        )


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    settings = get_settings()
    model = resolve_request_model(req, settings, getattr(request.app.state, "model_catalog", None))
    try:
        workdir = settings.resolve_workdir(req.workdir)
    except ValueError as e:
        raise OpenAIError(str(e), status_code=400, param="workdir")
    ensure_workdir(workdir)

    # ── tool-passthrough path (suspend/resume via the MCP bridge) ──────────
    if req.tools or ConversationManager.is_continuation(req):
        if request.app.state.conv_manager is None:
            raise OpenAIError(
                "tool calling requires MCP config management "
                "(AGI_MANAGE_MCP_CONFIG=true)",
                status_code=400, type="invalid_request_error", param="tools",
            )
        return await _handle_tools(req, request, model, workdir)

    cache: ConversationCache = request.app.state.conv_cache

    # System prompt → managed agent (--agent <slug>). The agent also carries
    # the plain-chat tools allowlist, so it is created even without a system
    # message. Restrictive modes fail closed if the plugin cannot be written.
    convo, system = split_system(req.messages)
    check_prompt_size(build_prompt(convo, system, inject_system=True), settings)
    agent = None
    if settings.system_prompt_via_agent or settings.plain_chat_tools != "all":
        agent = await request.app.state.agent_plugin.ensure_agent(
            system if settings.system_prompt_via_agent else None,
            tools_mode=settings.plain_chat_tools,
        )
    if agent is None and settings.plain_chat_tools != "all":
        raise OpenAIError(
            "cannot install the required tool-restricted agent",
            status_code=503, code="agent_unavailable",
        )

    # ── conversation-cache hit: resume with only the newest user message ─────
    hit = cache.lookup(req.messages)
    if hit is not None:
        prompt = (
            message_text_with_images(req.messages[-1], workdir)
            if settings.image_input else hit.new_user_text
        )
        conversation_id = hit.conversation_id
        logger.info("conv-cache hit → resuming agy conversation %s", conversation_id)
    else:
        if settings.image_input:
            convo = materialize_messages(convo, workdir)
        # No agent (mode off, or plugin dir unwritable) → automatic fallback:
        # the system message rides as an instruction preamble instead.
        prompt = build_prompt(convo, system, inject_system=(agent is None or not settings.system_prompt_via_agent))
        conversation_id = None

    if not prompt.strip():
        raise OpenAIError("no user content in messages", status_code=400, param="messages")
    check_prompt_size(prompt, settings)

    sess = AgySession(
        agy_bin=settings.agy_bin,
        model=model,
        workdir=workdir,
        prompt=prompt,
        conversation_id=conversation_id,
        agent=agent,
        print_timeout_s=settings.request_timeout_s,
        disable_slash_commands=settings.disable_slash_commands,
        timing_log=settings.timing_log,
        timing_label="autonomous",
    )
    await sess.start()

    cid = new_completion_id()
    created = now()
    timeout = settings.request_timeout_s

    # Thinking passthrough (always on): bound to the agy conversation upfront
    # on a resume, or from the init event on a fresh one (see app/thinking.py).
    brain = settings.agy_brain_path or default_brain_path()
    thinking = TurnThinking(brain, conversation_id)

    flatten = settings.flatten_markdown_tables
    if req.stream:
        return StreamingResponse(
            _stream(sess, cache, req, cid, model, created, timeout, flatten,
                    settings.timing_log, thinking),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    try:
        body = await _collect(sess, cache, req, cid, model, created, timeout, flatten,
                              settings.timing_log, thinking, request,
                              settings.disconnect_poll_s)
    except ClientDisconnected:
        logger.info("client disconnected mid-turn; agy subprocess terminated")
        # Starlette discards the response body anyway; 499 mirrors nginx.
        return JSONResponse(status_code=499, content=error_envelope(
            "client closed request", type="client_disconnected"))
    return JSONResponse(content=body)


async def _stream(
    sess: AgySession, cache: ConversationCache, req: ChatCompletionRequest,
    cid: str, model: str, created: int, timeout: float,
    flatten_tables: bool = True, timing: bool = False,
    thinking: TurnThinking | None = None,
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
            if thinking is not None:
                if isinstance(ev, Init):
                    thinking.on_init(ev.conversation_id)
                for block in thinking.poll():
                    yield sse(reasoning_chunk(cid, model, created, block))
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


async def _next_event_or_disconnect(
    sess: AgySession, request: Optional[Request], remaining: float, poll_s: float
):
    """``sess.next_event`` that also notices the client hanging up."""
    if request is None or poll_s <= 0:
        return await sess.next_event(timeout=remaining)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + remaining
    while True:
        slice_ = min(poll_s, deadline - loop.time())
        if slice_ <= 0:
            return None
        if await request.is_disconnected():
            raise ClientDisconnected()
        ev = await sess.next_event(timeout=slice_)
        if ev is not None:
            return ev


async def _collect(
    sess: AgySession, cache: ConversationCache, req: ChatCompletionRequest,
    cid: str, model: str, created: int, timeout: float,
    flatten_tables: bool = True, timing: bool = False,
    thinking: TurnThinking | None = None,
    request: Optional[Request] = None, disconnect_poll_s: float = 1.0,
) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    joiner = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "autonomous")
    completion_tokens = 0
    chunks: list[str] = []
    reasoning_parts: list[str] = []
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise OpenAIError("upstream timeout", status_code=504, type="timeout_error")
            ev = await _next_event_or_disconnect(sess, request, remaining, disconnect_poll_s)
            if thinking is not None:
                if isinstance(ev, Init):
                    thinking.on_init(ev.conversation_id)
                reasoning_parts.extend(thinking.poll())
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
                    cid, model, created, content=text, finish_reason="stop",
                    reasoning="\n\n".join(reasoning_parts) or None,
                )
            if isinstance(ev, TextDelta):
                timer.first_token()
                chunks.append(joiner.feed(ev.text))
            elif isinstance(ev, TurnDone):
                chunks.append(joiner.flush())
                if thinking is not None:
                    reasoning_parts.extend(thinking.poll())
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
                    reasoning="\n\n".join(reasoning_parts) or None,
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


# ── tool-passthrough path ───────────────────────────────────────────────────


async def _handle_tools(
    req: ChatCompletionRequest, request: Request, model: str, workdir
):
    mgr: ConversationManager = request.app.state.conv_manager

    if mgr.is_continuation(req):
        try:
            conv = await mgr.resume(req)
        except ExpiredContinuation as e:
            raise OpenAIError(
                str(e), status_code=409, type="invalid_request_error",
                code="tool_result_expired",
            )
    else:
        conv = await mgr.create(req, model=model, workdir=workdir)

    cid = new_completion_id()
    created = now()
    settings = get_settings()
    flatten = settings.flatten_markdown_tables
    timing = settings.timing_log

    if req.stream:
        return StreamingResponse(
            _tool_stream(mgr, conv, cid, model, created, flatten, timing),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    task = asyncio.create_task(_tool_collect(mgr, conv, cid, model, created, flatten, timing))
    try:
        while not task.done():
            if settings.disconnect_poll_s > 0 and await request.is_disconnected():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                await mgr._close(conv)
                return JSONResponse(status_code=499, content=error_envelope("client closed request"))
            await asyncio.wait({task}, timeout=max(0.05, settings.disconnect_poll_s))
        return JSONResponse(content=task.result())
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _tool_stream(mgr, conv, cid, model, created, flatten_tables: bool = True,
                       timing: bool = False):
    filt = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "tool")
    completion_tokens = 0
    try:
        yield sse(role_chunk(cid, model, created))
        async for ch in mgr.run_turn(conv):
            if isinstance(ch, TextChunk):
                out = filt.feed(ch.text)
                if out:
                    timer.first_token()
                    yield sse(text_chunk(cid, model, created, out))
            elif isinstance(ch, ReasoningChunk):
                yield sse(reasoning_chunk(cid, model, created, ch.text))
            elif isinstance(ch, ToolBoundaryChunk):
                filt.tool_boundary()
            elif isinstance(ch, ToolCallsChunk):
                tail = filt.flush()
                if tail:
                    yield sse(text_chunk(cid, model, created, tail))
                calls = [
                    {**pc.openai_tool_call(), "index": i} for i, pc in enumerate(ch.calls)
                ]
                yield sse(tool_calls_chunk(cid, model, created, calls))
                yield sse(finish_chunk(cid, model, created, "tool_calls"))
                break
            elif isinstance(ch, DoneChunk):
                tail = filt.flush()
                if tail:
                    yield sse(text_chunk(cid, model, created, tail))
                completion_tokens = (ch.usage or {}).get("completion_tokens", 0)
                yield sse(finish_chunk(cid, model, created, ch.finish_reason, ch.usage or None))
                break
            elif isinstance(ch, ErrorChunk):
                yield sse(error_envelope(ch.message, type="upstream_error"))
                break
        yield DONE
    finally:
        timer.done(completion_tokens)
        if conv.state == "running":
            await mgr._close(conv)


async def _tool_collect(mgr, conv, cid, model, created, flatten_tables: bool = True,
                        timing: bool = False) -> dict:
    filt = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "tool")
    completion_tokens = 0
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    try:
        async for ch in mgr.run_turn(conv):
            if isinstance(ch, TextChunk):
                if ch.text:
                    timer.first_token()
                text_parts.append(filt.feed(ch.text))
            elif isinstance(ch, ReasoningChunk):
                reasoning_parts.append(ch.text)
            elif isinstance(ch, ToolBoundaryChunk):
                filt.tool_boundary()
            elif isinstance(ch, ToolCallsChunk):
                text_parts.append(filt.flush())
                return completion_response(
                    cid, model, created,
                    content="".join(text_parts) or None,
                    finish_reason="tool_calls",
                    tool_calls=[pc.openai_tool_call() for pc in ch.calls],
                    reasoning="\n\n".join(reasoning_parts) or None,
                )
            elif isinstance(ch, DoneChunk):
                text_parts.append(filt.flush())
                completion_tokens = (ch.usage or {}).get("completion_tokens", 0)
                return completion_response(
                    cid, model, created,
                    content="".join(text_parts),
                    finish_reason=ch.finish_reason,
                    usage=ch.usage or None,
                    reasoning="\n\n".join(reasoning_parts) or None,
                )
            elif isinstance(ch, ErrorChunk):
                raise OpenAIError(ch.message, status_code=ch.status_code, type="upstream_error")
        raise OpenAIError("no response produced", status_code=502, type="upstream_error")
    finally:
        timer.done(completion_tokens)
