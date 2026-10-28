"""``POST /v1/chat/completions`` — one path for every request.

Every request — with or without ``tools`` — goes through the
:class:`~app.conversation.ConversationManager`: a continuation (trailing
``tool`` results) resumes its suspended agy process (or rebuilds it from the
transcript); anything else reuses an IDLE process whose history matches,
adopts a warm-pool process, or spawns fresh. The manager's turn loop yields one chunk vocabulary (text / reasoning /
tool calls / done / error); this module only renders it, as SSE or as one JSON
completion, so streaming and non-streaming cannot drift apart.

Plain chat (no ``tools``) runs under the ``plain_chat_tools`` builtin allowlist
(default: read only) — this path answers hermes' side calls on untrusted
content. Tool turns get the full builtin set plus their MCP server.

Client disconnects: a streaming response is cancelled by Starlette when the
client goes away — the turn loop sees the cancellation and kills the process
(agy has no per-turn cancel). A non-streaming request is not cancelled, so the
turn runs in a task while the handler polls ``request.is_disconnected()`` and
cancels it. Once a conversation is claimed, ANY exit before the turn loop
finished (disconnect at the first SSE byte, an error) is closed by the
request-scoped cleanup, pinned to this turn's token.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent_config import SpawnTimeout
from app.agy_session import AgySession  # noqa: F401  (tests spy on it via this module)
from app.config import (
    PromptTooLarge,
    Settings,
    UnknownModel,
    ensure_workdir,
    get_settings,
    resolve_model,
)
from app.config import check_prompt_size as _check_prompt_size
from app.conversation import (
    Conversation,
    ConversationManager,
    DoneChunk,
    ErrorChunk,
    ExpiredContinuation,
    ReasoningChunk,
    TextChunk,
    ToolCallsChunk,
)
from app.errors import OpenAIError, error_envelope
from app.model_catalog import ModelCatalog
from app.openai_models import ChatCompletionRequest
from app.resources import validate_request
from app.timing import TurnTimer
from app.translate import (
    DONE,
    completion_response,
    finish_chunk,
    new_completion_id,
    now,
    reasoning_chunk,
    role_chunk,
    sse,
    text_chunk,
    tool_calls_chunk,
)

logger = logging.getLogger("agi.chat")

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _mark_first_token(request):
    if request is not None and getattr(request.state, 'first_token_at', None) is None:
        request.state.first_token_at = time.monotonic()


def _remaining(request, settings):
    started = getattr(request.state, 'request_started', time.monotonic())
    remaining = settings.request_timeout_s - (time.monotonic() - started)
    if remaining <= 0:
        raise OpenAIError('request deadline exceeded', status_code=504, type='timeout_error')
    return remaining


async def _deadline_stream(iterator, timeout):
    try:
        async with asyncio.timeout(timeout):
            async for chunk in iterator:
                yield chunk
    except TimeoutError:
        yield sse(error_envelope('request deadline exceeded', type='timeout_error'))
        yield DONE
    finally:
        await iterator.aclose()


def resolve_request_model(
    req: ChatCompletionRequest, settings: Settings, catalog: Optional[ModelCatalog]
) -> str:
    """``resolve_model`` against the live catalog, mapped to an OpenAI error."""
    known = catalog.display_names() if catalog and catalog.loaded else None
    fams = catalog.families() if catalog and catalog.loaded else None
    try:
        return resolve_model(
            req.model, settings, effort=req.effort or req.reasoning_effort or settings.default_effort,
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
            "(AGI_MAX_PROMPT_BYTES). Shorten the conversation.",
            status_code=400, type="invalid_request_error", code="context_length_exceeded",
        )


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    settings = get_settings()
    images = validate_request(req, settings)
    request.state.ignored_parameters = [key for key in (
        'temperature', 'top_p', 'max_tokens', 'max_completion_tokens', 'stop'
    ) if getattr(req, key) is not None]
    if not req.messages:
        raise OpenAIError("messages must not be empty", status_code=400, param="messages")
    mgr: ConversationManager = request.app.state.conv_manager
    continuation = mgr.is_continuation(req)
    if req.tool_choice == 'none':
        if continuation:
            raise OpenAIError('cannot disable tools on a suspended CLI turn',
                              param='tool_choice', code='unsupported_parameter')
        req = req.model_copy(update={'tools': None})
    model = resolve_request_model(req, settings, getattr(request.app.state, "model_catalog", None))
    try:
        workdir = settings.resolve_workdir(req.workdir)
    except ValueError as e:
        raise OpenAIError(str(e), status_code=400, param="workdir")
    if continuation:
        # A continuation without model/workdir targets its suspended
        # conversation's — not the server defaults.
        ids = [m.tool_call_id for m in mgr._trailing_tool_messages(req)]
        conv_id = mgr._pending_index.get(ids[0]) if ids else None
        existing = mgr._conversations.get(conv_id)
        if existing is not None:
            if req.workdir is None:
                workdir = existing.workdir or Path(existing.session.workdir)
            if req.model is None:
                model = existing.model
    await asyncio.to_thread(ensure_workdir, workdir)
    # No admission control: every new turn gets its own agy process now.
    attachment_release = request.app.state.attachments.pin_and_write(workdir, images)
    request.state.resource_releases.append(attachment_release)

    owns_releases = not continuation

    async def own_for_rebuild():
        nonlocal owns_releases
        owns_releases = True

    try:
        budget = _remaining(request, settings)
        # Spawns never queue (per-spawn agent files); only the request
        # deadline bounds them (504), never a capacity cap.
        if continuation:
            conv = await asyncio.wait_for(
                mgr.resume_or_rebuild(req, model=model, workdir=workdir,
                                      before_create=own_for_rebuild),
                timeout=budget)
        else:
            conv = await asyncio.wait_for(
                mgr.start_turn(req, model=model, workdir=workdir),
                timeout=budget)
    except ExpiredContinuation as e:
        raise OpenAIError(str(e), status_code=409, type="invalid_request_error",
                          code="tool_result_expired")
    except (TimeoutError, SpawnTimeout) as exc:
        raise OpenAIError('upstream startup timeout', status_code=504, type='timeout_error') from exc

    if owns_releases:
        # The attachment pin belongs to the turn now: held
        # while it is RUNNING or SUSPENDED, released when it parks or closes.
        releases = request.state.resource_releases
        request.state.resource_releases = []
        previous = conv.release_resources

        def release_resources():
            for release in reversed(releases):
                release()
            if previous is not None:
                previous()
        conv.release_resources = release_resources

    token = conv.turn_token

    async def abandon_if_running():
        # Claimed but never finished by this request (client gone before the
        # first byte, cancellation): kill it. Never touches a later turn.
        await mgr.abandon(conv, token)
    request.state.conversation_cleanup = abandon_if_running

    cid = new_completion_id()
    created = now()
    if req.stream:
        return StreamingResponse(
            _deadline_stream(_stream(mgr, conv, cid, model, created, settings.timing_log, request),
                             _remaining(request, settings)),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    task = asyncio.create_task(_collect(mgr, conv, cid, model, created, settings.timing_log, request))
    try:
        while not task.done():
            _remaining(request, settings)
            if settings.disconnect_poll_s > 0 and await request.is_disconnected():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                await mgr.abandon(conv, token)
                logger.info("client disconnected mid-turn; agy subprocess terminated")
                # Starlette discards the body anyway; 499 mirrors nginx.
                return JSONResponse(status_code=499, content=error_envelope(
                    "client closed request", type="client_disconnected"))
            await asyncio.wait({task}, timeout=max(0.05, settings.disconnect_poll_s))
        return JSONResponse(content=task.result())
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _error_type(status_code: int) -> str:
    return "timeout_error" if status_code == 504 else "upstream_error"


async def _stream(mgr: ConversationManager, conv: Conversation, cid: str, model: str,
                  created: int, timing: bool = False, request=None) -> AsyncIterator[str]:
    timer = TurnTimer(timing, "turn")
    completion_tokens = 0
    token = conv.turn_token
    try:
        # The role chunk is the first yield: a client that disconnects right
        # here closes this generator before run_turn started — the finally
        # below (and the request cleanup) still abandons the claimed turn.
        yield sse(role_chunk(cid, model, created))
        async with contextlib.aclosing(mgr.run_turn(conv)) as chunks:
            async for ch in chunks:
                if isinstance(ch, TextChunk):
                    if ch.text:
                        _mark_first_token(request)
                        timer.first_token()
                        yield sse(text_chunk(cid, model, created, ch.text))
                elif isinstance(ch, ReasoningChunk):
                    yield sse(reasoning_chunk(cid, model, created, ch.text))
                elif isinstance(ch, ToolCallsChunk):
                    calls = [
                        {**pc.openai_tool_call(), "index": i} for i, pc in enumerate(ch.calls)
                    ]
                    yield sse(tool_calls_chunk(cid, model, created, calls))
                    yield sse(finish_chunk(cid, model, created, "tool_calls"))
                    break
                elif isinstance(ch, DoneChunk):
                    completion_tokens = (ch.usage or {}).get("completion_tokens", 0)
                    yield sse(finish_chunk(cid, model, created, ch.finish_reason, ch.usage or None))
                    break
                elif isinstance(ch, ErrorChunk):
                    yield sse(error_envelope(ch.message, type=_error_type(ch.status_code)))
                    break
        yield DONE
    finally:
        timer.done(completion_tokens)
        # A no-op once the turn parked/suspended/closed — and pinned to this
        # turn: the client may already have claimed the next one.
        await mgr.abandon(conv, token)


async def _collect(mgr: ConversationManager, conv: Conversation, cid: str, model: str,
                   created: int, timing: bool = False, request=None) -> dict:
    timer = TurnTimer(timing, "turn")
    completion_tokens = 0
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    try:
        async with contextlib.aclosing(mgr.run_turn(conv)) as chunks:
            async for ch in chunks:
                if isinstance(ch, TextChunk):
                    if ch.text:
                        _mark_first_token(request)
                        timer.first_token()
                        text_parts.append(ch.text)
                elif isinstance(ch, ReasoningChunk):
                    reasoning_parts.append(ch.text)
                elif isinstance(ch, ToolCallsChunk):
                    return completion_response(
                        cid, model, created,
                        content="".join(text_parts) or None,
                        finish_reason="tool_calls",
                        tool_calls=[pc.openai_tool_call() for pc in ch.calls],
                        reasoning="\n\n".join(reasoning_parts) or None,
                    )
                elif isinstance(ch, DoneChunk):
                    completion_tokens = (ch.usage or {}).get("completion_tokens", 0)
                    return completion_response(
                        cid, model, created,
                        content="".join(text_parts),
                        finish_reason=ch.finish_reason,
                        usage=ch.usage or None,
                        reasoning="\n\n".join(reasoning_parts) or None,
                    )
                elif isinstance(ch, ErrorChunk):
                    raise OpenAIError(ch.message, status_code=ch.status_code,
                                      type=_error_type(ch.status_code))
        raise OpenAIError("no response produced", status_code=502, type="upstream_error")
    finally:
        timer.done(completion_tokens)
