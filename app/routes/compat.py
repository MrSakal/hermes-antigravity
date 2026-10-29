"""Compatibility shims for Ollama / llama.cpp-aware clients.

Frontends like Open WebUI probe a handful of non-OpenAI endpoints to detect
server capabilities (``/api/tags``, ``/api/show``, ``/v1/props``, ``/version`` …).
Without them the client falls back to a degraded mode or refuses the server.

These handlers reuse the OpenAI model list from :mod:`app.routes.models` so the
advertised models stay in one place; nothing here is hard-coded per model.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from app.routes.models import advertised_model_ids, known_model, model_card, model_limits

router = APIRouter()


def _app_version() -> str:
    """Server version from installed package metadata, with a sane fallback."""
    for dist in ("antigravity-interface", "antigravity_interface"):
        try:
            return _pkg_version(dist)
        except PackageNotFoundError:
            continue
    return "0.1.0"


async def _model_ids(request: Request) -> list[str]:
    """The advertised model ids, sourced from the OpenAI ``/v1/models`` handler."""
    return advertised_model_ids(request)


# Static "details" block reused by /api/tags and /api/show. Values are nominal —
# clients only need the shape to be present and well-formed.
_DETAILS = {
    "parent_model": "",
    "format": "gguf",
    "family": "antigravity",
    "families": ["antigravity"],
    "parameter_size": "",
    "quantization_level": "",
}


# ── Ollama: list models ───────────────────────────────────────────────────────


@router.get("/api/tags")
async def api_tags(request: Request) -> dict:
    ids = await _model_ids(request)
    return {
        "models": [
            {
                "name": mid,
                "model": mid,
                "modified_at": "1970-01-01T00:00:00Z",
                "size": 0,
                "digest": "",
                "details": dict(_DETAILS),
            }
            for mid in ids
        ]
    }


# ── Ollama: show model ────────────────────────────────────────────────────────


class ShowRequest(BaseModel):
    model_config = {"extra": "ignore"}
    name: str | None = None
    model: str | None = None


@router.post("/api/show")
async def api_show(body: ShowRequest, request: Request, response: Response) -> dict:
    requested = body.model or body.name
    if not requested or not known_model(requested, request):
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"error": f"model {requested!r} not found"}
    ctx, _out = model_limits(requested, request)
    model_info: dict = {
        "general.architecture": "antigravity",
        "general.basename": requested,
    }
    if ctx:
        model_info["antigravity.context_length"] = ctx
    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": dict(_DETAILS),
        "model_info": model_info,
        # What the server actually supports for every model: OpenAI tool
        # calling (MCP bridge), reasoning passthrough, data-URL image input.
        "capabilities": ["completion", "tools", "thinking", "vision"],
    }


# ── Ollama / llama.cpp: version ──────────────────────────────────────────────


@router.get("/api/version")
async def api_version() -> dict:
    return {"version": _app_version()}


@router.get("/version")
async def version() -> dict:
    return {"version": _app_version()}


# ── llama.cpp: server props ──────────────────────────────────────────────────


async def _props(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "default_generation_settings": {
            "n_ctx": 0,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "total_slots": 1,
        "model_path": settings.default_model,
        "chat_template": "",
    }


@router.get("/v1/props")
async def v1_props(request: Request) -> dict:
    return await _props(request)


@router.get("/props")
async def props(request: Request) -> dict:
    return await _props(request)


# ── Ollama-style alias for the OpenAI model list ─────────────────────────────


@router.get("/api/v1/models")
async def api_v1_models(request: Request) -> dict:
    """OpenAI ``data`` list plus LM Studio's native ``models`` list.

    LM Studio serves this path natively, so clients that fingerprint servers
    (hermes' ``detect_local_server_type``) treat any 200 here as LM Studio and
    read limits from ``models[].loaded_instances[].config.context_length`` /
    ``models[].max_completion_tokens``. Both shapes carry the same ids."""
    cards = [model_card(mid, request) for mid in advertised_model_ids(request)]
    lmstudio = []
    for c in cards:
        m: dict = {"type": "llm", "key": c.id, "id": c.id, "display_name": c.id,
                   "publisher": "google", "loaded_instances": []}
        if c.context_length:
            m["max_context_length"] = c.context_length
            m["loaded_instances"] = [{"id": c.id, "config": {"context_length": c.context_length}}]
        if c.max_completion_tokens:
            m["max_completion_tokens"] = c.max_completion_tokens
        lmstudio.append(m)
    return {"object": "list", "data": [c.model_dump(exclude_none=True) for c in cards], "models": lmstudio}
