"""FastAPI app factory + uvicorn entrypoint.

The app exposes an OpenAI-compatible surface (``/v1/models``,
``/v1/chat/completions``) backed by the Antigravity CLI (`agy`). Phase 1 serves
the autonomous (no-tool) path; the in-process MCP bridge for tool passthrough is
layered on in Phase 2.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncIterator
import contextlib

from fastapi import FastAPI, Request

from app.config import get_settings
from app.convcache import ConversationCache
from app.errors import OpenAIError
from app.routes import chat, compat, health, models

logger = logging.getLogger("agi")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("antigravity-interface starting on %s:%d", settings.host, settings.port)
    logger.info("default model=%s workdir=%s", settings.default_model, settings.default_workdir)
    yield
    logger.info("antigravity-interface shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    # Safety interlock: the server drives the agy CLI with
    # --dangerously-skip-permissions, so an open, network-reachable bind is
    # remote code execution for anyone who can reach it. Refuse to start on a
    # non-loopback host unless an api_key is set to gate /v1 requests.
    if not settings.is_loopback_host() and not settings.api_key:
        raise RuntimeError(
            f"refusing to start: host={settings.host!r} is not loopback and no "
            "api_key is set — set AGI_API_KEY to require a bearer token, or bind "
            "to 127.0.0.1"
        )

    app = FastAPI(title="antigravity-interface", lifespan=lifespan)
    app.state.settings = settings
    app.state.conv_cache = ConversationCache(
        max_entries=settings.conversation_cache_max,
        ttl_s=settings.conversation_cache_ttl_s,
    )

    @app.exception_handler(OpenAIError)
    async def _openai_error_handler(_: Request, exc: OpenAIError):  # type: ignore[unused-ignore]
        return exc.json_response()

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        # When an api_key is configured, every /v1 request must carry a matching
        # bearer token.
        key = settings.api_key
        if key and request.url.path.startswith("/v1"):
            header = request.headers.get("authorization", "")
            token = header[7:] if header[:7].lower() == "bearer " else ""
            if not hmac.compare_digest(token, key):
                return OpenAIError(
                    "missing or invalid api key",
                    status_code=401,
                    type="invalid_request_error",
                    code="invalid_api_key",
                ).json_response()
        return await call_next(request)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(compat.router)
    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint (``antigravity-interface``)."""
    import sys

    import uvicorn

    settings = get_settings()
    kwargs = {}
    if sys.platform != "win32":
        kwargs = {"loop": "uvloop", "http": "httptools"}
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        **kwargs,
    )


if __name__ == "__main__":
    run()
