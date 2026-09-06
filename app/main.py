"""FastAPI app factory + uvicorn entrypoint.

The app exposes an OpenAI-compatible surface (``/v1/models``,
``/v1/chat/completions``) backed by the Antigravity CLI (`agy`), and mounts the
in-process MCP bridge at ``/mcp`` so agy can reach hermes' functions. Each
tool-turn conversation registers its own entry in the shared Antigravity MCP
config pointing at ``/mcp/<conv_id>`` (see :mod:`app.mcp_config`). The lifespan
hook owns the MCP session manager's task group, the ConversationManager + GC
loop, and stale-slot cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request

from app.agent_config import AgentPluginManager
from app.config import KNOWN_MODELS, get_settings
from app.convcache import ConversationCache
from app.conversation import ConversationManager
from app.errors import OpenAIError
from app.mcp_bridge import McpBridge
from app.mcp_config import McpConfigManager
from app.model_catalog import ModelCatalog
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

    mcp: McpBridge = app.state.mcp
    catalog: ModelCatalog = app.state.model_catalog
    # Load the live model list before serving; failures keep the static table.
    if await catalog.refresh():
        logger.info("model catalog loaded from `agy models` (%d models)", len(catalog.entries))
    else:
        logger.warning("model catalog: using static fallback table (%d models)", len(KNOWN_MODELS))
    catalog_task = None
    if settings.model_catalog_refresh_s > 0:
        catalog_task = asyncio.create_task(catalog.refresh_loop(), name="agi-model-catalog")
    async with mcp.lifespan():
        manager: ConversationManager | None = app.state.conv_manager
        gc_task = None
        if manager is not None:
            await manager.mcp_config.cleanup_stale(settings.port)
            gc_task = asyncio.create_task(manager.gc_loop(), name="agi-gc")
        try:
            yield
        finally:
            for task in (gc_task, catalog_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            if manager is not None:
                await manager.close_all()
                await manager.mcp_config.remove_all()
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

    mcp = McpBridge()

    app = FastAPI(title="antigravity-interface", lifespan=lifespan)
    app.state.settings = settings
    app.state.mcp = mcp
    app.state.conv_cache = ConversationCache(
        max_entries=settings.conversation_cache_max,
        ttl_s=settings.conversation_cache_ttl_s,
    )
    app.state.model_catalog = ModelCatalog(
        settings.agy_bin, refresh_interval_s=settings.model_catalog_refresh_s
    )
    agent_plugin = AgentPluginManager(
        settings.agent_plugin_path, max_agents=settings.agent_cache_max
    )
    app.state.agent_plugin = agent_plugin
    if settings.manage_mcp_config:
        mcp_config = McpConfigManager(settings.mcp_config_path)
        app.state.conv_manager = ConversationManager(
            mcp, mcp_config, settings, agents=agent_plugin
        )
    else:
        app.state.conv_manager = None

    @app.exception_handler(OpenAIError)
    async def _openai_error_handler(_: Request, exc: OpenAIError):  # type: ignore[unused-ignore]
        return exc.json_response()

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(compat.router)

    # Mount the in-process MCP server; the dispatcher reads conv_id from the path.
    app.mount("/mcp", mcp.asgi_app())

    # Bearer auth as a PURE ASGI middleware (not @app.middleware("http") /
    # BaseHTTPMiddleware): the Starlette base class wraps `receive` in a task
    # group, which makes Request.is_disconnected() / a receive()-based
    # disconnect watch downstream always see "still connected". The MCP mount
    # is intentionally exempt — it is reached only by the local agy subprocess
    # over loopback and carries no token.
    app.add_middleware(_BearerAuth, api_key=settings.api_key)
    return app


class _BearerAuth:
    def __init__(self, app, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope["type"] == "http"
            and self.api_key
            and (scope.get("path") or "").startswith("/v1")
        ):
            header = ""
            for k, v in scope.get("headers") or ():
                if k == b"authorization":
                    header = v.decode("latin-1")
                    break
            token = header[7:] if header[:7].lower() == "bearer " else ""
            if not hmac.compare_digest(token, self.api_key):
                response = OpenAIError(
                    "missing or invalid api key",
                    status_code=401,
                    type="invalid_request_error",
                    code="invalid_api_key",
                ).json_response()
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


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
