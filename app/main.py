"""FastAPI app factory + uvicorn entrypoint.

The app exposes an OpenAI-compatible surface (``/v1/models``,
``/v1/chat/completions``) backed by the Antigravity CLI (`agy`), and mounts the
in-process MCP bridge at ``/mcp`` so agy can reach hermes' functions. Each
tool-turn conversation declares its server (``/mcp/<conv_id>``) in the managed
agent file it boots with (see :mod:`app.agent_config`); the shared Antigravity
MCP config is never written. The lifespan hook owns the MCP session manager's
task group, the ConversationManager + GC loop, and the one-time migration of
older builds' leftovers (legacy per-prompt agent files, shared-config slots).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request

from app.agent_config import AgentPluginManager
from app.config import KNOWN_MODELS, get_settings
from app.conversation import ConversationManager
from app.errors import OpenAIError
from app.mcp_bridge import McpBridge
from app.mcp_config import cleanup_legacy_mcp_entries
from app.model_catalog import ModelCatalog
from app.resources import Admission, AttachmentStore, Metrics
from app.request_guard import RequestGuard
from app.routes import chat, compat, health, models
from app.warmpool import WarmPool

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
    attachment_task = asyncio.create_task(app.state.attachments.gc_loop(), name="agi-attachments")
    if settings.model_catalog_refresh_s > 0:
        catalog_task = asyncio.create_task(catalog.refresh_loop(), name="agi-model-catalog")
    pool: WarmPool = app.state.warm_pool
    pool_task = asyncio.create_task(_pool_gc_loop(pool, settings.gc_interval_s), name="agi-warmpool-gc") if pool.enabled else None
    # Migration from older builds: per-prompt agent files and hermes/hermesN
    # slots they left in the shared MCP config (backed up once; never ours to
    # write again).
    app.state.agent_plugin.cleanup_legacy()
    if settings.manage_mcp_config:
        await asyncio.to_thread(cleanup_legacy_mcp_entries, settings.mcp_config_path)
    async with mcp.lifespan():
        manager: ConversationManager = app.state.conv_manager
        gc_task = asyncio.create_task(manager.gc_loop(), name="agi-gc")
        try:
            yield
        finally:
            for task in (gc_task, catalog_task, attachment_task, pool_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            await pool.stop()
            await manager.close_all()
            logger.info("antigravity-interface shutting down")


async def _pool_gc_loop(pool: WarmPool, interval_s: int) -> None:
    while True:
        await asyncio.sleep(max(1, interval_s))
        await pool.gc()


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
    app.state.metrics = Metrics()
    app.state.admission = Admission(settings.max_sessions, settings.max_queued_requests,
                                    settings.admission_timeout_s, app.state.metrics)
    app.state.attachments = AttachmentStore(settings)
    agent_plugin = AgentPluginManager(
        settings.agent_plugin_path, start_timeout_s=settings.agent_start_timeout_s,
        agent_name=settings.agent_name,
    )
    app.state.agent_plugin = agent_plugin
    manager = ConversationManager(mcp, settings, agents=agent_plugin)
    # Pooled entries pre-mint their conv_id and bake /mcp/<conv_id> into the
    # agent file, so the pool shares the manager's MCP bridge and URL scheme.
    manager.pool = WarmPool(settings, settings.warm_pool_size,
                            max_idle_s=settings.warm_pool_max_idle_s,
                            agents=agent_plugin, mcp=mcp, mcp_url=manager._mcp_url)
    app.state.warm_pool = manager.pool
    app.state.mcp = mcp
    app.state.model_catalog = ModelCatalog(
        settings.agy_bin, refresh_interval_s=settings.model_catalog_refresh_s
    )
    app.state.conv_manager = manager

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
    app.add_middleware(RequestGuard, settings=settings, metrics=app.state.metrics)
    app.add_middleware(_BearerAuth, api_key=settings.api_key)
    return app


class _BearerAuth:
    def __init__(self, app, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == "/mcp" or path.startswith("/mcp/")):
            client = scope.get("client")
            try:
                local = bool(client) and ipaddress.ip_address(client[0]).is_loopback
            except ValueError:
                local = False
            if not local:
                await OpenAIError("MCP is loopback-only", status_code=403,
                                  code="local_only").json_response()(scope, receive, send)
                return
        if (
            scope["type"] == "http"
            and self.api_key
            and (path.startswith(("/v1", "/api")) or path in {"/readyz", "/metrics"})
        ):
            header = ""
            for k, v in scope.get("headers") or ():
                if k == b"authorization":
                    header = v.decode("latin-1")
                    break
            token = header[7:] if header[:7].lower() == "bearer " else ""
            if not hmac.compare_digest(token.encode("utf-8"), self.api_key.encode("utf-8")):
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
