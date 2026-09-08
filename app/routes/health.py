"""Health / service-info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import shutil

from app import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "service": "antigravity-interface", "version": __version__}


@router.get('/readyz')
async def readyz(request: Request):
    state = request.app.state
    catalog = state.model_catalog
    executable = shutil.which(state.settings.agy_bin) is not None
    ready = bool(catalog.loaded and executable)
    data = {'ready': ready, 'cli_available': executable,
            'catalog_loaded': bool(catalog.loaded), 'models': len(catalog.entries),
            'inference_probe': False}
    data['catalog'] = {'status': catalog.status, 'age_s': catalog.age_s,
                       'last_success_at': catalog.last_success_at,
                       'last_attempt_at': catalog.last_attempt_at,
                       'last_error': catalog.last_error}
    return JSONResponse(data, status_code=200 if ready else 503)


@router.get('/metrics')
async def metrics(request: Request):
    state = request.app.state
    data = state.metrics.snapshot()
    mgr = state.conv_manager
    conversations = list(mgr._conversations.values()) if mgr else []
    data.update(sessions_active=state.admission.active,
                sessions_capacity=state.admission.limit,
                sessions_waiting=state.admission.waiting,
                sessions_suspended=sum(c.state == 'suspended' for c in conversations),
                tool_stdout_queue_depth=sum(c.session._queue.qsize() for c in conversations),
                mcp_transport_sessions=len(state.mcp.session_manager._server_instances),
                warm_pool_idle=state.warm_pool.idle,
                warm_pool_hits=state.warm_pool.hits,
                warm_pool_misses=state.warm_pool.misses,
                attachment_workspaces_pinned=len(state.attachments.pins))
    return data


@router.get("/")
async def root(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "service": "antigravity-interface",
        "version": __version__,
        "openai_base_url": f"http://{settings.host}:{settings.port}/v1",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/healthz"],
    }
