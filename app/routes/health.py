"""Health / service-info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "service": "antigravity-interface", "version": __version__}


@router.get("/")
async def root(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "service": "antigravity-interface",
        "version": __version__,
        "openai_base_url": f"http://{settings.host}:{settings.port}/v1",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/healthz"],
    }
