"""``GET /v1/models`` — advertise the Antigravity models this server exposes.

We list OpenAI-friendly slugs (handy as a client ``default_model``) plus the
exact display names the `agy` CLI accepts, so a client can pick either; the
actual model used per request is whatever the client sends, resolved through
``config.resolve_model``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import KNOWN_MODELS
from app.openai_models import ModelCard, ModelList

router = APIRouter()

# Slugs first, then the CLI's canonical display names.
_SLUGS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-high",
    "gemini-3.1-pro",
    "gemini-3.1-pro-high",
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "gpt-oss-120b",
]

_MODEL_IDS = _SLUGS + KNOWN_MODELS


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=mid) for mid in _MODEL_IDS])


@router.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str) -> ModelCard:
    return ModelCard(id=model_id)
