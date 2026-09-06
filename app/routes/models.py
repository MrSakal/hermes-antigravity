"""``GET /v1/models`` — advertise the Antigravity models this server exposes.

The list comes from the live catalog (``agy models``, see
:mod:`app.model_catalog`) when it has loaded, else from the static fallback
table. Both the OpenAI-friendly slugs (handy as a client ``default_model``) and
the exact display names are listed, so a client can pick either; the model used
per request is whatever the client sends, resolved through
``config.resolve_model``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import KNOWN_MODELS, _FAMILY_VARIANTS
from app.errors import OpenAIError
from app.openai_models import ModelCard, ModelList

router = APIRouter()


def _static_ids() -> list[str]:
    slugs: list[str] = []
    for family, variants in _FAMILY_VARIANTS.items():
        slugs.append(family)  # bare family → medium/high/low preference
        slugs.extend(f"{family}-{eff}" for eff in variants if eff)
    return slugs + KNOWN_MODELS


def model_ids(request: Request) -> list[str]:
    catalog = getattr(request.app.state, "model_catalog", None)
    if catalog is None or not catalog.loaded:
        return _static_ids()
    ids: list[str] = []
    seen: set[str] = set()
    for e in catalog.entries:
        fam = e.family
        if fam and fam not in seen:  # bare family slug first ("gemini-3.8-flash")
            seen.add(fam)
            ids.append(fam)
        if e.slug not in seen:
            seen.add(e.slug)
            ids.append(e.slug)
    ids.extend(e.display for e in catalog.entries)
    return ids


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    return ModelList(data=[ModelCard(id=mid) for mid in model_ids(request)])


@router.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str, request: Request) -> ModelCard:
    if model_id not in model_ids(request):
        raise OpenAIError(
            f"model {model_id!r} not found", status_code=404,
            type="invalid_request_error", param="model", code="model_not_found",
        )
    return ModelCard(id=model_id)
