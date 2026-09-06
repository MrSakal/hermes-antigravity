"""``GET /v1/models`` — advertise the Antigravity models this server exposes.

The list comes from the live catalog (``agy models``, see
:mod:`app.model_catalog`) when it has loaded, else from the static fallback
table. Discovery lists one CLI slug per variant; aliases remain valid inputs.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import _FAMILY_VARIANTS, resolve_model, UnknownModel
from app.errors import OpenAIError
from app.openai_models import ModelCard, ModelList

router = APIRouter()


def _static_ids() -> list[str]:
    return [f"{family}-{eff}" if eff else family
            for family, variants in _FAMILY_VARIANTS.items() for eff in variants]


def model_ids(request: Request) -> list[str]:
    catalog = getattr(request.app.state, "model_catalog", None)
    return list(dict.fromkeys(catalog.slugs())) if catalog is not None and catalog.loaded else _static_ids()


def known_model(model_id: str, request: Request) -> bool:
    """Use completion alias resolution, but always reject unknown probe IDs."""
    if not model_id or not model_id.strip():
        return False
    if model_id in model_ids(request):
        return True
    catalog = getattr(request.app.state, "model_catalog", None)
    kwargs = {}
    if catalog is not None and catalog.loaded:
        kwargs = {"known_models": catalog.display_names(), "families": catalog.families()}
    settings = request.app.state.settings.model_copy(update={"strict_model": True})
    try:
        resolve_model(model_id, settings, **kwargs)
    except UnknownModel:
        return False
    return True


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    return ModelList(data=[ModelCard(id=mid) for mid in model_ids(request)])


@router.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str, request: Request) -> ModelCard:
    if not known_model(model_id, request):
        raise OpenAIError(
            f"model {model_id!r} not found", status_code=404,
            type="invalid_request_error", param="model", code="model_not_found",
        )
    return ModelCard(id=model_id)
