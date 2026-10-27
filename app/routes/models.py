"""``GET /v1/models`` — advertise the Antigravity models this server exposes.

The list comes from the live catalog (``agy models``, see
:mod:`app.model_catalog`) when it has loaded, else from the static fallback
table. It advertises every id a client can send, in this order: one CLI slug
per variant (agy's own order, so a client picking the first id gets a slug),
then the family slugs of multi-variant models (``gemini-4-argon``, effort
picked by ``effort``/``reasoning_effort``), then the display names agy itself
prints (``Gemini 4 Argon (Medium)``). Short aliases (``argon``, ``flash``, …)
remain valid inputs without being listed.
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
    if catalog is not None and catalog.loaded:
        slugs, families, displays = catalog.slugs(), catalog.families(), catalog.display_names()
    else:
        slugs, families = _static_ids(), _FAMILY_VARIANTS
        displays = [d for variants in families.values() for d in variants.values()]
    multi = [family for family, variants in families.items() if "" not in variants]
    return list(dict.fromkeys([*slugs, *multi, *displays]))


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
