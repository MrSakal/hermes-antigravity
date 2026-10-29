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

from typing import Optional

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


def advertised_model_ids(request: Request) -> list[str]:
    """What ``/v1/models`` lists: one id per model (``AGI_ADVERTISE_MODELS=
    families``, default) or the legacy superset (``all``). Multi-variant
    models appear as their family slug; single-variant ones as agy's slug."""
    settings = request.app.state.settings
    if settings.advertise_models == "all":
        return model_ids(request)
    catalog = getattr(request.app.state, "model_catalog", None)
    families = catalog.families() if catalog is not None and catalog.loaded else _FAMILY_VARIANTS
    # Family keys are agy's slugs minus the effort suffix; for a
    # single-variant model the key IS its slug ("" variant).
    return list(dict.fromkeys(families))


def model_limits(model_id: str, request: Request) -> tuple[Optional[int], Optional[int]]:
    """``(context_length, max_completion_tokens)`` for any accepted id.

    A family id (``gemini-3.8-flash``) covers several agy variants; the
    advertised limit is the smallest across them, so a client never assumes
    more room than the effort it picks actually has. Unknown → ``(None, None)``.
    """
    provider = getattr(request.app.state, "model_limits", None)
    catalog = getattr(request.app.state, "model_catalog", None)
    if provider is None or catalog is None or not catalog.loaded:
        return None, None
    entries = catalog.entries
    slugs = [e.slug for e in entries if model_id in (e.slug, e.family, e.display)]
    if not slugs:  # short aliases ("flash", "argon"): resolve like a request would
        settings = request.app.state.settings.model_copy(update={"strict_model": True})
        try:
            display = resolve_model(model_id, settings, known_models=catalog.display_names(),
                                    families=catalog.families())
        except UnknownModel:
            return None, None
        slugs = [e.slug for e in entries if e.display == display]
    found = [lim for s in slugs if (lim := provider.get(s)) is not None]
    ctx = [l.context_length for l in found if l.context_length]
    out = [l.max_output_tokens for l in found if l.max_output_tokens]
    return (min(ctx) if ctx else None, min(out) if out else None)


def model_card(model_id: str, request: Request) -> ModelCard:
    ctx, out = model_limits(model_id, request)
    return ModelCard(id=model_id, context_length=ctx, max_completion_tokens=out)


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


@router.get("/v1/models", response_model=ModelList, response_model_exclude_none=True)
async def list_models(request: Request) -> ModelList:
    return ModelList(data=[model_card(mid, request) for mid in advertised_model_ids(request)])


@router.get("/v1/models/{model_id}", response_model=ModelCard, response_model_exclude_none=True)
async def get_model(model_id: str, request: Request) -> ModelCard:
    if not known_model(model_id, request):
        raise OpenAIError(
            f"model {model_id!r} not found", status_code=404,
            type="invalid_request_error", param="model", code="model_not_found",
        )
    return model_card(model_id, request)
