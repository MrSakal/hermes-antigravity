"""Per-model token limits from Antigravity's own backend (no hardcoded table).

``agy models`` only prints ``<slug>\\t<Display Name>``. The limits the CLI
itself enforces come from the Cloud Code ``fetchAvailableModels`` RPC, which
returns ``maxTokens`` (input context window) and ``maxOutputTokens`` per model
(verified live 2026-09-23 against daily-cloudcode-pa, agy 1.1.x). We call the
same RPC with the CLI's own OAuth token and expose the numbers on
``/v1/models`` as ``context_length`` / ``max_completion_tokens`` so OpenAI-style
clients (hermes' ``model_metadata``) detect the real window instead of guessing.

agy slugs do not always equal backend ids: ``gemini-3.8-flash-high`` is served
by ``gemini-3.8-flash-tiered``. Resolution order: exact id, then matching
``displayName``, then ``<family>-tiered``. Unresolvable slugs get no limits
(the client falls back to its own detection) — never an invented number.

Every failure is soft: missing token file, 401/403, network errors or an
unexpected payload keep the previous limits and are logged once per refresh.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from app.model_catalog import ModelEntry

logger = logging.getLogger("agi.limits")


@dataclass(frozen=True)
class ModelLimits:
    context_length: Optional[int]
    max_output_tokens: Optional[int]


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def parse_available_models(payload: Any) -> dict[str, dict[str, Any]]:
    """``{backend_id: model_dict}`` from a fetchAvailableModels response."""
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        raise ValueError("fetchAvailableModels payload has no 'models' object")
    return {k: v for k, v in models.items() if isinstance(k, str) and isinstance(v, dict)}


def resolve_limits(entry: ModelEntry, backend: dict[str, dict[str, Any]]) -> Optional[ModelLimits]:
    """Match one agy catalog entry to its backend model; None when ambiguous/absent."""
    match = backend.get(entry.slug)
    if match is None:
        by_display = [m for m in backend.values() if m.get("displayName") == entry.display]
        if len(by_display) == 1:
            match = by_display[0]
    if match is None and entry.family:
        match = backend.get(f"{entry.family}-tiered")
    if match is None:
        return None
    limits = ModelLimits(_positive_int(match.get("maxTokens")), _positive_int(match.get("maxOutputTokens")))
    return limits if limits.context_length or limits.max_output_tokens else None


class ModelLimitsProvider:
    """Caches backend limits keyed by agy slug; refreshed alongside the catalog."""

    def __init__(
        self,
        *,
        endpoint: str,
        token_path: Path,
        project_path: Path,
        timeout_s: float = 20.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.endpoint = endpoint
        self.token_path = token_path
        self.project_path = project_path
        self.timeout_s = timeout_s
        self._transport = transport
        self._limits: dict[str, ModelLimits] = {}
        self._slugs: frozenset[str] = frozenset()
        self.last_success_at: Optional[float] = None
        self.last_error: Optional[str] = None

    def get(self, slug: str) -> Optional[ModelLimits]:
        return self._limits.get(slug)

    @property
    def loaded(self) -> bool:
        return self.last_success_at is not None

    def _access_token(self) -> str:
        doc = json.loads(self.token_path.read_text(encoding="utf-8"))
        token = (doc.get("token") or {}).get("access_token") if isinstance(doc, dict) else None
        if not token:
            raise ValueError("no access_token in agy OAuth token file")
        return token

    def _project(self) -> Optional[str]:
        try:
            value = self.project_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    async def refresh(self, entries: list[ModelEntry], *, force: bool = False) -> bool:
        """Fetch limits for ``entries``; never raises. True when updated.

        Limits are near-static, so the RPC only runs when nothing is loaded yet
        (startup, or retry after a failed fetch) or the catalog's slug set
        changed (a model was added/removed). Otherwise it is a no-op.
        """
        if not entries:
            return False
        slugs = frozenset(e.slug for e in entries)
        if not force and self.loaded and slugs == self._slugs:
            return False
        try:
            token = self._access_token()
            body: dict[str, Any] = {}
            project = self._project()
            if project:
                body["project"] = project
            async with httpx.AsyncClient(timeout=self.timeout_s, transport=self._transport) as client:
                resp = await client.post(
                    self.endpoint,
                    json=body,
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "antigravity"},
                )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            backend = parse_available_models(resp.json())
        except Exception as exc:  # soft failure: keep previous limits
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("model limits: fetchAvailableModels failed (%s); keeping %d cached entries",
                           self.last_error, len(self._limits))
            return False
        resolved = {e.slug: lim for e in entries if (lim := resolve_limits(e, backend)) is not None}
        missing = [e.slug for e in entries if e.slug not in resolved]
        if missing:
            logger.info("model limits: no backend limits for %s", ", ".join(missing))
        self._limits = resolved
        self._slugs = slugs
        self.last_success_at = time.time()
        self.last_error = None
        logger.info("model limits: %d/%d models resolved", len(resolved), len(entries))
        return True
