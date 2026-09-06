"""Model catalog: what `agy models` reports, refreshed on a timer.

``agy models`` prints one ``<slug>\\t<Display Name>`` line per model (verified
against 1.1.27, ~2.4 s wall time). Both forms are accepted by ``--model``. The
catalog is loaded at startup and refreshed in the background so new model
releases (e.g. Gemini 3.8 Flash, which appeared while the static table still
ended at 3.7) become visible on ``/v1/models`` and resolvable without a code
change. When the CLI cannot be queried (not installed, not signed in, offline)
the static table from :mod:`app.config` keeps everything working.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("agi.models")

_LINE_RE = re.compile(r"^(?P<slug>[A-Za-z0-9][A-Za-z0-9._-]*)\t(?P<display>.+?)\s*$")
_EFFORT_SUFFIX_RE = re.compile(r"-(low|medium|high)$")


@dataclass(frozen=True)
class ModelEntry:
    slug: str  # e.g. "gemini-3.8-flash-medium"
    display: str  # e.g. "Gemini 3.8 Flash (Medium)"

    @property
    def family(self) -> Optional[str]:
        """Slug minus a trailing effort suffix ("gemini-3.8-flash"), or None
        for single-variant models ("claude-sonnet-4-6")."""
        m = _EFFORT_SUFFIX_RE.search(self.slug)
        return self.slug[: m.start()] if m else None

    @property
    def effort(self) -> Optional[str]:
        m = _EFFORT_SUFFIX_RE.search(self.slug)
        return m.group(1) if m else None


def parse_models_output(text: str) -> list[ModelEntry]:
    """Parse ``agy models`` stdout. Tolerates the ``Fetching available
    models...`` banner and any other non-matching line."""
    out: list[ModelEntry] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = _LINE_RE.match(line.rstrip("\r"))
        if not m:
            continue
        slug = m.group("slug")
        if slug in seen:
            continue
        seen.add(slug)
        out.append(ModelEntry(slug=slug, display=m.group("display")))
    return out


class ModelCatalog:
    """Live view of the models the local agy install can use."""

    def __init__(
        self,
        agy_bin: str,
        *,
        refresh_interval_s: int = 3600,
        query_timeout_s: float = 20.0,
    ) -> None:
        self.agy_bin = agy_bin
        self.refresh_interval_s = refresh_interval_s
        self.query_timeout_s = query_timeout_s
        self._entries: list[ModelEntry] = []
        self._loaded_at: Optional[float] = None
        self._lock = asyncio.Lock()
        self.last_attempt_at: Optional[float] = None
        self.last_success_at: Optional[float] = None
        self.last_error: Optional[str] = None

    # ── read side ────────────────────────────────────────────────────────────

    @property
    def entries(self) -> list[ModelEntry]:
        return list(self._entries)

    @property
    def loaded(self) -> bool:
        return self._loaded_at is not None

    @property
    def age_s(self) -> Optional[float]:
        return None if self._loaded_at is None else max(0.0, time.monotonic() - self._loaded_at)

    @property
    def status(self) -> str:
        if not self.loaded:
            return "fallback"
        return "stale" if self.last_error or self.age_s > max(60, self.refresh_interval_s) else "fresh"

    def display_names(self) -> list[str]:
        return [e.display for e in self._entries]

    def slugs(self) -> list[str]:
        return [e.slug for e in self._entries]

    def families(self) -> dict[str, dict[str, str]]:
        """``{family_slug: {effort: display}}`` for multi-variant models and
        ``{slug: {"": display}}`` for single-variant ones — the same shape as
        ``config._FAMILY_VARIANTS`` so ``resolve_model`` can consume either."""
        fam: dict[str, dict[str, str]] = {}
        for e in self._entries:
            if e.family and e.effort:
                fam.setdefault(e.family, {})[e.effort] = e.display
            else:
                fam[e.slug] = {"": e.display}
        return fam

    # ── refresh ──────────────────────────────────────────────────────────────

    async def refresh(self) -> bool:
        """Run ``agy models`` once; returns True when the catalog was updated.
        Never raises — a failure keeps the previous (or static) catalog."""
        async with self._lock:
            self.last_attempt_at = time.time()
            proc: Optional[asyncio.subprocess.Process] = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.agy_bin, "models",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.query_timeout_s)
            except FileNotFoundError:
                self.last_error = "binary_not_found"
                logger.warning("model catalog: agy binary %r not found", self.agy_bin)
                return False
            except asyncio.TimeoutError:
                self.last_error = "timeout"
                logger.warning("model catalog: `agy models` timed out after %.0fs", self.query_timeout_s)
                if proc is not None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
                return False
            except asyncio.CancelledError:
                if proc is not None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
                raise
            except OSError as e:
                self.last_error = str(e)
                logger.warning("model catalog: cannot run `agy models`: %s", e)
                return False
            entries = parse_models_output(out.decode("utf-8", "replace"))
            if proc.returncode != 0 or not entries:
                self.last_error = f"exit={proc.returncode}; entries={len(entries)}"
                logger.warning(
                    "model catalog: `agy models` rc=%s entries=%d stderr=%r",
                    proc.returncode, len(entries), err.decode("utf-8", "replace")[:200],
                )
                return False
            if [e.slug for e in entries] != [e.slug for e in self._entries]:
                logger.info("model catalog: %d models: %s", len(entries), ", ".join(e.slug for e in entries))
            self._entries = entries
            self._loaded_at = time.monotonic()
            self.last_success_at = time.time()
            self.last_error = None
            return True

    async def refresh_loop(self) -> None:
        interval = max(60, self.refresh_interval_s)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("model catalog refresh failed")
