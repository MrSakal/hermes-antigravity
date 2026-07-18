"""Runtime configuration (env-driven, prefix ``AGI_``) and model-alias resolution.

All knobs are read from the environment with an ``AGI_`` prefix (e.g.
``AGI_PORT=9000``). Defaults are chosen so the server runs out of the box against
a local Antigravity login (shared with the Antigravity IDE) with no API key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The model identifiers the `agy` CLI accepts on `--model` are EXACT display
# names (verified live against agy 1.1.4: slugs are rejected with a structured
# error listing these). `agy models` is the source of truth; this static list
# is the fallback/alias basis and should track it.
KNOWN_MODELS = [
    "Gemini 3.5 Flash (Medium)",
    "Gemini 3.5 Flash (High)",
    "Gemini 3.5 Flash (Low)",
    "Gemini 3.1 Pro (Low)",
    "Gemini 3.1 Pro (High)",
    "Claude Sonnet 4.6 (Thinking)",
    "Claude Opus 4.6 (Thinking)",
    "GPT-OSS 120B (Medium)",
]

# OpenAI-friendly slug → display-name family. Effort (low/medium/high) selects
# the variant when the family has one; see resolve_model.
_FAMILY_VARIANTS: dict[str, dict[str, str]] = {
    "gemini-3.5-flash": {
        "low": "Gemini 3.5 Flash (Low)",
        "medium": "Gemini 3.5 Flash (Medium)",
        "high": "Gemini 3.5 Flash (High)",
    },
    "gemini-3.1-pro": {
        "low": "Gemini 3.1 Pro (Low)",
        "high": "Gemini 3.1 Pro (High)",
    },
    "claude-sonnet-4.6": {"": "Claude Sonnet 4.6 (Thinking)"},
    "claude-opus-4.6": {"": "Claude Opus 4.6 (Thinking)"},
    "gpt-oss-120b": {"": "GPT-OSS 120B (Medium)"},
}

# Extra aliases → family slug.
_ALIASES: dict[str, str] = {
    "gemini-flash": "gemini-3.5-flash",
    "flash": "gemini-3.5-flash",
    "gemini-pro": "gemini-3.1-pro",
    "pro": "gemini-3.1-pro",
    "claude-sonnet": "claude-sonnet-4.6",
    "sonnet": "claude-sonnet-4.6",
    "claude-opus": "claude-opus-4.6",
    "opus": "claude-opus-4.6",
    "gpt-oss": "gpt-oss-120b",
}


class Settings(BaseSettings):
    """Server settings. Override any field via an ``AGI_<FIELD>`` environment var."""

    model_config = SettingsConfigDict(env_prefix="AGI_", extra="ignore")

    # ── HTTP server ──────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    # 8788 by default so it can run next to hermes-claude's 8787.
    port: int = 8788

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Optional bearer token. When set, every /v1 request must carry
    # `Authorization: Bearer <api_key>`. Left unset, the server is open — which
    # is only safe on a loopback bind (see the startup guard in main.create_app).
    api_key: Optional[str] = None

    # ── Antigravity CLI ──────────────────────────────────────────────────────
    agy_bin: str = "agy"
    default_model: str = "Gemini 3.5 Flash (Medium)"
    default_effort: Optional[str] = None

    # ── Prompt handling ──────────────────────────────────────────────────────
    # agy takes the prompt as an argv argument (stdin piping is NOT supported in
    # 1.1.4 — verified live: `echo .. | agy -p` yields "no output produced").
    # Windows caps a command line at ~32,767 chars; stay safely below it.
    max_prompt_chars: int = 28000
    # Inject the request's system message as an instruction preamble at the top
    # of the folded first turn (agy has no --system-prompt / --append-system-prompt
    # flag). Documented deviation: Antigravity's own identity + builtin tools
    # remain active underneath.
    system_prompt_injection: bool = True

    # ── Conversation continuity ─────────────────────────────────────────────—
    # agy stores conversation history server-side; `--conversation <id>` resumes
    # it. We cache (history-prefix hash → agy conversation id) so a follow-up
    # request sends only the newest user message instead of re-folding the whole
    # transcript (cheaper, and immune to the argv length cap).
    conversation_cache_max: int = 256
    conversation_cache_ttl_s: int = 3600

    # ── Output formatting ──────────────────────────────────────────────────—
    # Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
    # clients (e.g. Discord) that don't support Markdown tables.
    flatten_markdown_tables: bool = True

    # ── Workspace ──────────────────────────────────────────────────────────—
    default_workdir: Path = Path("~/agi-workspace").expanduser()
    # Per-request `workdir` overrides must resolve under one of these roots.
    # Empty => only `default_workdir` is allowed.
    allowed_workdir_roots: list[str] = []

    # ── Lifecycle / timeouts (seconds) ─────────────────────────────────────—
    request_timeout_s: int = 600

    # ── Logging ────────────────────────────────────────────────────────────—
    log_level: str = "INFO"
    # When true, emit per-turn latency metrics (spawn_ms / ttft_ms / total_ms /
    # tok_per_s) to the "agi.timing" logger at INFO.
    timing_log: bool = False

    @field_validator("default_workdir", mode="before")
    @classmethod
    def _expand_workdir(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def is_loopback_host(self) -> bool:
        """True when the bind host only accepts connections from this machine."""
        h = self.host.strip().lower()
        return h in {"localhost", "::1", "0:0:0:0:0:0:0:1"} or h.startswith("127.")

    def resolved_workdir_roots(self) -> list[Path]:
        """The set of allowed workspace roots as resolved absolute paths."""
        roots = [self.default_workdir]
        roots.extend(Path(r).expanduser() for r in self.allowed_workdir_roots)
        return [p.resolve() for p in roots]

    def resolve_workdir(self, requested: Optional[str]) -> Path:
        """Resolve and validate a per-request workdir override.

        Returns ``default_workdir`` when no override is given. Raises
        ``ValueError`` if the override escapes every allowed root (traversal
        guard).
        """
        if not requested:
            return self.default_workdir
        candidate = Path(requested).expanduser().resolve()
        for root in self.resolved_workdir_roots():
            if candidate == root or root in candidate.parents:
                return candidate
        raise ValueError(
            f"workdir {requested!r} is not under an allowed root "
            f"({[str(r) for r in self.resolved_workdir_roots()]})"
        )


def resolve_model(
    requested: Optional[str], settings: Settings, effort: Optional[str] = None
) -> str:
    """Map an OpenAI ``model`` field (slug or display name) to an ``--model``
    value the `agy` CLI accepts.

    * An exact display name (case-insensitive) passes through as the canonical
      display name.
    * A known slug/alias resolves through its family; ``effort`` (or a
      ``-low``/``-medium``/``-high`` suffix on the slug) picks the variant when
      the family has effort variants.
    * Anything else falls back to ``default_model``.
    """
    if not requested:
        requested = ""
    r = requested.strip()

    # Exact display-name match (case-insensitive).
    for m in KNOWN_MODELS:
        if r.lower() == m.lower():
            return m

    slug = r.lower().replace(" ", "-")
    eff = (effort or "").strip().lower()
    # A trailing -low/-medium/-high on the slug overrides `effort`.
    for suffix in ("low", "medium", "high"):
        if slug.endswith("-" + suffix):
            slug = slug[: -(len(suffix) + 1)]
            eff = suffix
            break

    family = _ALIASES.get(slug, slug)
    variants = _FAMILY_VARIANTS.get(family)
    if variants:
        if "" in variants:  # single-variant family; effort has nothing to pick
            return variants[""]
        if eff in variants:
            return variants[eff]
        # Effort absent/unknown: prefer medium, then high, then low.
        for pick in ("medium", "high", "low"):
            if pick in variants:
                return variants[pick]
    return settings.default_model


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def ensure_workdir(path: Path) -> Path:
    """Create the workspace dir if missing; return it. Best-effort."""
    os.makedirs(path, exist_ok=True)
    return path
