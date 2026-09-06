"""Runtime configuration (env-driven, prefix ``AGI_``) and model-alias resolution.

All knobs are read from the environment with an ``AGI_`` prefix (e.g.
``AGI_PORT=9000``). Defaults are chosen so the server runs out of the box against
a local Antigravity login (shared with the Antigravity IDE) with no API key.
"""

from __future__ import annotations

import os
import ipaddress
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Static fallback catalog — what `agy models` reported on 1.1.27 (Gemini 3.5
# is gone: the CLI rejects it with "invalid model selection"). The live catalog
# (app.model_catalog.ModelCatalog, refreshed from `agy models`) is authoritative
# whenever it has loaded; this table only matters when the CLI cannot be queried
# (offline, not signed in) and for unit tests.
KNOWN_MODELS = [
    "Gemini 3.8 Flash (Medium)",
    "Gemini 3.8 Flash (High)",
    "Gemini 3.8 Flash (Low)",
    "Gemini 3.7 Flash (Medium)",
    "Gemini 3.7 Flash (High)",
    "Gemini 3.7 Flash (Low)",
    "Gemini 3.6 Flash (Medium)",
    "Gemini 3.6 Flash (High)",
    "Gemini 3.6 Flash (Low)",
    "Gemini 3.1 Pro (Low)",
    "Gemini 3.1 Pro (High)",
    "Claude Sonnet 4.6 (Thinking)",
    "Claude Opus 4.6 (Thinking)",
    "GPT-OSS 120B (Medium)",
]

# Family slug → {effort: display name}, keyed exactly like agy's own slugs
# (`agy models` column 1 minus the -low/-medium/-high suffix) so the static and
# live tables have the same shape. Single-variant models use the "" key.
_FAMILY_VARIANTS: dict[str, dict[str, str]] = {
    "gemini-3.8-flash": {
        "low": "Gemini 3.8 Flash (Low)",
        "medium": "Gemini 3.8 Flash (Medium)",
        "high": "Gemini 3.8 Flash (High)",
    },
    "gemini-3.7-flash": {
        "low": "Gemini 3.7 Flash (Low)",
        "medium": "Gemini 3.7 Flash (Medium)",
        "high": "Gemini 3.7 Flash (High)",
    },
    "gemini-3.6-flash": {
        "low": "Gemini 3.6 Flash (Low)",
        "medium": "Gemini 3.6 Flash (Medium)",
        "high": "Gemini 3.6 Flash (High)",
    },
    "gemini-3.1-pro": {
        "low": "Gemini 3.1 Pro (Low)",
        "high": "Gemini 3.1 Pro (High)",
    },
    "claude-sonnet-4-6": {"": "Claude Sonnet 4.6 (Thinking)"},
    "claude-opus-4-6-thinking": {"": "Claude Opus 4.6 (Thinking)"},
    "gpt-oss-120b": {"medium": "GPT-OSS 120B (Medium)"},
}

# Client-friendly aliases → family slug (agy spelling).
_ALIASES: dict[str, str] = {
    "gemini-flash": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "gemini-pro": "gemini-3.1-pro",
    "pro": "gemini-3.1-pro",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-6-thinking",
    "claude-opus-4.6": "claude-opus-4-6-thinking",
    "claude-opus-4-6": "claude-opus-4-6-thinking",
    "opus": "claude-opus-4-6-thinking",
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
    # `Authorization: Bearer <key>`. Left unset, the server is open — which
    # is only safe on a loopback bind (see the startup guard in main.create_app).
    api_key: Optional[str] = None

    # ── Antigravity CLI ──────────────────────────────────────────────────────
    agy_bin: str = "agy"
    default_model: str = "Gemini 3.8 Flash (Medium)"
    default_effort: Optional[str] = None
    # Unknown `model` ids: True → 400 model_not_found (OpenAI semantics, the
    # client sees its mistake); False → silently use default_model (legacy).
    strict_model: bool = True
    # Live catalog: `agy models` at startup + every N seconds. 0 disables the
    # periodic refresh (startup load still happens).
    model_catalog_refresh_s: int = 3600

    # ── Prompt handling ──────────────────────────────────────────────────────
    # The prompt travels over stdin (`--input-format stream-json`), so there is
    # no argv ceiling. This is a sanity cap in UTF-8 bytes (~1 M tokens of
    # context on the largest models); requests above it get 400
    # context_length_exceeded instead of a doomed upstream call.
    max_prompt_bytes: int = 4_000_000
    max_request_bytes: int = Field(default=32_000_000, gt=0)
    body_timeout_s: float = Field(default=30.0, gt=0)
    max_image_bytes: int = Field(default=8_000_000, gt=0)
    max_total_image_bytes: int = Field(default=16_000_000, gt=0)
    max_images: int = Field(default=16, ge=0)
    max_tools: int = Field(default=128, ge=0)
    max_tool_schema_bytes: int = Field(default=1_000_000, gt=0)
    max_sessions: int = Field(default=8, gt=0)
    max_suspended_sessions: int = Field(default=4, gt=0)
    max_queued_requests: int = Field(default=16, ge=0)
    admission_timeout_s: float = Field(default=5.0, gt=0)
    event_queue_size: int = Field(default=256, gt=0)
    max_pending_tool_calls: int = Field(default=32, gt=0)
    stdin_timeout_s: float = Field(default=30.0, gt=0)
    attachment_ttl_s: int = Field(default=86400, gt=0)
    attachment_max_bytes: int = Field(default=256_000_000, gt=0)
    attachment_gc_interval_s: int = Field(default=300, gt=0)
    # Never let user content be interpreted as an agy slash command / skill.
    disable_slash_commands: bool = True

    # ── System prompt via managed agent ─────────────────────────────────────
    # Deliver the request's system message as a REAL system prompt: materialize
    # it as an agent definition (agents/hermes-<hash>.md) inside a server-managed
    # plugin and spawn agy with --agent <slug>. Verified against agy 1.1.4: the
    # agent body overrides the CLI's built-in persona, agent files are discovered
    # at conversation start with no install step, and the agent stays in effect
    # across --conversation resumes. When disabled, the system message uses a
    # preamble. Restricted plain-chat modes still require a neutral agent and
    # fail closed (503) if the plugin cannot be written.
    system_prompt_via_agent: bool = True
    # Managed plugin directory (never touches other plugins). Override for tests.
    agent_plugin_path: Optional[Path] = None
    # Max generated agent files kept (LRU by mtime); older ones are pruned.
    agent_cache_max: int = 64

    # ── Builtin-tool exposure (security) ────────────────────────────────────
    # agy runs under --dangerously-skip-permissions, so every builtin it can
    # reach (run_command, write_to_file, browser, …) is arbitrary code
    # execution as this user — driven by whatever the request contains,
    # including untrusted web pages / tool outputs a client asks to summarize.
    # Verified against 1.1.27: the agent frontmatter `tools:` list IS honored
    # as an allowlist (["view_file"] leaves only view_file + manage_task;
    # `tools: []` leaves none). `call_mcp_tool` can NOT be listed explicitly
    # (the agent fails to start) — it is granted by `tools: "*"` only, so
    # tool-passthrough conversations always get the full set.
    #
    # Modes for requests WITHOUT `tools` (plain chat: hermes' compression,
    # title generation, approval, vision, curator, …):
    #   "readonly" (default) — view_file only (still lets the model look at
    #                         attached images); no shell, no writes, no browser.
    #   "none"               — no builtin tools at all.
    #   "all"                — legacy behavior, tools: "*".
    plain_chat_tools: str = "readonly"
    # Builtins exposed on top of call_mcp_tool for tool-passthrough turns. Only
    # "all" is possible today (see above); kept as a setting so a future agy
    # that lets call_mcp_tool be allowlisted can be adopted without a release.
    tool_turn_tools: Literal["all"] = "all"

    # ── Reasoning / thinking passthrough ─────────────────────────────────────
    # The model's thinking always surfaces as OpenAI-style `reasoning_content`
    # (base functionality, no toggle): a delta chunk before the answer text /
    # `message.reasoning_content` on non-streaming responses. agy's stream-json
    # hides the thinking text, but the CLI writes it to its internal brain
    # transcript (undocumented, verified 1.1.4) — read best-effort from there;
    # degrades to "no reasoning" if the file/format changes.
    # Brain directory override (tests / nonstandard installs). Default:
    # ~/.gemini/antigravity-cli/brain
    agy_brain_path: Optional[Path] = None

    # ── Image input ──────────────────────────────────────────────────────────
    # agy has no image flag, but its builtin file tools can view local images
    # (verified live). When enabled, data-URL images in OpenAI `image_url`
    # content parts are written to <workdir>/attachments/ and the prompt
    # references the file so the model looks at it. Requires view_file, i.e.
    # plain_chat_tools != "none".
    image_input: bool = True

    # ── Conversation continuity ─────────────────────────────────────────────—
    # agy stores conversation history server-side; `--conversation <id>` resumes
    # it. We cache (history-prefix hash → agy conversation id) so a follow-up
    # request sends only the newest user message instead of re-folding the whole
    # transcript (cheaper: the prefix is already in agy's prompt cache).
    conversation_cache_max: int = 256
    conversation_cache_ttl_s: int = 3600

    # ── Output formatting ──────────────────────────────────────────────────—
    # Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
    # clients (e.g. Discord) that don't support Markdown tables.
    flatten_markdown_tables: bool = False

    # ── Workspace ──────────────────────────────────────────────────────────—
    default_workdir: Path = Path("~/agi-workspace").expanduser()
    # Per-request `workdir` overrides must resolve under one of these roots.
    # Empty => only `default_workdir` is allowed.
    allowed_workdir_roots: list[str] = []

    # ── MCP config registration ─────────────────────────────────────────────
    # Tool-turn conversations register per-conversation server entries
    # ("hermes", "hermes2", …) in the shared Antigravity MCP config — the ONLY
    # file agy honors. Override the path for tests; disable management entirely
    # to run the autonomous path without ever touching the user's config.
    mcp_config_path: Optional[Path] = None
    manage_mcp_config: bool = True

    # ── Lifecycle / timeouts (seconds) ─────────────────────────────────────—
    request_timeout_s: int = 600
    suspended_ttl_s: int = 300
    idle_session_ttl_s: int = 900
    gc_interval_s: int = 30
    # How often a non-streaming request checks whether the client went away
    # (so the agy subprocess is killed instead of running to request_timeout_s).
    disconnect_poll_s: float = 1.0

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

    @field_validator("plain_chat_tools", "tool_turn_tools", mode="before")
    @classmethod
    def _tools_mode(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip().lower()
            if v not in {"readonly", "none", "all"}:
                raise ValueError("must be one of: readonly, none, all")
        return v

    def is_loopback_host(self) -> bool:
        """True when the bind host only accepts connections from this machine."""
        h = self.host.strip().lower()
        if h == 'localhost':
            return True
        try:
            return ipaddress.ip_address(h).is_loopback
        except ValueError:
            return False

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


class UnknownModel(ValueError):
    """Raised by :func:`resolve_model` in strict mode for an unresolvable id."""


class PromptTooLarge(ValueError):
    """Raised by :func:`check_prompt_size`; carries the byte count."""

    def __init__(self, nbytes: int, cap: int) -> None:
        super().__init__(f"prompt is {nbytes} UTF-8 bytes; cap is {cap}")
        self.nbytes = nbytes
        self.cap = cap


def check_prompt_size(prompt: str, settings: Settings) -> int:
    """Return the prompt's UTF-8 byte size; raise :class:`PromptTooLarge` when
    it exceeds ``settings.max_prompt_bytes``."""
    n = len(prompt.encode("utf-8"))
    if n > settings.max_prompt_bytes:
        raise PromptTooLarge(n, settings.max_prompt_bytes)
    return n


def resolve_model(
    requested: Optional[str],
    settings: Settings,
    effort: Optional[str] = None,
    *,
    known_models: Optional[list[str]] = None,
    families: Optional[dict[str, dict[str, str]]] = None,
) -> str:
    """Map an OpenAI ``model`` field (slug or display name) to an ``--model``
    value the `agy` CLI accepts.

    * An exact display name (case-insensitive) passes through as the canonical
      display name.
    * A known slug/alias resolves through its family; ``effort`` (or a
      ``-low``/``-medium``/``-high`` suffix on the slug) picks the variant when
      the family has effort variants.
    * Empty/None → ``default_model``.
    * Anything else → :class:`UnknownModel` when ``settings.strict_model``,
      otherwise ``default_model``.

    ``known_models`` / ``families`` let the caller pass the live catalog
    (:mod:`app.model_catalog`); when given they REPLACE the static tables (a
    model agy no longer lists must not resolve), aliases still apply.
    """
    models = known_models if known_models else KNOWN_MODELS
    fams = families if families else _FAMILY_VARIANTS

    r = (requested or settings.default_model).strip()

    # Exact display-name match (case-insensitive).
    for m in models:
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
    variants = fams.get(family)
    if variants:
        if "" in variants:  # single-variant family; effort has nothing to pick
            return variants[""]
        if eff in variants:
            return variants[eff]
        # Effort absent/unknown: prefer medium, then high, then low.
        for pick in ("medium", "high", "low"):
            if pick in variants:
                return variants[pick]
    # A full agy slug whose family the alias table redirects (e.g. the client
    # sent "claude-opus-4.6-high"): nothing sensible to do → unknown.
    if settings.strict_model:
        raise UnknownModel(r)
    return settings.default_model


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def ensure_workdir(path: Path) -> Path:
    """Create the workspace dir if missing; return it. Best-effort."""
    os.makedirs(path, exist_ok=True)
    return path
