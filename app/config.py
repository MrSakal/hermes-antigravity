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

# Static fallback catalog — what `agy models` reports on 1.4.8. The live
# catalog (app.model_catalog.ModelCatalog, refreshed from `agy models`) is
# authoritative whenever it has loaded; this table only matters when the CLI
# cannot be queried (offline, not signed in) and for unit tests.
KNOWN_MODELS = [
    "Gemini 4 Argon (Medium)",
    "Gemini 4 Argon (High)",
    "Gemini 4 Argon (Low)",
    "Gemini 3.8 Flash (High)",
    "Gemini 3.8 Flash (Medium)",
    "Gemini 3.8 Flash (Low)",
    "Claude Opus 5.5 (Low)",
    "Claude Opus 5.5 (Medium)",
    "Claude Opus 5.5 (High)",
    "Claude Sonnet 5.5 (Low)",
    "Claude Sonnet 5.5 (Medium)",
    "Claude Sonnet 5.5 (High)",
    "GPT-OSS 120B (Medium)",
]

# Family slug → {effort: display name}, keyed exactly like agy's own slugs
# (`agy models` column 1 minus the -low/-medium/-high suffix) so the static and
# live tables have the same shape. Single-variant models use the "" key.
_FAMILY_VARIANTS: dict[str, dict[str, str]] = {
    "gemini-4-argon": {
        "low": "Gemini 4 Argon (Low)",
        "medium": "Gemini 4 Argon (Medium)",
        "high": "Gemini 4 Argon (High)",
    },
    "gemini-3.8-flash": {
        "low": "Gemini 3.8 Flash (Low)",
        "medium": "Gemini 3.8 Flash (Medium)",
        "high": "Gemini 3.8 Flash (High)",
    },
    "claude-opus-5-5": {
        "low": "Claude Opus 5.5 (Low)",
        "medium": "Claude Opus 5.5 (Medium)",
        "high": "Claude Opus 5.5 (High)",
    },
    "claude-sonnet-5-5": {
        "low": "Claude Sonnet 5.5 (Low)",
        "medium": "Claude Sonnet 5.5 (Medium)",
        "high": "Claude Sonnet 5.5 (High)",
    },
    "gpt-oss-120b": {"medium": "GPT-OSS 120B (Medium)"},
}

# Client-friendly aliases → family slug (agy spelling).
_ALIASES: dict[str, str] = {
    "gemini-flash": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "argon": "gemini-4-argon",
    "gemini-argon": "gemini-4-argon",
    "gemini": "gemini-4-argon",
    "claude-sonnet": "claude-sonnet-5-5",
    "claude-sonnet-5.5": "claude-sonnet-5-5",
    "sonnet": "claude-sonnet-5-5",
    "claude-opus": "claude-opus-5-5",
    "claude-opus-5.5": "claude-opus-5-5",
    "opus": "claude-opus-5-5",
    "claude": "claude-opus-5-5",
    "gpt-oss": "gpt-oss-120b",
}

# Short aliases → keywords matched against the families the catalog actually
# has, first family in agy's order wins. Used when the alias table above points
# at a family the live catalog no longer lists (agy 1.4.8 replaced Gemini 3.1
# Pro and Claude 4.6 with Gemini 4 Argon / Claude 5.5) — so "sonnet" follows
# the current Sonnet without a code change. Only consulted after an exact
# display name / slug / family match failed, so it never shadows a real id.
_ALIAS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "argon": ("argon",),
    "gemini-argon": ("argon",),
    "gemini": ("argon", "gemini"),
    "flash": ("flash",),
    "gemini-flash": ("flash",),
    "pro": ("pro",),
    "gemini-pro": ("pro",),
    "sonnet": ("sonnet",),
    "claude-sonnet": ("sonnet",),
    "opus": ("opus",),
    "claude-opus": ("opus",),
    "claude": ("opus", "claude"),
    "gpt-oss": ("gpt-oss",),
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
    # Deliver the request's system message as a REAL system prompt: write it
    # into the ONE managed agent file (plugins/hermes-bridge/agents/hermes.md)
    # and spawn agy with --agent hermes. Verified against agy 1.4.8: the body
    # overrides the CLI's built-in persona and is read at process boot, before
    # the first stdout line — every spawn holds a lock from the write until that
    # line (see app/agent_config.py), so concurrent requests never see each
    # other's prompt. The file is rewritten only when its bytes change. When
    # disabled, the system message uses a preamble. Restricted plain-chat modes
    # and tool turns still need the agent (tool allowlist / MCP server) and
    # fail closed (503) if the plugin cannot be written.
    system_prompt_via_agent: bool = True
    # Managed plugin directory (never touches other plugins). Override for tests.
    agent_plugin_path: Optional[Path] = None
    # Name of the managed agent (``agents/<name>.md``, ``--agent <name>``). A
    # second server on the same HOME must use its own plugin path AND name —
    # two plugins defining one agent name make ``--agent`` ambiguous.
    agent_name: str = Field(default="hermes", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    # Max wait for a spawned agy's first stdout line (proof it read the agent
    # file) while holding the agent-file lock. On timeout the process is killed
    # and the request fails 504 — a slow agy must never read the next spawn's
    # file. Boot is ~1-3 s; this only bounds a hung CLI, which blocks every
    # other spawn for this long — so it is kept short. A request whose spawn
    # queue would outlast its own deadline fails fast with 503.
    agent_start_timeout_s: float = Field(default=30.0, gt=0)
    # Let agy also see the user's own MCP servers from the shared
    # ~/.gemini/config/mcp_config.json (agent frontmatter `inheritMcp`). Off by
    # default: tool turns then see exactly one MCP server (this conversation's
    # hermes functions) and plain chat sees none.
    inherit_user_mcp: bool = False

    # ── Builtin-tool exposure (security) ────────────────────────────────────
    # agy runs under --dangerously-skip-permissions, so every builtin it can
    # reach (run_command, write_to_file, browser, …) is arbitrary code
    # execution as this user — driven by whatever the request contains,
    # including untrusted web pages / tool outputs a client asks to summarize.
    # Verified against 1.4.8: the agent frontmatter `tools:` YAML list IS
    # honored as an allowlist ([view_file] leaves only view_file + manage_task;
    # `tools: []` leaves manage_task only). `call_mcp_tool` can NOT be listed
    # explicitly, and the scalar `"*"` is rejected — the full set (incl.
    # call_mcp_tool) is granted by omitting `tools:`, so tool-passthrough
    # conversations always get it.
    #
    # Modes for requests WITHOUT `tools` (plain chat: hermes' compression,
    # title generation, approval, vision, curator, …):
    #   "readonly" (default) — view_file only (still lets the model look at
    #                         attached images); no shell, no writes, no browser.
    #   "none"               — no builtin tools at all.
    #   "all"                — every builtin (no `tools:` field).
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

    # ── Conversation continuity (cross-turn reuse) ─────────────────────────—
    # A process survives its turn: after a clean final answer it is parked IDLE
    # under (spawn signature + normalized history incl. the answer exactly as
    # returned). A request whose messages[:-1] match adopts it and only the
    # newest user message is sent (see app/conversation.py). False = one process
    # per turn (every follow-up re-folds the whole transcript).
    session_reuse: bool = True
    # Dormant records: when a parked process is gone (LRU-evicted, idle TTL,
    # crashed) its agy conversation id is remembered under the same history key
    # so a follow-up respawns with `--conversation <id>` and still sends only
    # the newest message. Bound (entries) and lifetime (seconds).
    conversation_cache_max: int = Field(default=256, ge=0)
    conversation_cache_ttl_s: int = Field(default=3600, gt=0)
    # Recently delivered tool_call_ids: a second copy of a continuation whose
    # results were already delivered is answered 409 (DuplicateContinuation)
    # instead of being rebuilt — a rebuild would re-run the turn and its side
    # effects. Bounded by count and age.
    completed_call_memory_max: int = Field(default=1024, ge=0)
    completed_call_memory_s: float = Field(default=600.0, gt=0)

    # ── Output formatting ──────────────────────────────────────────────────—
    # Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
    # clients (e.g. Discord) that don't support Markdown tables.
    flatten_markdown_tables: bool = False

    # ── Workspace ──────────────────────────────────────────────────────────—
    default_workdir: Path = Path("~/agi-workspace").expanduser()
    # Per-request `workdir` overrides must resolve under one of these roots.
    # Empty => only `default_workdir` is allowed.
    allowed_workdir_roots: list[str] = []

    # ── Shared MCP config (legacy migration only) ──────────────────────────
    # Tool turns declare their MCP server in the agent file (strict, per spawn)
    # and never write the shared ~/.gemini/config/mcp_config.json. Older builds
    # did ("hermes", "hermes2", … slots); at startup the server removes those
    # leftovers once — only entries provably ours, after a one-time backup
    # (see app/mcp_config.py). Override the path for tests; set
    # manage_mcp_config=false to never touch the file at all.
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
    # Tool results arriving for a conversation that is already gone (GC'd past
    # suspended_ttl_s, or wiped by a restart): rebuild a fresh turn from the
    # full history the stateless client resent (True) or answer 409 (False).
    # A live conversation that is simply not waiting on these ids is always 409.
    rebuild_on_expiry: bool = True

    # ── Warm pool ───────────────────────────────────────────────────────────
    # Pre-booted idle agy procs that the FIRST turn of a new conversation —
    # plain or tool turn — adopts instead of paying the 1–3 s CLI boot (see
    # app/warmpool.py). Each idle proc costs ≈200 MB RSS plus one empty
    # conversation DB on disk; 1–2 is the useful range for a single user.
    # 0 = off. Pooled procs are not counted against max_sessions.
    warm_pool_size: int = Field(default=2, ge=0, le=8)
    # How many distinct spawn signatures (model + agent/system prompt +
    # toolset) the pool keeps warm at once — the hottest ones over the recent
    # misses/hits. warm_pool_size is the TOTAL cap, split across them (2/2 =
    # one warm proc each for e.g. hermes' main loop + its side-call prompt).
    warm_pool_signatures: int = Field(default=2, ge=1, le=8)
    warm_pool_max_idle_s: float = Field(default=600.0, gt=0)

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
      the family has effort variants. A display family name ("Gemini 4 Argon")
      is slugified into its family slug. Short aliases whose fixed family the
      catalog no longer lists fall back to keyword matching ("argon",
      "sonnet", "claude", … → the first such family agy lists).
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
    if not variants:
        # Alias whose fixed family is gone from this catalog: follow the
        # current family of that kind (see _ALIAS_KEYWORDS).
        for keyword in _ALIAS_KEYWORDS.get(slug, ()):
            match = next((f for f in fams if keyword in f), None)
            if match is not None:
                variants = fams[match]
                break
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
