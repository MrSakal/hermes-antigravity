# antigravity-openai-server

Use **Google Antigravity** from any OpenAI-compatible client.

A small HTTP server that makes the `agy` CLI look like an OpenAI API (the way
LM Studio or Ollama do). Point any OpenAI client — hermes, the OpenAI Python SDK,
Open WebUI, etc. — at it and you get Antigravity's agent, using your existing
Google-account sign-in shared with the Antigravity IDE (no API key, no per-token
billing).

Under the hood it drives `agy` as one subprocess per turn over its
`stream-json` output protocol; conversation history lives server-side in agy and
follow-up turns resume it via `--conversation <id>`.

## What it does

- **OpenAI surface:** `GET /v1/models` and `POST /v1/chat/completions`
  (streaming + non-streaming).
- **OpenAI function calling:** your client's `tools` are passed through to
  Antigravity via an in-process MCP bridge. The model calls them through agy's
  `call_mcp_tool`, the server returns a normal `tool_calls` response, your
  client runs the tool and sends the result back, and the same agy subprocess
  resumes — the whole multi-step tool loop is one conversation kept alive
  across the continuations.
- **Antigravity's own built-in tools** (file ops, shell, browser automation,
  subagents, …) run internally the whole time.
- **Conversation cache:** a follow-up request whose history matches a previous
  turn resumes the stored agy conversation and sends only the newest user
  message — cheaper, and immune to the prompt-length cap (see below).
- **Real system prompts via managed agents** (default on): agy has no
  system-prompt flag, but it does support custom agents. The request's system
  message is materialized as an agent definition inside a server-managed plugin
  (`~/.gemini/config/plugins/hermes-bridge/agents/hermes-<hash>.md`) and agy is
  spawned with `--agent <slug>` — the agent body rides at true system-prompt
  level, behaviorally overrides the CLI's built-in persona, and stays in effect
  across conversation resumes. (Antigravity's base identity/environment context
  remains present underneath — the equivalent of hermes-claude's
  `--append-system-prompt` mode; a full "bare model" replacement is not
  possible, see the deviations note below.) Fallback (when disabled or the
  plugin dir is unwritable): the system message is injected as an instruction
  preamble at the top of the first turn.
- **Reasoning passthrough** (default on): the model's thinking surfaces as
  OpenAI-style `reasoning_content` — a delta chunk right before the answer
  text (streaming) or `message.reasoning_content` (non-streaming), so hermes'
  thinking pane works. agy's stream-json hides the thinking text; it is read
  best-effort from the CLI's internal brain transcript (undocumented, verified
  against 1.1.4). Blocks arrive whole per model step — there is no incremental
  thinking stream. If the file/format ever changes, responses simply carry no
  reasoning; nothing breaks.
- **Image input**: agy has no image flag and no stdin channel, but its builtin
  file tools can view local images. Data-URL images in OpenAI `image_url`
  content parts are written to `<workdir>/attachments/` (content-addressed,
  deduplicated) and the prompt references the file so the model looks at it.
  Remote `image_url` URLs are not fetched and are skipped.
- **Markdown table flattening** (default on): rewrites pipe tables to fenced
  ASCII so they render in clients that don't support Markdown tables (Discord).
- **Ollama / llama.cpp compatibility shims** so capability-probing frontends
  (Open WebUI, etc.) accept the server.

## Requirements

- The `agy` CLI installed and signed in (it shares auth with the Antigravity
  IDE/app). Windows: `irm https://antigravity.google/cli/install.ps1 | iex`;
  other platforms: see the [install docs](https://antigravity.google/docs/cli-install).
- Python 3.11+.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8788
```

Or via the installed console script:

```bash
.venv/bin/antigravity-interface
```

Configuration is entirely through `AGI_*` environment variables (or a `.env`
file). See the full list below.

## Configuration

Copy `.env.example` to `.env` and edit, or export the vars directly. Every field
is overridable via `AGI_<FIELD>`.

```bash
# antigravity-interface configuration (env prefix AGI_). Copy to .env or export.

# ── HTTP server ──────────────────────────────────────────────────────────────
AGI_HOST=127.0.0.1
AGI_PORT=8788

# ── Auth ─────────────────────────────────────────────────────────────────────
# Optional bearer token gating every /v1 request. Safe to leave UNSET only on a
# loopback bind. The server REFUSES to start on a non-loopback host unless this
# is set — it drives agy with --dangerously-skip-permissions, so an open bind
# with no key is remote code execution for anyone who can reach the port.
# AGI_API_KEY=choose-a-long-random-secret

# ── Antigravity CLI ──────────────────────────────────────────────────────────
AGI_AGY_BIN=agy
AGI_DEFAULT_MODEL=Gemini 3.5 Flash (Medium)
# AGI_DEFAULT_EFFORT=high

# ── Prompt handling ──────────────────────────────────────────────────────────
# agy takes the prompt as an argv argument (stdin is not supported); Windows
# caps a command line at ~32,767 chars.
AGI_MAX_PROMPT_CHARS=28000

# ── System prompt via managed agent ──────────────────────────────────────────
# Deliver the request's system message as a REAL system prompt: it is written
# as an agent definition (agents/hermes-<hash>.md) into a server-managed plugin
# and agy is spawned with --agent <slug>. The agent body overrides the CLI's
# built-in persona and stays in effect across conversation resumes. When false
# (or the plugin dir cannot be written) the system message automatically falls
# back to an instruction preamble at the top of the first turn.
AGI_SYSTEM_PROMPT_VIA_AGENT=true
# AGI_AGENT_PLUGIN_PATH=~/.gemini/config/plugins/hermes-bridge
AGI_AGENT_CACHE_MAX=64

# ── Reasoning / thinking passthrough ─────────────────────────────────────────
# Surface the model's thinking as `reasoning_content` (delta chunk before the
# answer / message field on non-streaming). Read best-effort from agy's
# internal brain transcript; degrades to "no reasoning" on any mismatch.
AGI_REASONING_CONTENT=true
# AGI_AGY_BRAIN_PATH=~/.gemini/antigravity-cli/brain

# ── Image input ──────────────────────────────────────────────────────────────
# Write data-URL images from OpenAI image_url content parts to
# <workdir>/attachments/ and reference the file in the prompt; agy's builtin
# file tools let the (multimodal) model view them.
AGI_IMAGE_INPUT=true

# ── Conversation continuity ──────────────────────────────────────────────────
# History-prefix -> agy conversation id cache: follow-ups send only the newest
# user message with --conversation <id> instead of re-folding the transcript.
AGI_CONVERSATION_CACHE_MAX=256
AGI_CONVERSATION_CACHE_TTL_S=3600

# ── MCP config registration (tool calling) ──────────────────────────────────
# Tool-turn conversations register per-conversation server entries ("hermes",
# "hermes2", ...) in the shared Antigravity MCP config — the ONLY file agy
# honors. Entries are added before spawn and removed at close; foreign entries
# are never touched.
AGI_MANAGE_MCP_CONFIG=true
# AGI_MCP_CONFIG_PATH=~/.gemini/config/mcp_config.json

# ── Output formatting ────────────────────────────────────────────────────────
# Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
# clients that don't support Markdown tables (e.g. Discord).
AGI_FLATTEN_MARKDOWN_TABLES=true

# ── Workspace (agy's --add-dir + cwd) ────────────────────────────────────────
# Per-request `workdir` overrides must resolve under one of
# AGI_ALLOWED_WORKDIR_ROOTS (a JSON list); empty => only the default is allowed.
AGI_DEFAULT_WORKDIR=~/agi-workspace
# AGI_ALLOWED_WORKDIR_ROOTS=["C:\\Projects"]

# ── Lifecycle (seconds) ──────────────────────────────────────────────────────
AGI_REQUEST_TIMEOUT_S=600
AGI_SUSPENDED_TTL_S=300
AGI_IDLE_SESSION_TTL_S=900
AGI_GC_INTERVAL_S=30

# ── Logging ──────────────────────────────────────────────────────────────────
AGI_LOG_LEVEL=INFO
# Emit per-turn latency metrics (spawn_ms / ttft_ms / total_ms / tok_per_s) to
# the "agi.timing" logger at INFO. Off by default so normal operation pays
# nothing; scripts/bench.py turns it on for the throwaway benchmark instance.
AGI_TIMING_LOG=false
```

### Config reference

| Var | Default | Meaning |
|-----|---------|---------|
| `AGI_HOST` | `127.0.0.1` | Bind host. Non-loopback requires `AGI_API_KEY` (see Security). |
| `AGI_PORT` | `8788` | Bind port. Also used to build the per-conversation MCP callback URL. |
| `AGI_API_KEY` | _(unset)_ | Bearer token required on every `/v1` request when set. Mandatory for a non-loopback bind. |
| `AGI_AGY_BIN` | `agy` | Path to the `agy` CLI. |
| `AGI_DEFAULT_MODEL` | `Gemini 3.5 Flash (Medium)` | Model used when the request names none / an unknown one (exact `agy models` display name). |
| `AGI_DEFAULT_EFFORT` | _(unset)_ | Effort used to pick a model variant (`low`/`medium`/`high`). |
| `AGI_MAX_PROMPT_CHARS` | `28000` | Fresh-fold prompt cap (agy takes the prompt via argv; Windows limit ~32,767). |
| `AGI_SYSTEM_PROMPT_VIA_AGENT` | `true` | Deliver the system message as a generated agent definition (`--agent <slug>`), overriding the CLI persona. Off/unwritable → automatic prompt-preamble fallback. |
| `AGI_AGENT_PLUGIN_PATH` | `~/.gemini/config/plugins/hermes-bridge` | Managed agent-plugin directory override. |
| `AGI_AGENT_CACHE_MAX` | `64` | Max generated agent files kept (LRU by mtime). |
| `AGI_REASONING_CONTENT` | `true` | Surface the model's thinking as `reasoning_content` (read from agy's internal brain transcript, best-effort). |
| `AGI_AGY_BRAIN_PATH` | `~/.gemini/antigravity-cli/brain` | Brain (conversation transcript) directory override. |
| `AGI_IMAGE_INPUT` | `true` | Materialize data-URL images as files under `<workdir>/attachments/` and reference them in the prompt. |
| `AGI_CONVERSATION_CACHE_MAX` | `256` | Max entries in the history→conversation-id cache. |
| `AGI_CONVERSATION_CACHE_TTL_S` | `3600` | Cache entry lifetime. |
| `AGI_MANAGE_MCP_CONFIG` | `true` | Allow registering per-conversation entries in the shared MCP config (required for tool calling). |
| `AGI_MCP_CONFIG_PATH` | `~/.gemini/config/mcp_config.json` | Shared Antigravity MCP config location override. |
| `AGI_FLATTEN_MARKDOWN_TABLES` | `true` | Rewrite pipe tables to fenced ASCII. |
| `AGI_DEFAULT_WORKDIR` | `~/agi-workspace` | Working dir granted to agy via `--add-dir` (and used as cwd). |
| `AGI_ALLOWED_WORKDIR_ROOTS` | `[]` | JSON list of extra roots a per-request `workdir` may resolve under. |
| `AGI_REQUEST_TIMEOUT_S` | `600` | Per-turn upstream timeout. |
| `AGI_SUSPENDED_TTL_S` | `300` | How long a tool-suspended conversation may wait for its results before GC. |
| `AGI_IDLE_SESSION_TTL_S` | `900` | Idle conversation eviction age. |
| `AGI_GC_INTERVAL_S` | `30` | GC sweep interval. |
| `AGI_LOG_LEVEL` | `INFO` | Log level. |
| `AGI_TIMING_LOG` | `false` | Emit per-turn latency metrics to the `agi.timing` logger. |

## Security

The server drives the agy CLI with `--dangerously-skip-permissions`, which means
**anyone who can reach the port can run arbitrary code as your user.** Two
interlocks guard this:

1. **Non-loopback bind requires an API key.** `create_app()` refuses to start
   when `AGI_HOST` is not a loopback address and `AGI_API_KEY` is unset:

   ```
   RuntimeError: refusing to start: host='0.0.0.0' is not loopback and no
   api_key is set — set AGI_API_KEY to require a bearer token, or bind to 127.0.0.1
   ```

2. **Bearer auth on `/v1`.** When `AGI_API_KEY` is set, every `/v1` request must
   carry `Authorization: Bearer <key>` (constant-time compared). The MCP mount is
   intentionally exempt — it is reached only by the local agy subprocess over
   loopback and carries no token.

**Recommended:** keep the bind on `127.0.0.1`. If you must expose it (containers,
remote clients), set a long random `AGI_API_KEY` and put it behind TLS. Binding
`0.0.0.0` / a public interface with no key will refuse to boot — by design.

## Endpoints

**OpenAI**
- `POST /v1/chat/completions` — chat (streaming + non-streaming, tool calling)
- `GET /v1/models` — model list

**Health / info**
- `GET /healthz` — `{"status":"ok",…}` (a JSON 404 here means the server is up but the route moved)
- `GET /` — service info

**Ollama / llama.cpp compatibility** (for capability-probing frontends)
- `GET /api/tags`, `POST /api/show`, `GET /api/version`, `GET /version`
- `GET /v1/props`, `GET /props`, `GET /api/v1/models`

**Internal**
- `/mcp/<conv_id>` — in-process MCP bridge, dialed only by the spawned agy over loopback

## Use it

Any OpenAI client, `base_url = http://127.0.0.1:8788/v1`, any `api_key` when none
is configured:

```bash
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"hello"}]}'
```

With `AGI_API_KEY` set, add the bearer header:

```bash
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $AGI_API_KEY" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"hello"}]}'
```

From **hermes**, add a provider in `~/.hermes/config.yaml` and select it:

```yaml
providers:
  antigravity:
    name: Antigravity
    base_url: http://127.0.0.1:8788/v1
    api_key: any-string-accepted   # or your AGI_API_KEY if one is set
    api_mode: chat_completions
    default_model: gemini-3.5-flash
    models: [gemini-3.5-flash, gemini-3.1-pro, claude-sonnet-4.6, claude-opus-4.6, gpt-oss-120b]
```

Models: `agy --model` accepts **exact display names** only; the server maps
OpenAI-friendly slugs onto them, with `effort` (or a `-low/-medium/-high` slug
suffix) picking the variant:

| Slug | agy model |
|------|-----------|
| `gemini-3.5-flash` (+`-low/-high`, or `effort`) | `Gemini 3.5 Flash (Low/Medium/High)` |
| `gemini-3.1-pro` (+`-high`, or `effort`) | `Gemini 3.1 Pro (Low/High)` |
| `claude-sonnet-4.6` / `sonnet` | `Claude Sonnet 4.6 (Thinking)` |
| `claude-opus-4.6` / `opus` | `Claude Opus 4.6 (Thinking)` |
| `gpt-oss-120b` | `GPT-OSS 120B (Medium)` |

Unknown ids fall back to `AGI_DEFAULT_MODEL`. All of them bill through the same
Google subscription the Antigravity IDE uses.

## The shared MCP config

The only MCP configuration agy honors (verified against 1.1.4) is
`~/.gemini/config/mcp_config.json` — the same file the Antigravity IDE manages,
including your own servers and their secrets. For tool calling this server
merges per-conversation entries into it and removes them at conversation end:

- Foreign entries are **never modified**; writes are atomic (temp file +
  replace) and serialized behind one lock.
- Leftover `hermes*` entries from a crashed run are cleaned at startup
  (matched by our port).
- Set `AGI_MANAGE_MCP_CONFIG=false` to forbid touching the file entirely —
  tool calling then returns 400; plain chat is unaffected.

There is no warm pool (and none is needed): agy's persistent backend daemon
absorbs the cold start, so a per-turn subprocess spawn measures ~30–50 ms
(versus ~2.5 s TTFT for a cold `claude` spawn in the sibling project).

## The managed agent plugin

For system-prompt delivery the server owns one plugin directory,
`~/.gemini/config/plugins/hermes-bridge/` (override with
`AGI_AGENT_PLUGIN_PATH`). Each distinct system prompt becomes
`agents/hermes-<sha1[:12]>.md` — created on first use, discovered by agy at
conversation start with no install step, reused on repeat requests and
conversation resumes (the slug is deterministic), and pruned LRU beyond
`AGI_AGENT_CACHE_MAX`. Only files matching the `hermes-<hash>.md` pattern are
ever pruned; other plugins and hand-written agents are never touched. Set
`AGI_SYSTEM_PROMPT_VIA_AGENT=false` to forbid touching the plugin dir — the
system message then falls back to the prompt-preamble injection.

Limits of the agent mechanism (verified against agy 1.1.4) — this is why
hermes-claude's full `bare_model_mode` cannot be mapped:

- The frontmatter `tools:` field is an **additive grant**, not an allowlist: it
  is required to expose `call_mcp_tool` (hence the generated `tools: "*"`),
  but listing a subset does NOT strip the builtin tools — an agent restricted
  to `view_file` still calls `list_dir` freely. There is no `--tools ""`
  equivalent.
- The base Antigravity system context (identity, environment/workspace
  details, date) stays underneath the agent prompt; there is no
  `--exclude-dynamic-system-prompt-sections` equivalent. The agent persona
  dominates behaviorally, matching hermes-claude's append mode
  (`--append-system-prompt`), not its bare replace mode.

## Deploying as a service (systemd user unit)

Example `~/.config/systemd/user/agi-server.service`:

```ini
[Unit]
Description=Antigravity OpenAI Server (AGI)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/youruser/antigravity-openai-server
EnvironmentFile=/home/youruser/antigravity-openai-server/.env
Environment=PATH=/home/youruser/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=HOME=/home/youruser
ExecStart=/home/youruser/antigravity-openai-server/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8788
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now agi-server.service
loginctl enable-linger "$USER"   # so it keeps running after you log out (headless)
```

Notes:
- The unit reads config from `.env` via `EnvironmentFile`.
- The `agy` binary must be on the unit's `PATH` and already signed in for the
  same user.
- There is no `--reload`: editing files on disk does not affect the running
  process until you restart the unit.

## Test

```bash
.venv/bin/python -m pytest -q                    # unit tests (no CLI needed)
```

Live scripts (need the `agy` CLI signed in; the e2e ones also need a running
server on `:8788`):

```bash
.venv/bin/python tests/scripts/e2e_autonomous.py # live: plain chat, stream + non-stream
.venv/bin/python tests/scripts/e2e_tool.py       # live: full tool loop
.venv/bin/python tests/scripts/smoke_session.py  # live: drives agy directly, 2 turns + resume
.venv/bin/python tests/scripts/probe_mcp.py      # live: MCP bridge end-to-end (no server needed)
tests/scripts/curl_nostream.sh                   # live: one-liner sanity checks
tests/scripts/curl_stream.sh
```

### Benchmark

`scripts/bench.py` launches its **own** throwaway uvicorn instance (never touches
a running server), fires representative autonomous / tool / continuation turns
with `AGI_TIMING_LOG=1`, and writes p50/p95 of spawn / TTFT / total / throughput
to JSON:

```bash
.venv/bin/python scripts/bench.py --port 8799 --iters 3 --out bench.json
```

It sets `AGI_PORT` (not just `--port`) so the per-conversation MCP callback URL
the spawned `agy` dials self-matches the throwaway port.

## How it works

```
OpenAI client ──HTTP /v1──▶ agi-server ──argv + stream-json (stdout)──▶ agy CLI
       ▲                        │                                          │
       └──── tool_calls ────────┤          call_mcp_tool(hermesN,…)        │
       │                        │◀──────── in-process MCP bridge ◀─────────┘
       └──── tool result ──────▶┘           (/mcp/<conv_id>, loopback)
```

- **Autonomous turn** (no `tools`): the conversation is folded into one prompt
  for a fresh agy conversation (or resumed via the conversation cache); the
  streamed `step_update` text deltas become SSE or a single JSON completion.
- **Tool turn:** a conversation owns one agy subprocess, an MCP bridge, and its
  own entry in the shared MCP config. When the model calls a bridged function,
  the subprocess blocks inside the MCP call; the server returns a `tool_calls`
  response and parks the conversation `SUSPENDED`. The next request (carrying
  the tool results) resolves the pending futures and the same subprocess
  resumes. Matching is by `tool_call_id`.
- **finish_reason** is inferred (`stop` on success, `tool_calls` on suspension);
  `usage` carries prompt/completion/total tokens plus
  `completion_tokens_details.reasoning_tokens`.
- A background GC reaps suspended-too-long and idle conversations (and their
  config entries).
