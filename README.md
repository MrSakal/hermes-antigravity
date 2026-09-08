# antigravity-openai-server

Use **Google Antigravity** from any OpenAI-compatible client.

A small HTTP server that makes the `agy` CLI look like an OpenAI API (the way
LM Studio or Ollama do). Point any OpenAI client — hermes, the OpenAI Python SDK,
Open WebUI, etc. — at it and you get Antigravity's agent, using your existing
Google-account sign-in shared with the Antigravity IDE (no API key, no per-token
billing).

Under the hood it drives `agy` as one subprocess per turn over its
stdin/stdout `stream-json` protocol (requires agy >= 1.1.15); conversation history lives server-side in agy and
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
- **Builtin-tool restrictions:** plain chat defaults to a `view_file` allowlist
  (including image viewing). Tool-passthrough turns still require the full builtin
  set for dynamic MCP dispatch. This is not an OS or filesystem sandbox.
- **Conversation cache:** a follow-up request whose history matches a previous
  turn resumes the stored agy conversation and sends only the newest user
  message — cheaper and smaller on the upstream wire. The request byte cap still applies.
- **Real system prompts via managed agents** (default on): agy has no
  system-prompt flag, but it does support custom agents. The request's system
  message is materialized as an agent definition inside a server-managed plugin
  (`~/.gemini/config/plugins/hermes-bridge/agents/hermes-<hash>.md`) and agy is
  spawned with `--agent <slug>` — the agent body rides at true system-prompt
  level, behaviorally overrides the CLI's built-in persona, and stays in effect
  across conversation resumes. (Antigravity's base identity/environment context
  remains present underneath — the equivalent of hermes-claude's
  `--append-system-prompt` mode; a full "bare model" replacement is not
  possible, see the deviations note below.) When disabled the system message
  uses a prompt preamble. Restrictive tool modes still require a neutral agent;
  an unwritable plugin returns 503 rather than silently granting full tools.
- **Reasoning passthrough** (always on): the model's thinking surfaces as
  OpenAI-style `reasoning_content` — a delta chunk right before the answer
  text (streaming) or `message.reasoning_content` (non-streaming), so hermes'
  thinking pane works. agy's stream-json hides the thinking text; it is read
  best-effort from the CLI's internal brain transcript (undocumented, verified
  against 1.1.4). Blocks arrive whole per model step — there is no incremental
  thinking stream. If the file/format ever changes, responses simply carry no
  reasoning; nothing breaks.
- **Image input**: image attachments use local files; agy's builtin
  file tools can view local images. Data-URL images in OpenAI `image_url`
  content parts are written to `<workdir>/attachments/` (content-addressed,
  deduplicated) and the prompt references the file so the model looks at it.
  Remote URLs, invalid base64, and oversized images are rejected explicitly.
- **Markdown table flattening** (opt-in): rewrites pipe tables to fenced
  ASCII so they render in clients that don't support Markdown tables (Discord).
- **Ollama / llama.cpp compatibility shims** so capability-probing frontends
  (Open WebUI, etc.) accept the server.

## Requirements

- The `agy` CLI **>= 1.1.15** installed and signed in (it shares auth with the Antigravity
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

The authoritative, commented template is [`.env.example`](.env.example).
For shell startup use `uvicorn --env-file .env`; `Settings` does not load `.env`
automatically. The systemd example below uses `EnvironmentFile`.

### Config reference

| Var | Default | Meaning |
|-----|---------|---------|
| `AGI_HOST` | `127.0.0.1` | Bind host. Non-loopback requires `AGI_API_KEY` (see Security). |
| `AGI_PORT` | `8788` | Bind port. Also used to build the per-conversation MCP callback URL. |
| `AGI_API_KEY` | _(unset)_ | Bearer token required on every `/v1` request when set. Mandatory for a non-loopback bind. |
| `AGI_AGY_BIN` | `agy` | Path to the `agy` CLI. |
| `AGI_DEFAULT_MODEL` | `Gemini 3.8 Flash (Medium)` | Model used when the request names none (display name or agy slug). Unknown requests only fall back when strict mode is disabled. |
| `AGI_DEFAULT_EFFORT` | _(unset)_ | Effort used to pick a model variant (`low`/`medium`/`high`). |
| `AGI_MAX_PROMPT_BYTES` | `4000000` | UTF-8 prompt sanity cap; oversized requests return 400 `context_length_exceeded`. Replaces `AGI_MAX_PROMPT_CHARS`; no disk prompt handoff. |
| `AGI_DISABLE_SLASH_COMMANDS` | `true` | Prevent user content from invoking CLI slash commands. |
| `AGI_STRICT_MODEL` | `true` | Reject unknown models with 400 `model_not_found`; false retains legacy default fallback. |
| `AGI_MODEL_CATALOG_REFRESH_S` | `3600` | Query `agy models` at startup and periodically; 0 disables periodic refresh. Failed refresh retains the last good catalog, with static fallback before the first successful load. |
| `AGI_PLAIN_CHAT_TOOLS` | `readonly` | Agent builtin allowlist: `readonly` = view_file, `none` = no tools, `all` = unrestricted. |
| `AGI_TOOL_TURN_TOOLS` | `all` | Dynamic MCP dispatch requires the full builtin set on current agy. |
| `AGI_DISCONNECT_POLL_S` | `1.0` | Nonstream disconnect polling interval for plain and tool turns; 0 disables polling. Streaming cancellation also reaps the process. |
| `AGI_SYSTEM_PROMPT_VIA_AGENT` | `true` | Deliver the system message as a generated agent definition (`--agent <slug>`), overriding the CLI persona. Off → prompt preamble; readonly/none still require an agent and fail closed if it cannot be created. |
| `AGI_AGENT_PLUGIN_PATH` | `~/.gemini/config/plugins/hermes-bridge` | Managed agent-plugin directory override. |
| `AGI_AGENT_CACHE_MAX` | `64` | Max generated agent files kept (LRU by mtime). |
| `AGI_AGY_BRAIN_PATH` | `~/.gemini/antigravity-cli/brain` | Brain (conversation transcript) directory override — the source of the always-on `reasoning_content` passthrough. |
| `AGI_IMAGE_INPUT` | `true` | Materialize data-URL images as files under `<workdir>/attachments/` and reference them in the prompt. |
| `AGI_CONVERSATION_CACHE_MAX` | `256` | Max entries in the history→conversation-id cache. |
| `AGI_CONVERSATION_CACHE_TTL_S` | `3600` | Cache entry lifetime. |
| `AGI_MANAGE_MCP_CONFIG` | `true` | Allow registering per-conversation entries in the shared MCP config (required for tool calling). |
| `AGI_MCP_CONFIG_PATH` | `~/.gemini/config/mcp_config.json` | Shared Antigravity MCP config location override. |
| `AGI_FLATTEN_MARKDOWN_TABLES` | `false` | Optional bounded pipe-table to ASCII conversion. |
| `AGI_DEFAULT_WORKDIR` | `~/agi-workspace` | Working dir granted to agy via `--add-dir` (and used as cwd). |
| `AGI_ALLOWED_WORKDIR_ROOTS` | `[]` | JSON list of extra roots a per-request `workdir` may resolve under. |
| `AGI_REQUEST_TIMEOUT_S` | `600` | Absolute HTTP turn deadline, including queue/startup time. |
| `AGI_SUSPENDED_TTL_S` | `300` | How long a tool-suspended conversation may wait for its results before GC. |
| `AGI_IDLE_SESSION_TTL_S` | `900` | Idle conversation eviction age. |
| `AGI_GC_INTERVAL_S` | `30` | GC sweep interval. |
| `AGI_REBUILD_ON_EXPIRY` | `true` | Rebuild an expired tool continuation from the resent history instead of 409. |
| `AGI_WARM_POOL_SIZE` | `0` | Pre-booted idle agy procs for fresh autonomous turns (see [Warm pool](#warm-pool)). |
| `AGI_WARM_POOL_MAX_IDLE_S` | `600` | Recycle an idle pooled proc after this many seconds. |
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
- `GET /readyz` — CLI/catalog readiness (503 until available; not an inference probe)
- `GET /metrics` — bounded-window TTFT/duration statistics, session/queue counts and errors
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
  -d '{"model":"gemini-3.8-flash","messages":[{"role":"user","content":"hello"}]}'
```

With `AGI_API_KEY` set, add the bearer header:

```bash
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $AGI_API_KEY" \
  -d '{"model":"gemini-3.8-flash","messages":[{"role":"user","content":"hello"}]}'
```

From **hermes**, add a provider in `~/.hermes/config.yaml` and select it:

```yaml
providers:
  antigravity:
    name: Antigravity
    base_url: http://127.0.0.1:8788/v1
    api_key: any-string-accepted   # or your AGI_API_KEY if one is set
    api_mode: chat_completions
    default_model: gemini-3.8-flash
    models: [gemini-3.8-flash, gemini-3.1-pro, claude-sonnet-4.6, claude-opus-4.6, gpt-oss-120b]
```

Models are discovered from **`agy models`**. The server maps
OpenAI-friendly slugs onto them, with `effort` (or a `-low/-medium/-high` slug
suffix) picking the variant:

| Slug | agy model |
|------|-----------|
| `gemini-3.8-flash` (+`-low/-high`, or `effort`) | `Gemini 3.8 Flash (Low/Medium/High)` |
| `gemini-3.1-pro` (+`-high`, or `effort`) | `Gemini 3.1 Pro (Low/High)` |
| `claude-sonnet-4.6` / `sonnet` | `Claude Sonnet 4.6 (Thinking)` |
| `claude-opus-4.6` / `opus` | `Claude Opus 4.6 (Thinking)` |
| `gpt-oss-120b` | `GPT-OSS 120B (Medium)` |

Unknown IDs return **400 `model_not_found`** by default. Set `AGI_STRICT_MODEL=false`
only when legacy default-model fallback is intentional. `/v1/models` is the live
source of truth; the table above illustrates aliases, not a frozen catalog.
Discovery returns one canonical CLI slug per real variant. Display names and
family aliases remain accepted as input but are not advertised as duplicate models.
The standard `reasoning_effort` field is accepted alongside legacy `effort`;
conflicting values are rejected.

### Hardening and capacity

See [hardening and operational limits](docs/hardening.md) for session admission,
request/image/schema caps, attachment retention, error semantics, and migration.

### Warm pool

agy boots for ~3.5 s before it reads its prompt, on every turn, regardless of
the prompt. `AGI_WARM_POOL_SIZE=N` keeps `N` procs pre-booted with the prompt
deferred; a fresh autonomous turn whose model/workdir/agent match adopts one and
pays only model latency. Measured on this host (Gemini 3.8 Flash Low, short
prompt): **5.5 s → 1.5 s** per turn, TTFT p50 1.26 s. Conversation-cache
resumes (`--conversation` is spawn-time argv) and tool turns (per-conversation
MCP slot) always cold-start. Each idle proc is ~175 MB RSS and one conversation
DB on disk; 1–2 is the sane range for one user. `/metrics` exposes
`warm_pool_idle` / `warm_pool_hits` / `warm_pool_misses`.

### agy leaves state behind

The CLI persists every conversation, transcript and per-process log under
`~/.gemini/antigravity-cli` with no retention, and does not checkpoint its
SQLite WALs (measured: 480 MB after ~900 turns, 255 MB of it WAL).
`scripts/prune_agy_state.py` removes conversations older than `--keep-days`
(db + wal + brain + presence lock as a unit), stale logs and orphan locks, and
with `--checkpoint` folds the kept WALs back into their DBs. `deploy/` ships a
daily systemd user timer for it. The bridge tolerates pruned conversations: a
resume of a missing id is detected from agy's `init` (it silently starts a new
conversation) and the full history is refolded.

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
`AGI_AGENT_PLUGIN_PATH`). Each distinct (system prompt, tool mode) pair becomes
`agents/hermes-<sha1[:12]>.md` — created on first use, discovered by agy at
conversation start with no install step, reused on repeat requests and
conversation resumes (the slug is deterministic), and pruned LRU beyond
`AGI_AGENT_CACHE_MAX`. Only files matching the `hermes-<hash>.md` pattern are
ever pruned; other plugins and hand-written agents are never touched. Set
`AGI_SYSTEM_PROMPT_VIA_AGENT=false` to use a system-message preamble. A neutral
agent is still installed for readonly/none tool restrictions. Only explicit
`AGI_PLAIN_CHAT_TOOLS=all` permits agent-free fallback; plugin write failures in
restricted modes return **503 `agent_unavailable`**.

Limits of the agent mechanism (tool behavior checked against agy 1.1.27):

- `tools: ["view_file"]` restricts builtin access (agy may also expose its internal
  `manage_task` helper). `tools: []` disables it; `tools: "*"` grants all builtins.
  Explicit `call_mcp_tool` allowlisting fails on this CLI, so tool turns retain
  full permissions. The CLI's `--sandbox` flag did not enforce isolation in live
  probes and is deliberately **not** used as a security boundary.
- Readonly does **not** mean confidential: `view_file` can read files accessible
  to the service user, not merely attachments. Use `none` for text-only tasks that
  must not read local files, or isolate the service account/filesystem. Image
  attachments and CLI-owned conversation transcripts remain on disk; only the
  obsolete `.prompt-payloads` handoff has been removed.
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
.venv/bin/python tests/scripts/e2e_live.py --port 8799 # local live acceptance suite
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

## Native prompt transport

`agy --print "" --input-format stream-json --output-format stream-json` receives
one UTF-8 NDJSON event on stdin, followed by EOF:

```json
{"event":"user","message":{"role":"user","content":"the complete prompt"}}
```

The empty print argument is mandatory: `-p --input-format` makes agy consume
`--input-format` as the prompt. Prompts never enter argv or `.prompt-payloads`.
The byte cap remains a sanity guard, not a guarantee of model context capacity.

## How it works

```
OpenAI client ──HTTP /v1──▶ agi-server ──NDJSON stdin / stdout──▶ agy CLI
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
- Disconnect teardown uses pure ASGI auth middleware and shielded cancellation
  cleanup: TERM, then KILL after 5 seconds if necessary. Non-streaming tool and
  plain-chat requests poll the connection; streaming responses close on cancellation.
