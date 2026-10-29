# antigravity-openai-server

Use **Google Antigravity** from any OpenAI-compatible client.

A small HTTP server that makes the `agy` CLI look like an OpenAI API (the way
LM Studio or Ollama do). Point any OpenAI client — hermes, the OpenAI Python SDK,
Open WebUI, etc. — at it and you get Antigravity's agent, using your existing
Google-account sign-in shared with the Antigravity IDE (no API key, no per-token
billing).

Under the hood it drives `agy` as one PERSISTENT subprocess per conversation over
its stdin/stdout `stream-json` protocol (requires agy >= 1.1.15; tuned for 1.4.8):
a process survives its turn and answers the next one, a pre-booted warm pool
serves the first turn of new conversations, and a follow-up whose process is
gone simply starts fresh with the whole history folded in (that process' agy
state is deleted the moment it exits).

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
- **Cross-turn reuse:** after a clean answer the process is parked IDLE under
  (spawn signature + normalized history incl. the answer exactly as returned).
  A follow-up whose `messages[:-1]` match adopts it and only the newest user
  message is written to its stdin — no boot, no re-folded transcript. When the
  parked process is gone (idle TTL, crash) or anything mismatches (edited
  history, other model/system prompt/tools) it is a safe miss: a fresh process
  (a warm-pool one when it matches) with the whole history folded in. There is
  no `--conversation` resume — the client resends the full history anyway, and
  a resume (a boot argument) could not use the warm pool.
- **Real system prompts via managed agents** (default on): agy has no
  system-prompt flag, but it does support custom agents. The request's system
  message is written into a per-spawn, transient agent file
  (`~/.gemini/config/plugins/hermes-bridge/agents/hermes-<12 hex>.md`) and agy
  is spawned with `--agent hermes-<12 hex>` — the agent body rides at true
  system-prompt level and behaviorally overrides the CLI's built-in persona.
  The file is deleted as soon as agy printed its first stdout line (it has
  read it by then). No lock: concurrent spawns never share a file, so fresh
  boots run fully in parallel. (Antigravity's base identity/environment context
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
| `AGI_ADVERTISE_MODELS` | `families` | `/v1/models` content. `families`: one id per model (`gemini-3.8-flash`); the client picks the variant via `reasoning_effort` (`none`…`max`, clamped to the nearest agy variant: `none`/`minimal` → low, `xhigh`/`max` → high). `all`: legacy list of variant slugs + families + display names. All ids are accepted as input either way. |
| `AGI_MODEL_LIMITS_ENABLED` | `true` | Fetch real per-model limits (`maxTokens`/`maxOutputTokens`) from the Cloud Code `fetchAvailableModels` RPC with agy's own OAuth token; advertised as `context_length`/`max_completion_tokens` on `/v1/models` (family = smallest variant), `/api/v1/models` (also in LM Studio's native `models` shape) and `/api/show`. Unknown models get no fields — never a guessed number. |
| `AGI_MODEL_LIMITS_ENDPOINT` | `https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels` | Override for the limits RPC. |
| `AGI_MAX_PROMPT_BYTES` | `4000000` | UTF-8 prompt sanity cap; oversized requests return 400 `context_length_exceeded`. Replaces `AGI_MAX_PROMPT_CHARS`; no disk prompt handoff. |
| `AGI_DISABLE_SLASH_COMMANDS` | `true` | Prevent user content from invoking CLI slash commands. |
| `AGI_STRICT_MODEL` | `true` | Reject unknown models with 400 `model_not_found`; false retains legacy default fallback. |
| `AGI_MODEL_CATALOG_REFRESH_S` | `3600` | Query `agy models` at startup and periodically; 0 disables periodic refresh. Failed refresh retains the last good catalog, with static fallback before the first successful load. |
| `AGI_PLAIN_CHAT_TOOLS` | `readonly` | Agent builtin allowlist: `readonly` = `tools: [view_file]`, `none` = `tools: []`, `all` = no `tools:` field (every builtin). |
| `AGI_TOOL_TURN_TOOLS` | `all` | Dynamic MCP dispatch requires the full builtin set on current agy. |
| `AGI_INHERIT_USER_MCP` | `false` | Agent `inheritMcp`. False: tool turns see exactly their own hermes MCP server and plain chat none; true also exposes the user's servers from the shared `mcp_config.json`. |
| `AGI_DISCONNECT_POLL_S` | `1.0` | Nonstream disconnect polling interval for plain and tool turns; 0 disables polling. Streaming cancellation also reaps the process. |
| `AGI_SYSTEM_PROMPT_VIA_AGENT` | `true` | Deliver the system message as the body of the managed agent file (`--agent hermes`), overriding the CLI persona. Off → prompt preamble; readonly/none and tool turns still require the agent and fail closed (503) if it cannot be written. |
| `AGI_AGENT_PLUGIN_PATH` | `~/.gemini/config/plugins/hermes-bridge` | Managed agent-plugin directory override. A second server on the same HOME should use its own directory **and** `AGI_AGENT_NAME` (`scripts/bench.py` does). |
| `AGI_AGENT_NAME` | `hermes` | Managed agent base name (per-spawn files `agents/<name>-<12 hex>.md`, `--agent <name>-<12 hex>`). |
| `AGI_AGENT_START_TIMEOUT_S` | `30` | Max wait for a spawned agy's first stdout line; on timeout the process is killed, its agent file deleted and the request fails 504. Only that one spawn waits — boots never queue. Transient agent files older than 2x this value are swept by the GC loop. |
| `AGI_AGY_BRAIN_PATH` | `~/.gemini/antigravity-cli/brain` | Brain (conversation transcript) directory override — the source of the always-on `reasoning_content` passthrough. |
| `AGI_IMAGE_INPUT` | `true` | Materialize data-URL images as files under `<workdir>/attachments/` and reference them in the prompt. |
| `AGI_SESSION_REUSE` | `true` | Park processes IDLE after a turn for cross-turn reuse (`false` = one process per turn, every follow-up re-folds the history). |
| `AGI_COMPLETED_CALL_MEMORY_MAX` / `_S` | `1024` / `600` | Recently delivered `tool_call_id`s; a duplicate continuation for them is 409, not a rebuild. |
| `AGI_MANAGE_MCP_CONFIG` | `true` | Allow the one-time startup removal of `hermes`/`hermesN` slots older builds left in the shared MCP config (backed up first). The server never writes that file otherwise; tool calling no longer depends on it. |
| `AGI_MCP_CONFIG_PATH` | `~/.gemini/config/mcp_config.json` | Shared Antigravity MCP config location override (legacy cleanup only). |
| `AGI_FLATTEN_MARKDOWN_TABLES` | `false` | Optional bounded pipe-table to ASCII conversion. |
| `AGI_DEFAULT_WORKDIR` | `~/agi-workspace` | Working dir granted to agy via `--add-dir` (and used as cwd). |
| `AGI_ALLOWED_WORKDIR_ROOTS` | `[]` | JSON list of extra roots a per-request `workdir` may resolve under. |
| `AGI_REQUEST_TIMEOUT_S` | `600` | Absolute HTTP turn deadline, including queue/startup time. |
| `AGI_SUSPENDED_TTL_S` | `300` | How long a tool-suspended conversation may wait for its results before GC. |
| `AGI_IDLE_SESSION_TTL_S` | `900` | A parked IDLE process is shut down after this long (its agy state is deleted; a later follow-up starts fresh). |
| `AGI_GC_INTERVAL_S` | `30` | GC sweep interval. |
| `AGI_REBUILD_ON_EXPIRY` | `true` | Rebuild an expired tool continuation from the resent history instead of 409. |
| `AGI_WARM_POOL_SIZE` | `2` | Pre-booted idle agy procs (total) for the first turn of new conversations, plain or tool (≈200 MB each; `0` = off; see [Warm pool](#warm-pool)). |
| `AGI_WARM_POOL_SIGNATURES` | `2` | How many distinct spawn signatures (model + agent/system prompt + toolset) the pool keeps warm at once; `AGI_WARM_POOL_SIZE` is split across them. |
| `AGI_WARM_POOL_MAX_IDLE_S` | `600` | Recycle an idle pooled proc after this many seconds. |
| `AGI_STATE_DIR` | Linux `$XDG_STATE_HOME/hermes-antigravity` (`~/.local/state/...`), Windows `%LOCALAPPDATA%\hermes-antigravity` | The server's own state: ownership ledger, per-process agy logs, instance lock (see [Footprint & cleanup](#footprint--cleanup)). |
| `AGI_AGY_STATE_ROOT` | `~/.gemini/antigravity-cli` | agy's state dir. Only conversations in our ledger are ever deleted from it. |
| `AGI_HOUSEKEEPING` | `true` | Ledger + per-process `--log-file` + deletion of our agy state. `false` = agy's default footprint (nothing cleaned). |
| `AGI_HOUSEKEEPING_INTERVAL_S` | `600` | Periodic sweep (crashed runs, leftovers a held file delayed, schema caches, temp files, stale logs). |
| `AGI_MCP_CACHE_MAX_AGE_S` | `604800` | Our `acli/mcp/hermes-<8hex>/` tool-schema caches unused this long are removed. |
| `AGI_KEEP_AGY_LOGS` | `false` | Keep the agy log of a FAILED process (boot failure, error result) in `<state>/logs` for 7 days. Other logs are always deleted when their process ends. |
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

There is **no concurrency cap**: every call gets its own agy process at once —
no admission control, no queue limit, no 503 for capacity, no LRU eviction.
Each live agy process costs **~200 MB RSS**, so size the host for your peak
concurrency. Idle processes leave via `AGI_IDLE_SESSION_TTL_S`.
See [hardening and operational limits](docs/hardening.md) for
request/image/schema caps, attachment retention, error semantics, and migration.

### Warm pool

agy boots for 1–3 s (1.4.8) before it reads its first message. The warm pool
(**on by default**, `AGI_WARM_POOL_SIZE=2`, `AGI_WARM_POOL_SIGNATURES=2`)
keeps pre-booted procs for the hottest spawn signatures (model / workdir /
agent file / toolset) among the recent fresh turns — two by default, so hermes'
main loop and its side calls with a different system prompt both stay warm. A
signature already warm is not displaced by one miss of another: a newcomer must
recur (seen twice within the last 16 fresh turns; LRU among those); the FIRST turn
of a new conversation adopts one and pays only model latency (follow-ups use
cross-turn reuse instead). It serves **tool turns too**: a pooled entry pre-mints
its `conv_id`, registers an MCP bridge under `/mcp/<conv_id>`, boots with that
URL baked into its agent file (`hermes-<toolset sha8>` server), and the
request's tools are late-bound before the first stdin message — safe because
agy connects to MCP servers lazily, at that first message. Because agy caches
tool schemas per server *name*, a pooled tool proc is only adopted by a request
with exactly the same toolset (the name stays content-addressed, nothing to
clean up); the first request of a new toolset is cold.
Refills boot in parallel with request spawns (each spawn has its own agent
file; nothing is locked). A pooled boot that fails (agy exits before its first
line: logged out, unknown model, rejected agent) or a pooled proc dying idle
puts the pool into exponential backoff (1 s doubling to 60 s, one log line per
step, reset on the next good boot) — no respawn loop. A cold request hitting
the same boot failure gets a **502 `upstream_start_failed`** carrying agy's
stderr tail.

Cost: **≈200 MB RSS per idle pooled proc**. Its (empty) agy conversation is
deleted as soon as a never-adopted proc is recycled or discarded. Set `AGI_WARM_POOL_SIZE=0` to disable it (e.g. on a
small VPS); 1–2 is the sane range for one user. `/metrics` exposes
`warm_pool_idle` / `warm_pool_hits` / `warm_pool_tool_hits` /
`warm_pool_misses` / `warm_pool_spawn_failures` / `warm_pool_targets`, next to `processes_live` / `processes_idle` /
`processes_total`, `reuse_hits` / `reuse_misses` and
`duplicate_continuations`.

## Footprint & cleanup

agy persists everything, forever, under `~/.gemini/antigravity-cli` (per
process: `conversations/<uuid>.db{,-wal,-shm}`, `presence/<uuid>.lock`,
`brain/<uuid>/`, `annotations/<uuid>.pbtxt`, `implicit/<uuid>.pb`, a
`log/cli-*.log`, entries in `cache/conversation_metadata.json` /
`cache/last_conversations.json`). That directory also holds **your own** agy /
IDE conversations, indistinguishable by name or age. The server therefore
deletes nothing by pattern or age — only what its ledger proves it created
(`app/housekeeping.py`):

- **Ledger** — `<AGI_STATE_DIR>/ledger.sqlite3` records every conversation uuid
  one of our agy processes announced (warm-pool boots included), every process
  (pid + OS start time + log file) and every workdir. Written off the event
  loop; each server instance holds an OS lock on `run/<instance>.lock`.
- **Per-process logs** — every spawn and every `agy models` refresh runs with
  `--log-file <state>/logs/<id>.log` (verified on 1.4.8: honored, nothing goes
  to `acli/log`); deleted when the process ends (`AGI_KEEP_AGY_LOGS=true` keeps
  a failed process' log for 7 days).
- **Prompt deletion** — a ledger conversation goes as soon as its process
  exits (the server never resumes one): an IDLE process past its TTL, a
  crashed or evicted one, an unadopted warm-pool boot on discard, everything
  at a graceful shutdown, and at the next start whatever a crashed
  run left (its surviving agy processes are killed first — pid + start time,
  so a recycled pid is never hit). Deleted per uuid: the files above plus its
  entries in the two shared cache JSONs (rewritten atomically, other entries
  untouched; skipped when unparsable, held open, or changed under us). Never
  touched: `conversation_summaries.db`, any uuid not in the ledger.
- **Sweep** (every `AGI_HOUSEKEEPING_INTERVAL_S`) — ledger entries whose files
  survived (a Windows file still held), crashed instances, orphan agent files
  and `*.tmp` in the `hermes-bridge` plugin dir, our `acli/mcp/hermes-<8hex>/`
  schema caches unused for 7 days (and legacy `acli/mcp/hermes/` unless you
  configured a server named `hermes`), attachments in every ledger workdir,
  crash-left `.agi-*` temp files, the legacy `mcp_config.json.hermes.lock` and,
  once the legacy MCP-config migration has completed, its
  `.hermes-antigravity.bak`.

What stays while the server runs: the ledger + the `hermes-bridge` plugin
(`plugin.json`) and, for live / parked / resumable conversations, their agy
state. `/metrics` reports `ledger_conversations` and `housekeeping_deleted`.

**Remove everything**: stop the server, then

```bash
python -m app.purge            # --dry-run to preview; exit 2 = a server is running
```

Run it with the server's `AGI_*` environment (e.g. `set -a; . ./.env; set +a`
first) so it finds the same state dir, plugin dir and workdir.

It deletes the ledger conversations' agy state, the `hermes-bridge` plugin,
our MCP schema caches, attachments (and the default workdir if then empty),
the MCP-config backup / legacy lock, and the state dir itself. Nothing that
is not in the ledger.

Upgrading from a build with `scripts/prune_agy_state.py` / `agy-prune.timer`:
both are gone — that timer deleted agy conversations by age alone, your own
included. Disable the timer (`deploy/README.md`).

## MCP: strict, per spawn, in the agent file

agy 1.4.8 lets an agent declare its own MCP servers. A tool turn's agent file
carries exactly one:

```yaml
inheritMcp: false            # AGI_INHERIT_USER_MCP
mcpServers:
  - name: hermes-<sha8 of the canonical tools JSON>
    serverUrl: "http://127.0.0.1:<port>/mcp/<conv_id>"
```

`inheritMcp: false` hides the user's own servers from the shared
`~/.gemini/config/mcp_config.json` (builtins stay). The server name is per
toolset because agy caches tool schemas per server name; the conversation is
routed by the URL. agy contacts the server only at the first user message.

The shared `mcp_config.json` (the Antigravity IDE's file, with your servers and
secrets) is **never written** during operation. At startup, `hermes`/`hermesN`
entries that older builds left there are removed once — only those whose URL is
`http://127.0.0.1:<any port>/mcp/conv<N>-<old id>`; a user's own `hermes`
server survives — after a one-time backup to
`mcp_config.json.hermes-antigravity.bak`. `AGI_MANAGE_MCP_CONFIG=false` skips
even that. Per-prompt `hermes-<12hex>.md` agent files from older builds are
deleted at startup too.

## The managed agent plugin

For system prompt, tool allowlist and MCP server the server owns one plugin
directory, `~/.gemini/config/plugins/hermes-bridge/` (override with
`AGI_AGENT_PLUGIN_PATH`). agy reads `--agent` at process boot, always before it
prints its first stdout line (the `init` event, printed even before any prompt
arrives); deleting the file afterwards does not affect the running process —
verified live on 1.4.8: the persona still applies and the MCP URL from the
deleted file is still connected at the first message. So every spawn — cold
or warm-pool refill — writes its own
`agents/hermes-<12 hex>.md` (frontmatter `name` = file name), spawns with
`--agent hermes-<12 hex>` and deletes the file once agy printed its first line.
No lock, no queue: N fresh conversations boot side by side, and two servers
sharing the directory cannot collide (random names). If the first line does not
arrive within `AGI_AGENT_START_TIMEOUT_S` the process is killed, then the file
deleted (504); if agy exits first, the spawn fails 502 with its stderr tail.
On Windows a delete refused because agy still holds the handle is retried in
the background; a GC-loop sweep removes our transient files older than
2x `AGI_AGENT_START_TIMEOUT_S` that no booting spawn owns, and the startup
sweep removes crash leftovers plus the previous build's `hermes.md`,
`.hermes.md.lock` and legacy `hermes-<12hex>.md`. Warm-pool / reuse signatures
hash the agent *template*, never the random name.
agy snapshots the agent (persona, MCP servers) into the conversation it
creates (a `--conversation` resume would ignore `--agent` and dial the original
MCP URL; the server never resumes). Other plugins and hand-written agents are never touched. Set `AGI_SYSTEM_PROMPT_VIA_AGENT=false`
to use a system-message preamble; the agent is still used (neutral body) for
readonly/none restrictions and tool turns. Only explicit
`AGI_PLAIN_CHAT_TOOLS=all` permits agent-free plain chat; plugin write failures
otherwise return **503 `agent_unavailable`**.

Limits of the agent mechanism (frontmatter checked against agy 1.4.8):

- `tools: [view_file]` restricts builtin access (agy also keeps its internal
  `manage_task` helper); `tools: []` leaves only `manage_task`. The scalar
  `tools: "*"` older builds wrote is **rejected** by 1.4.8 (agent skipped,
  every turn "unknown agent"), and `["*"]` fails at runtime — the full builtin
  set is granted by omitting `tools:`. Explicit `call_mcp_tool` allowlisting
  fails, so tool turns retain full permissions. The CLI's `--sandbox` flag did not enforce isolation in live
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
- Stop it gracefully (`systemctl --user stop`, SIGTERM): shutdown deletes the
  run's agy conversations. A sandboxed unit (`ProtectHome=read-only`) must list
  the state dir in `ReadWritePaths` — see `deploy/agi-server.service`.
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
.venv/bin/python tests/scripts/smoke_session.py  # live: drives agy directly, 2 turns on one process
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

- **Every turn** goes through one conversation manager: reuse a parked IDLE
  process (newest message only), or adopt a warm-pool proc / spawn fresh with
  the history folded into one prompt; the streamed `step_update` text deltas become SSE or a
  single JSON completion.
- **Tool turn:** a conversation owns one agy subprocess and an MCP bridge
  declared in its agent file (the shared MCP config is never written). When the model calls a bridged function,
  the subprocess blocks inside the MCP call; the server returns a `tool_calls`
  response and parks the conversation `SUSPENDED`. The next request (carrying
  the tool results) resolves the pending futures and the same subprocess
  resumes. Matching is by `tool_call_id`.
- **finish_reason** is inferred (`stop` on success, `tool_calls` on suspension);
  `usage` carries prompt/completion/total tokens plus
  `completion_tokens_details.reasoning_tokens`.
- A background GC reaps suspended-too-long conversations, idle ones past
  `AGI_IDLE_SESSION_TTL_S` (their agy state deleted on exit), and claimed turns nobody drove.
- A duplicate of a continuation whose results were already delivered gets 409
  (`tool_result_expired`), never a rebuild that would re-run the turn: the
  server remembers recently completed `tool_call_id`s (10 min / 1024 ids).
- A process that dies mid-turn is an error (502 / SSE error event), never a
  truncated `stop`.
- Disconnect teardown uses pure ASGI auth middleware and shielded cancellation
  cleanup: TERM, then KILL after 5 seconds if necessary. Non-streaming tool and
  plain-chat requests poll the connection; streaming responses close on cancellation.
