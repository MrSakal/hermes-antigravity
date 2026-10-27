# Hardening and operational limits

Run **one uvicorn worker**. Conversation continuations, admission and attachment pins
are process-local; extra workers are not a scaling mechanism for this architecture.

## Discovery and compatibility

`/v1/models` advertises one canonical agy slug for each upstream variant. Display
names, family aliases and historical friendly aliases still resolve as inputs, but
no longer create duplicate picker entries. High/medium/low variants remain distinct.
The compatibility discovery endpoints use the same catalog. Refresh client discovery
caches after upgrading; a client with an explicitly pinned model list may need that
list refreshed separately.

`tool_choice: "none"` starts a restricted plain-chat turn even if `tools` was supplied.
`auto` is supported. `required` and named forced choices return an explicit 400 because
the CLI cannot enforce them. A suspended tool turn cannot switch to `none` mid-flight.
Unknown names and invalid argument schemas are rejected before enqueueing external
calls; continuation IDs must all belong to the same suspended conversation and include
its complete outstanding result set. Invalid continuations do not consume valid IDs.
They retain the existing 409 `tool_result_expired` response contract. Malformed
declaration schemas return 400 before allocating workspaces or session permits.

The standard `reasoning_effort` (`low`, `medium`, `high`) aliases `effort`; conflicting
values are rejected. Exact variant slugs retain their explicit effort selection.
`temperature`, `top_p`, `max_tokens`, `max_completion_tokens` and `stop` still have no
reliable CLI equivalents. They are accepted for client compatibility, **not enforced**;
responses enumerate supplied ignored fields in `X-AGI-Ignored-Parameters`. Do not rely
on these fields as cost/output limits. Native output queue limits and CLI/request
deadlines provide separate resource bounds.

## Capacity and input limits

See `.env.example` for all defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `AGI_MAX_SESSIONS` | 8 | Allocated turn permits (incl. suspended turns); also the cap on live conversation processes (IDLE ones LRU-evicted) |
| `AGI_WARM_POOL_SIZE` | 2 | Pre-booted procs on top of `AGI_MAX_SESSIONS` (≈200 MB each; 0 = off) |
| `AGI_WARM_POOL_SIGNATURES` | 2 | Distinct signatures kept warm at once (size is split across them) |
| `AGI_AGENT_START_TIMEOUT_S` | 30 | Per-spawn boot bound; also how long a hung agy blocks other spawns |
| `AGI_MAX_SUSPENDED_SESSIONS` | 4 | Sessions waiting for client tool results |
| `AGI_MAX_QUEUED_REQUESTS` | 16 | Maximum waiting session admissions |
| `AGI_ADMISSION_TIMEOUT_S` | 5 | Maximum wait for a new session; overload returns 503 |
| `AGI_EVENT_QUEUE_SIZE` | 256 | Bounded CLI event queue; backpressure on producers |
| `AGI_MAX_PENDING_TOOL_CALLS` | 32 | Maximum pending bridge calls per conversation |
| `AGI_STDIN_TIMEOUT_S` | 30 | Deadline feeding the CLI's input pipe |
| `AGI_MAX_REQUEST_BYTES` | 32,000,000 | Full HTTP body, including JSON/base64/schemas; 413 on excess |
| `AGI_BODY_TIMEOUT_S` | 30 | Absolute body upload deadline; 408 on expiry |
| `AGI_MAX_PROMPT_BYTES` | 4,000,000 | UTF-8 text prompt limit; 400 on excess |
| `AGI_MAX_IMAGES` | 16 | Number of image parts across the full request |
| `AGI_MAX_IMAGE_BYTES` | 8,000,000 | Decoded bytes per image |
| `AGI_MAX_TOTAL_IMAGE_BYTES` | 16,000,000 | Total decoded image bytes per request |
| `AGI_MAX_TOOLS` | 128 | Number of declared functions |
| `AGI_MAX_TOOL_SCHEMA_BYTES` | 1,000,000 | Serialized tool definitions including descriptions |

HTTP completions being parsed/served are also bounded by session capacity plus
queue capacity. Continuations reuse their session permits, so a fully allocated
session pool does not itself prevent existing conversations from finishing.
Tune limits for the host's RAM and actual CLI footprint; input caps are per request,
not a promise of total process memory. Only supported base64 image data URLs are
accepted; remote URLs are not fetched, and invalid images are not silently skipped.

An absolute HTTP-turn deadline includes admission and startup time. Each resumed
tool-result HTTP request receives a new deadline; suspended time is separately
bounded by `AGI_SUSPENDED_TTL_S`. Unexpected EOF without a terminal result is an
upstream error, including after partial text. SSE failures appear as OpenAI-shaped
`error` events followed by `[DONE]`, not a successful `stop` finish.

## Conversation and process ownership

Processes are persistent. After a clean answer a process is parked IDLE under
`H(spawn signature, normalized history)`: the signature covers model, workspace,
agent-file template digest (system prompt, builtin allowlist, MCP server name),
toolset hash and a preamble-delivered system prompt; the history includes the
assistant text exactly as returned (whitespace-stripped), list content (images)
verbatim, and tool calls by id + name. A claim is atomic and single-use (IDLE →
RUNNING under one lock; dormant records are popped), so concurrent branches fall
back to fresh conversations instead of driving one mutable conversation twice.
A turn abandoned mid-flight (disconnect, timeout, error) kills its process —
agy has no per-turn cancel and its history would diverge from the client's — so
only processes whose history equals what the client saw are ever parked.

`AGI_MAX_SESSIONS` caps live conversation processes: IDLE ones are LRU-evicted
(gracefully: stdin EOF, then kill after 2 s) and kept dormant. A dormant resume
(`--conversation`) waits for the predecessor process to have fully exited
(15 s, else it starts fresh), so two processes never write one agy
conversation; if agy answers with a different conversation id (pruned DB) the
resume is discarded before any message is sent and the history is refolded.
Request-scoped cleanup is pinned to the turn it started (a turn token), so a
late cleanup never kills the next turn a fast client already claimed; GC kills
claimed-but-undriven turns after 30 s. Kills are issued synchronously
(`killpg(SIGKILL)` / `taskkill /T /F` Popen) before any await, and the teardown
runs shielded, so a second cancellation cannot leave a process alive.

A duplicate of a continuation whose tool results were already delivered (the
tool loop finished) is answered 409 `tool_result_expired` from a bounded memory
of completed `tool_call_id`s (`AGI_COMPLETED_CALL_MEMORY_MAX` / `_S`, default
1024 / 600 s) — never rebuilt, which would re-run the turn. A continuation whose
suspended process died is a plain expiry and is rebuilt (if enabled).

Each CLI subprocess owns a process group on POSIX (`start_new_session`).
Cleanup sends SIGTERM to that group, waits up to 5 s, then SIGKILLs the group —
also when the leader already exited. Processes that deliberately escape into a
new session are outside this mechanism; the service manager's cgroup remains
useful defense in depth. On Windows there are no process groups: cleanup runs
`taskkill /PID <pid> /T /F` (synchronous Popen, so a cancellation cannot
interrupt it) while the leader is alive, then kills the leader if needed; a
descendant whose leader already exited cannot be reached that way. The Windows
path costs nothing on Linux (it is behind an `os.name` check). Registry and
process acquisition is rolled back on errors and cancellation.

The shared MCP configuration is not written during operation: each tool turn's
MCP server is declared in the agent file it boots with (`inheritMcp: false`,
one `mcpServers` entry). The managed agent file is a single path shared by all
spawns; one in-process lock is held from its write until the spawned agy printed
its first stdout line (agy reads the file before that line). A spawn that
produces no output within `AGI_AGENT_START_TIMEOUT_S`, fails, or is cancelled is
killed and reaped **before** the lock is released, so a slow agy can never read
the next spawn's file. Writes are atomic (temp + replace, retried briefly on a
Windows sharing violation). The same window is also held under an advisory OS
file lock (`agents/.hermes.md.lock`; `flock` on POSIX, `msvcrt.locking` on
Windows, polled with `asyncio.sleep`), so two servers sharing one plugin
directory serialize too; still prefer one directory + agent name per server
(`AGI_AGENT_PLUGIN_PATH` + `AGI_AGENT_NAME`, as `scripts/bench.py` does).
Request-driven spawns take the lock before warm-pool refills; a request whose
queue would outlast its deadline fails fast with 503 `spawn_queue_full`. An agy
that exits before its first stdout line is a failed spawn (502
`upstream_start_failed` with its stderr tail), never a live-looking dead
process. The startup migration of older builds' shared-config slots only
removes `hermes`/`hermesN` entries with a loopback `/mcp/conv<N>-<id>` URL,
backs the original up once, and leaves an unparsable file untouched.

Conversation teardown also terminates and evicts its MCP SDK transport sessions.
This cleanup uses a small, HTTP-regression-tested adapter to private SDK maps;
`mcp` is constrained to the verified 1.28 series. Revalidate the adapter before
upgrading that dependency. `mcp_transport_sessions` exposes retained transports.

## Attachment retention

`AGI_ATTACHMENT_TTL_S=86400`, `AGI_ATTACHMENT_MAX_BYTES=256000000` and
`AGI_ATTACHMENT_GC_INTERVAL_S=300` control **per-workspace** storage. Only generated
`attachments/img-<16 hex>.<supported image extension>` files are eligible. Symlink
directories/files are rejected and unrelated files are not deleted.

Active and suspended requests pin their entire workspace against GC. Incoming images
are validated before writes and materialized atomically; identical images reuse files.
Insufficient quota returns 507 rather than evicting files an active model may still
need. Idle files are aged/pruned during periodic sweeps and request preparation.
Workspaces encountered in the current process and the default workspace are tracked;
this is not a whole-home filesystem cleaner. Multiple independent bridge processes
must not share an attachment workspace because their pins are not shared.

## Readiness and measurements

- `/healthz`: process liveness only.
- `/readyz`: availability of the CLI executable and a successfully loaded catalog;
  this does **not** claim a model inference or account quota probe.
- `/metrics`: JSON counters, allocated/waiting/suspended sessions, bridge queue depth,
  pinned-workspace count, and p50/p95 duration/TTFT over at most 1,024 recent samples.
  TTFT is recorded on text arrival from the upstream event stream, relative to HTTP
  request entry (including body/queue/startup), not inferred from final response time.

Metrics contain no prompts or tool results. Optional bearer authentication also covers
readiness, metrics and compatibility APIs. MCP endpoints additionally require a
loopback client address, even when an API key is configured. **Do not publish `/mcp`
through a reverse proxy that makes remote callers appear local.** Readonly agents still
have filesystem read access as the service UID; a tool allowlist is not a filesystem
or tenant-isolation sandbox.

Markdown table rewriting is now off by default. Desktop clients render Markdown
natively; enable `AGI_FLATTEN_MARKDOWN_TABLES=true` only for clients needing ASCII.
The optional transform has bounded buffers and passes oversized lines/tables through
without conversion rather than accumulating unbounded content.

## Warm pool and rebuild (ported from claude-code-openai-server)

Profiling after the lifecycle fixes showed agy's boot (~3.5 s exec→`init`) is
the dominant, prompt-independent cost of a turn. `AGI_WARM_POOL_SIZE` pre-boots
that many procs with the prompt deferred (`AgySession(prompt=None)` + `send()`;
`--print-timeout` only starts counting at the prompt, verified). Only fresh
autonomous turns adopt; the pool keeps the `AGI_WARM_POOL_SIGNATURES` hottest
signatures over the last 16 fresh turns (seen at least twice, LRU among those;
a seen-once signature only fills a free slot and never displaces a warm one), so
alternating traffic does not thrash it. Failed pooled boots back off exponentially (1 s to 60 s). Pooled procs are outside the admission count until
adopted, are recycled after `AGI_WARM_POOL_MAX_IDLE_S`, and are drained at
shutdown. Disconnect cleanup of an adopted proc is covered by the live E2E,
which asserts on `/metrics` (`sessions_active`, `warm_pool_idle`) rather than
PID deltas, since the newly appearing PID under a pool is the background refill.

`AGI_REBUILD_ON_EXPIRY` turns a tool continuation for a vanished conversation
into a fresh turn folded from the full history (the transcript carries the
assistant's `tool_calls` and the `tool` results, now rendered as
`[result of <id>]`). A conversation that is alive but not waiting on these ids
(`DuplicateContinuation`) is never rebuilt and stays 409 — rebuilding would run
the turn twice.

## agy-side state

The bridge never resumes a conversation older than `AGI_CONVERSATION_CACHE_TTL_S`
(1 h), and agy answers `--conversation <pruned id>` with a stderr warning and a
**new** conversation. The route compares the `init` conversation id against the
requested one and refolds the whole history on mismatch, so pruning agy's
`conversations/` is safe for the bridge. `scripts/prune_agy_state.py` +
`deploy/agy-prune.timer` implement that retention (default 7 days, WAL
checkpoint of kept DBs, plus old `mcp/hermes-<8hex>/` schema caches). It runs
alongside the always-on persistent processes: conversations a live process
references (`--conversation <id>`, from `/proc` or the Windows command line) or
touched within the keep window are skipped; it refuses (exit 2) only when it
cannot list processes at all. Interactive agy sessions on the same host are not
distinguishable from bridge turns; raise `--keep-days` if you rely on them.
