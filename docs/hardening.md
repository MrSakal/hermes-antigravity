# Hardening and operational limits

Run **one uvicorn worker**. Conversation continuations and attachment pins
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
| `AGI_WARM_POOL_SIZE` | 2 | Pre-booted procs on top of the live ones (≈200 MB each; 0 = off) |
| `AGI_WARM_POOL_SIGNATURES` | 2 | Distinct signatures kept warm at once (size is split across them) |
| `AGI_AGENT_START_TIMEOUT_S` | 30 | Per-spawn boot bound (no other spawn waits on it); transient agent files older than 2x are swept |
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

There is **no concurrency limit**: this is a provider, so every call gets its own
agy process immediately — no admission control, no queue cap, no 503 for
capacity, no LRU eviction, no cap on turns parked for tool results. Each live agy
process costs ~200 MB RSS; size the host's RAM for peak concurrency. Input caps
are per request, not a promise of total process memory. Only supported base64 image data URLs are
accepted; remote URLs are not fetched, and invalid images are not silently skipped.

An absolute HTTP-turn deadline includes startup time (504 on expiry). Each resumed
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
RUNNING under one lock), so concurrent branches fall
back to fresh conversations instead of driving one mutable conversation twice.
A turn abandoned mid-flight (disconnect, timeout, error) kills its process —
agy has no per-turn cancel and its history would diverge from the client's — so
only processes whose history equals what the client saw are ever parked.

Live processes are never evicted to make room. IDLE ones leave only via
`AGI_IDLE_SESSION_TTL_S` (gracefully: stdin EOF, then kill after 2 s) or when
they die. An exited process is never resumed (no `--conversation`): its agy
state is deleted at once, and a later follow-up starts fresh — a warm-pool
process when one matches — with the whole resent history folded in, so two
processes never write one agy conversation.
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
one `mcpServers` entry). Every spawn writes its OWN transient agent file
(`agents/<AGI_AGENT_NAME>-<12 hex>.md`, frontmatter `name` = file name, atomic
temp + replace) and deletes it as soon as the spawned agy printed its first
stdout line — agy reads the file before that line, and (verified live on
1.4.8) neither the persona nor the MCP URL depends on the file afterwards. No
spawn ever waits on another (no in-process lock, no OS file lock, no
request-vs-refill priority): N simultaneous fresh conversations boot in
parallel, and two servers sharing a directory cannot collide (random names;
still prefer one directory + agent name per server, as `scripts/bench.py`
does). A spawn that produces no output within `AGI_AGENT_START_TIMEOUT_S`,
fails, or is cancelled is killed and reaped first, then its file is deleted.
On Windows a delete refused with a sharing violation is retried in the
background; anything left is removed by the GC-loop sweep (our transient
files older than 2x `AGI_AGENT_START_TIMEOUT_S` that no booting spawn owns —
never other names or plugins) and the startup sweep (crash leftovers, the
previous build's `hermes.md` + `.hermes.md.lock`, legacy per-prompt files).
agy snapshots the agent into the conversation it creates (a
`--conversation` resume would ignore `--agent` and dial the ORIGINAL
`/mcp/<conv_id>`; the server never resumes). An agy
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
Workspaces are recorded in the housekeeping ledger, so attachments are GC'd in every
workspace any run used (not only those seen since the process started) and removed by
`python -m app.purge`; crash-left `.agi-*` temp files are swept after an hour. This is
not a whole-home filesystem cleaner. Multiple independent bridge processes
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
alternating traffic does not thrash it. Failed pooled boots back off exponentially (1 s to 60 s). Pooled procs are
recycled after `AGI_WARM_POOL_MAX_IDLE_S` and drained at
shutdown. Disconnect cleanup of an adopted proc is covered by the live E2E,
which asserts on `/metrics` (`processes_running`, `warm_pool_idle`) rather than
PID deltas, since the newly appearing PID under a pool is the background refill.

`AGI_REBUILD_ON_EXPIRY` turns a tool continuation for a vanished conversation
into a fresh turn folded from the full history (the transcript carries the
assistant's `tool_calls` and the `tool` results, now rendered as
`[result of <id>]`). A conversation that is alive but not waiting on these ids
(`DuplicateContinuation`) is never rebuilt and stays 409 — rebuilding would run
the turn twice.

## agy-side state (ledger-based, zero footprint)

agy keeps every conversation forever in `~/.gemini/antigravity-cli`, next to the
user's own agy / IDE conversations, which are indistinguishable by name or age.
The former `scripts/prune_agy_state.py` timer pruned by age alone and therefore
deleted the user's conversations too; it is removed. `app/housekeeping.py`
replaces it:

- **Ownership.** Only uuids in the ledger (`<AGI_STATE_DIR>/ledger.sqlite3`) are
  ever deleted: those one of our agy processes announced in its `init` line. A
  non-canonical id is refused outright (no path games). Rows carry the server
  instance id; each instance holds an OS lock (`run/<instance>.lock`, flock /
  msvcrt), so a second server on the same state dir is never touched and a
  crashed one is recognisable.
- **When.** Only after the owning process exited AND the conversation manager no
  longer references the id (a live, IDLE or closing process). Nothing is kept
  for a resume, so an exited process' conversation goes right away. Shutdown deletes everything the instance owns; the next start kills a
  crashed run's surviving agy processes (pid + OS start time — Linux `/proc`
  starttime, Windows `GetProcessTimes` — so a recycled pid is never signalled)
  and then deletes its conversations.
- **What.** `conversations/<uuid>.db{,-wal,-shm}`, `presence/<uuid>.lock`,
  `brain/<uuid>/` (symlinks unlinked, never followed), `annotations/<uuid>.pbtxt`,
  `implicit/<uuid>.pb`, plus the uuid's entries in `cache/last_conversations.json`
  and `cache/conversation_metadata.json`: temp file + `os.replace`, every other
  entry kept verbatim, abandoned if the file changed since it was read (agy's
  write wins) or is held open; an unparsable file is left alone. Residual race:
  agy renaming its own temp over the file between our re-read and our replace
  (microseconds). `conversation_summaries.db` is never touched.
- **Windows.** A `PermissionError` (file still held for a moment) is retried
  briefly; otherwise the ledger row stays and the next sweep retries.
- **Logs.** Every spawn runs with `--log-file <state>/logs/<id>.log` and the log is
  deleted at process exit, so `acli/log/` gains nothing.

Deleting a conversation is always safe for the bridge: the server never passes
`--conversation`, so nothing can ask agy for a deleted id.

