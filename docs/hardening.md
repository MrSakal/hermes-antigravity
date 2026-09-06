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
| `AGI_MAX_SESSIONS` | 8 | Allocated subprocess sessions, including suspended turns |
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

Cache fingerprints include structured message/image identity and the route's model,
workspace and agent configuration namespace. A lookup claims its upstream conversation
and removes stale aliases before resume; concurrent branches fall back to fresh
conversations instead of driving one mutable conversation twice.

Each CLI subprocess owns a process group on POSIX. Cleanup targets that group rather
than assuming terminating the immediate parent will terminate every child. Processes
that deliberately escape into a new session are outside this mechanism; the service
manager's cgroup remains useful defense in depth. Registry/config/process acquisition
is rolled back on errors and cancellation.

MCP configuration updates use cooperating interprocess locking and atomic replacement.
Cleanup checks current ownership before deleting a slot. An IDE that ignores the lock
can still race a shared-file read/modify/write; this is not a multi-writer transaction
protocol for arbitrary third-party writers. Corrupt configuration is an error, not an
invitation to overwrite it with an empty object. Lock acquisition has a five-second
deadline, including during shielded rollback. The deployed locking implementation
requires POSIX (`fcntl`); native Windows is not supported by this hardened release.

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

Persistent cross-turn CLI process reuse remains deliberately out of scope: profile
startup overhead after correctness/lifecycle fixes before adding another stateful pool.
