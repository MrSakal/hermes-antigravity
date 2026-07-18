# hermes-antigravity — design

**Date:** 2026-07-18
**Status:** approved (brainstorm session, hermes-claude repo)

## Goal

An exact functional twin of **hermes-claude** (`claude-code-interface`, `C:\Sajat\hermes-claude`)
wrapping the **Antigravity CLI (`agy`)** instead of the `claude` CLI: a small local
HTTP server exposing an OpenAI-compatible surface (`/v1/chat/completions`,
`/v1/models`, Ollama/llama.cpp compat shims) so hermes and any OpenAI client can
drive Antigravity's agent — using the existing Google-account (subscription)
login shared with the Antigravity IDE, no API key.

Reference documents (Claude Code memory, project `C--Sajat-hermes-claude`):

- `hermes-claude-feature-inventory.md` — the complete mechanism map this project mirrors.
- `antigravity-cli-verified-capabilities-2026-07.md` — live-tested agy behaviors + gap table + open spikes.

## Verified foundations (live-tested 2026-07-18, agy v1.1.4)

- `agy -p "<prompt>" --output-format stream-json` emits line-delimited JSON:
  `init` (conversation_id, cwd, tools[], permission_mode) → `step_update`
  (`step_type`: `agent_response` with `text_delta` / `tool` with
  `tool_info{name,parameters,output}`, `state` ACTIVE→DONE, per-step `usage`) →
  `result` (status, response, num_turns, usage{input,output,thinking,total}).
- `--conversation <id>` resumes server-side history; `--dangerously-skip-permissions`
  → permission_mode `always-proceed`; `--add-dir` identical to claude's.
- MCP: workspace `.agents/mcp_config.json` (standard `mcpServers` schema) is
  auto-discovered from cwd; the model calls tools through one generic builtin
  `call_mcp_tool(ServerName, ToolName, Arguments)`.
- Auth shared with the logged-in Antigravity IDE (Google OAuth, subscription).
- NOT available: `--system-prompt` / `--append-system-prompt` / `--tools` /
  `--mcp-config` / `--permission-mode` / `--effort` flags; effort is baked into
  model variant names (`Gemini 3.5 Flash (Low/Medium/High)`, `Claude Sonnet 4.6
  (Thinking)`, `Claude Opus 4.6 (Thinking)`, `Gemini 3.1 Pro (Low/High)`,
  `GPT-OSS 120B (Medium)`).

## Architecture

Same module layout as hermes-claude. Package `app/`, project name
`antigravity-interface`, env prefix `AGI_`.

### Unchanged modules (copy + rename)

| Module | Notes |
|---|---|
| `main.py` | App factory, lifespan, bearer-auth middleware, **loopback interlock** (agy runs with skip-permissions ⇒ open bind = RCE; refuse non-loopback without `AGI_API_KEY`), GC task. |
| `openai_models.py` | Unchanged; `workdir` + `effort` extensions kept. |
| `errors.py`, `timing.py`, `textfilter.py` | Unchanged (SegmentJoiner + TableFlattener apply identically — agy also splits text at tool boundaries). |
| `routes/health.py`, `routes/compat.py` | Unchanged (family "antigravity"). |
| `translate.py` | `fold_conversation`, `split_system`, SSE builders unchanged. `usage_from_turn` rewritten for agy's usage shape: `input_tokens`→prompt, `output_tokens`→completion, `thinking_tokens` surfaced as non-standard extension; no cache split, no cost_usd. `finish_reason` inferred (agy result has no stop_reason): tool-suspend → `tool_calls`, SUCCESS → `stop`, else error path. |

### Reworked core modules

- **`agy_session.py`** (replaces `claude_session.py`): **process-per-turn**, not
  persistent stdin/stdout. Fresh turn: `agy -p <content> --output-format stream-json
  --dangerously-skip-permissions --print-timeout <request_timeout> [--model <mapped>]
  [--add-dir <workdir>]` with cwd = the conversation's generated workspace dir.
  Continuation turn: same + `--conversation <agy_conv_id>`. Same reader-task +
  event-queue + `STREAM_CLOSED` sentinel, defensive line parser, stderr drain
  (glog noise), staged aclose (terminate → wait 5s → kill).
- **`events.py`**: parses `init`/`step_update`/`result`.
  `agent_response.text_delta` → `TextDelta`; `tool` steps with
  `tool_name == "call_mcp_tool"` and `Arguments.ServerName == "hermes"` →
  hermes tool call; every other tool step → `ToolBoundary` (internal). `result`
  → `TurnDone` (status SUCCESS) or `Error`. Total, never-raising parser.
- **`mcp_bridge.py`**: PendingCall + Future + ContextVar dispatcher **unchanged**.
  Connection direction changes: per-conversation generated workspace dir containing
  `.agents/mcp_config.json` pointing at our loopback bridge
  (`http://127.0.0.1:<port>/mcp/<conv_id>` natively if agy supports http servers,
  else via `mcp-remote`/tiny stdio→http proxy — spike 2 decides).
- **`conversation.py`**: state machine (RUNNING/SUSPENDED/CLOSED), `_pending_index`,
  atomic resume-claim (TOCTOU guard), partial-continuation fail-all, GC — all kept.
  New: `conv_id ↔ agy conversation_id` mapping (captured from the `init` event);
  per-conversation workspace dir lifecycle (created on first turn, removed by GC/close).
- **`config.py`**: `AGI_*` prefix. `agy_bin`, `default_model`, model alias map
  (OpenAI-friendly ids → agy `--model` strings; exact accepted format = spike 4),
  effort → model-variant suffix selection, `system_prompt_injection` (see below),
  same timeout/TTL/GC knobs, `flatten_markdown_tables`, `timing_log`.
- **System prompt (bare-mode substitute):** no CLI flag exists. The request's
  system message is injected as an instruction preamble at the top of the folded
  first turn. Documented deviation: Antigravity's own identity + builtin tools
  remain active. `AGI_SYSTEM_PROMPT_INJECTION=true` default.
- **Warm pool:** omitted in v1 (shipped dark in hermes-claude too). agy's backend
  daemon is persistent; bench decides later whether proc-per-turn cold cost matters.

### Suspend/resume (the one architectural risk)

Primary design — same as hermes-claude: when the model calls
`call_mcp_tool(ServerName="hermes", …)`, our bridge parks the call on a Future,
the server returns `tool_calls` to the OpenAI client, conversation → SUSPENDED;
the next request's tool results resolve the Futures and the **same agy process**
(still blocked inside the MCP call) resumes.

**Fallback** (if spike 1 shows agy won't block long in call_mcp_tool): the bridge
answers the MCP call immediately with a sentinel ("result pending"), lets the turn
end, and the continuation re-invokes `agy --conversation <id>` delivering the tool
result as the next turn's content. Costlier (extra turn) but functionally identical
to the client.

## Phase 0 spikes (gate the build)

1. **call_mcp_tool blocking**: slow MCP server (90s+ delay), `--print-timeout` raised —
   does the tool step stay ACTIVE and complete? → picks primary vs fallback design.
2. **HTTP MCP servers**: does `.agents/mcp_config.json` accept `{"type":"http","url":…}`
   (or `httpUrl`/`serverUrl`) natively, or stdio-only (mcp-remote shim)?
3. **Fresh-server schema cache**: does a never-seen MCP server's tool schema populate
   `~/.gemini/antigravity-cli/mcp/<server>/` on first run without manual steps?
4. **`--model` accepted identifiers**: display names vs slugs; build the alias map.
5. **Error shapes**: result `status` values on failure; behavior on quota exhaustion.
6. **Parallel tool calls**: does agy batch multiple call_mcp_tool in one step?
7. **`-i` mode**: viable as a persistent-session alternative? (curiosity, not blocking)

Spike results are written back to the Claude memory file
`antigravity-cli-verified-capabilities-2026-07.md`.

## Phases

- **Phase 0** — spikes above.
- **Phase 1** — autonomous path (no tools): chat stream + non-stream, models route,
  compat shims, auth, config, textfilter; unit tests (fake agy subprocess) + live
  e2e script.
- **Phase 2** — tool passthrough: bridge + suspend/resume (or fallback), conversation
  manager, GC; unit + live tool-loop tests.
- **Phase 3** — polish: usage passthrough, bench.py, README, .env.example.

## Security

Identical posture to hermes-claude: default bind 127.0.0.1; refuse non-loopback
bind without `AGI_API_KEY`; bearer auth on `/v1`; MCP mount loopback-only and
token-exempt; workdir override validated against `AGI_ALLOWED_WORKDIR_ROOTS`.

## Testing

Mirror hermes-claude's test suite: test_auth, test_agy_session, test_compat,
test_conversation, test_events, test_textfilter, test_translate — pytest
asyncio_mode=auto, no CLI needed (fake subprocess). Live scripts under
tests/scripts/ (e2e_autonomous, e2e_tool, smoke_session).
