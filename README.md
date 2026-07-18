# antigravity-interface

Use **Google Antigravity** from any OpenAI-compatible client.

A small HTTP server that makes the `agy` CLI (Antigravity's terminal agent) look
like an OpenAI API, the way LM Studio or Ollama do. Point any OpenAI client —
hermes, the OpenAI Python SDK, Open WebUI, etc. — at it and you get Antigravity's
agent harness, using your existing Google-account sign-in shared with the
Antigravity IDE (no API key, no per-token billing).

Under the hood it drives `agy -p … --output-format stream-json`, one subprocess
per turn; conversation history lives server-side in agy and follow-up turns
resume it via `--conversation <id>`.

Sibling project: [hermes-claude / claude-code-interface](../hermes-claude) — the
same idea wrapping the `claude` CLI. This repo mirrors its architecture, adapted
to agy's differences (documented throughout the code).

## What it does

- **OpenAI surface:** `GET /v1/models` and `POST /v1/chat/completions`
  (streaming + non-streaming).
- **OpenAI function calling:** your client's `tools` are exposed to agy via an
  in-process MCP bridge. Each tool-turn conversation registers its own entry
  (`hermes`, `hermes2`, …) in the shared Antigravity MCP config pointing back at
  this server; the model calls your functions through agy's `call_mcp_tool`,
  the server returns a normal `tool_calls` response, your client runs the tool
  and sends the result back, and the same agy subprocess — blocked inside the
  MCP call — resumes. The whole multi-step tool loop is one conversation.
- **Antigravity's own built-in tools** (file ops, shell, browser automation,
  subagents, …) run internally the whole time.
- **Conversation cache:** follow-up requests whose history matches a previous
  turn resume the stored agy conversation and send only the newest user message
  — cheaper, and immune to the prompt-length cap below.
- **System prompt injection:** agy has no system-prompt flag, so the request's
  `system` message rides at the top of the first turn as explicit instructions
  (Antigravity's own identity stays active underneath — a documented deviation
  from hermes-claude's bare-model mode).
- **Markdown table flattening** (default on): rewrites pipe tables to fenced
  ASCII so they render in clients that don't support Markdown tables (Discord).
- **Ollama / llama.cpp compatibility shims** so capability-probing frontends
  (Open WebUI, etc.) accept the server.

## Requirements

- The `agy` CLI installed and signed in (shares auth with the Antigravity
  IDE/app): `irm https://antigravity.google/cli/install.ps1 | iex` on Windows,
  see [the install docs](https://antigravity.google/docs/cli-install) for other
  platforms.
- Python 3.11+.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"       # Windows: .venv\Scripts\python
```

## Run

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8788
# or the console script:
.venv/bin/antigravity-interface
```

Configuration is entirely through `AGI_*` environment variables (or a `.env`
file) — see `.env.example` for the full annotated list.

## Use it

Any OpenAI client, `base_url = http://127.0.0.1:8788/v1`, any `api_key` when
none is configured:

```bash
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"hello"}]}'
```

From **hermes**, add a provider in `~/.hermes/config.yaml`:

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

### Models

`agy --model` accepts **exact display names** only; the server maps
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

## Security

The server drives agy with `--dangerously-skip-permissions`, which means
**anyone who can reach the port can run arbitrary code as your user.** Two
interlocks guard this (identical posture to hermes-claude):

1. **Non-loopback bind requires an API key** — `create_app()` refuses to start
   otherwise.
2. **Bearer auth on `/v1`** when `AGI_API_KEY` is set (constant-time compared).
   The `/mcp` mount is intentionally exempt: it is reached only by the local agy
   subprocess over loopback.

**Recommended:** keep the bind on `127.0.0.1`.

## The shared MCP config

The only MCP configuration agy honors (verified against 1.1.4) is
`~/.gemini/config/mcp_config.json` — the same file the Antigravity IDE manages,
including your own servers and their secrets. For tool calling this server
merges per-conversation entries into it and removes them at conversation end:

- foreign entries are **never modified**, writes are atomic;
- leftover `hermes*` entries from a crashed run are cleaned at startup
  (matched by our port);
- set `AGI_MANAGE_MCP_CONFIG=false` to forbid touching the file entirely
  (tool calling then returns 400; plain chat is unaffected).

## Endpoints

**OpenAI**
- `POST /v1/chat/completions` — chat (streaming + non-streaming, tool calling)
- `GET /v1/models` — model list

**Health / info**
- `GET /healthz`, `GET /`

**Ollama / llama.cpp compatibility**
- `GET /api/tags`, `POST /api/show`, `GET /api/version`, `GET /version`,
  `GET /v1/props`, `GET /props`, `GET /api/v1/models`

**Internal**
- `/mcp/<conv_id>` — in-process MCP bridge, dialed only by the spawned agy over
  loopback

## Known limitations

- **Prompt length:** agy takes the prompt as an argv argument (stdin is not
  supported), so a fresh fold is capped at `AGI_MAX_PROMPT_CHARS` (default
  28,000 — Windows' command-line limit minus headroom). The conversation cache
  makes this mostly irrelevant for ongoing chats; a brand-new request with a
  huge pasted history is rejected with `context_length_exceeded`.
- **No bare-model mode:** Antigravity's identity, built-in tools, and "Summary
  of Work" style stay active; the injected system message layers on top rather
  than replacing anything.
- **finish_reason** is inferred (`stop` on success, `tool_calls` on suspension);
  agy reports no stop-reason and no cost field. `usage` carries
  `prompt/completion/total_tokens` plus `completion_tokens_details.reasoning_tokens`.

## Test

```bash
.venv/bin/python -m pytest -q                    # unit tests (no CLI needed)
.venv/bin/python tests/scripts/e2e_tool.py       # live: full tool loop, needs a running server
```

### Benchmark

`scripts/bench.py` launches its **own** throwaway uvicorn instance (never
touches a running server), fires autonomous / tool / continuation turns with
`AGI_TIMING_LOG=1`, and writes p50/p95 of TTFT / total / throughput to JSON:

```bash
.venv/bin/python scripts/bench.py --port 8799 --iters 3 --out bench.json
```

## How it works

```
OpenAI client ──HTTP /v1──▶ antigravity-interface ──argv + stream-json──▶ agy CLI
       ▲                             │                                      │
       └──── tool_calls ─────────────┤        call_mcp_tool(hermesN,…)      │
       │                             │◀────── in-process MCP bridge ◀───────┘
       └──── tool result ───────────▶┘        (/mcp/<conv_id>, loopback)
```

- **Autonomous turn** (no `tools`): the conversation is folded into one prompt
  for a fresh agy conversation (or resumed via the conversation cache); streamed
  `step_update` text deltas become SSE or a single JSON completion.
- **Tool turn:** a conversation owns one agy subprocess + an MCP bridge slot in
  the shared config. When the model calls a bridged function, the subprocess
  blocks inside the MCP call; the server returns a `tool_calls` response and
  parks the conversation `SUSPENDED`. The next request (carrying the tool
  results) resolves the pending futures and the same subprocess resumes.
  Matching is by `tool_call_id`.
- A background GC reaps suspended-too-long and idle conversations (and their
  config entries).
