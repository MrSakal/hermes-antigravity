# Hardening verification

Baseline: `5cf6d78`. Target: the Linux deployment of `agi-server.service` on
`127.0.0.1:8788`. Production was restarted after isolated tests and live acceptance.

## Execution evidence

| Gate | Actual result |
|---|---|
| `.venv/bin/python -m pytest tests/ -q --tb=short` | **250 passed**, one Starlette/httpx deprecation warning |
| Original external audit, using `-c pyproject.toml` | **11 passed** |
| `tests/test_resource_limits.py` | **20 passed** (included in the 250) |
| Real upstream E2E on isolated port 8799 | **21/21 passed** |
| Real upstream E2E on production port 8788, final run | **21/21 passed** |
| `/readyz` after production restart | HTTP 200, CLI available, catalog fresh, 14 models |
| `/v1/models` independent readback | 14 IDs, 14 unique IDs, no display-name duplicates |
| Final E2E resource readback | Active/waiting/suspended sessions, HTTP requests, pinned workspaces and SDK transports all zero |
| `git diff --check` | Clean |
| `uv lock --check`, `uv pip check` | Passed; installed dependencies compatible |

The original regression suite is also preserved as `tests/test_audit_regressions.py`.
Tests under `tests/` use isolated configs/workspaces and controlled upstream fixtures;
the E2E script actually invokes signed-in `agy`. Its nonce and red PNG are explicitly
generated test fixtures, not fabricated external results.

The final production E2E observed disconnect cleanup in 1.08s / 0.01s / 4.08s / 0.01s
for plain nonstream / plain stream / tool nonstream / tool stream respectively.
No test-owned prompt files or MCP config slots remained. The 163,948-byte payload
succeeded through stdin both with and without tools. Readonly positive file/vision
access worked, and attempted writes did not create the forbidden fixture file.

## CPU-only filter benchmark

Command: `.venv/bin/python scripts/bench_filter.py --baseline-ref 5cf6d78`
Three-run medians, milliseconds; every output was checked for exact preservation.

| Characters | Feed granularity | Baseline | Current |
|---:|---|---:|---:|
| 16,000 | Whole chunk | 24.028 | 0.069 |
| 16,000 | One character | 58.115 | 66.464 |
| 64,000 | Whole chunk | 417.393 | 0.111 |
| 64,000 | One character | 566.824 | 200.204 |
| 256,000 | Whole chunk | 7,406.856 | 0.445 |
| 256,000 | One character | 7,741.583 | 966.207 |

These are host-local CPU measurements, not model-latency improvements. Short,
one-character feeds have some added bookkeeping overhead; the pathological long-line
quadratic behavior is removed. The optional transform is disabled by default.

## Review findings and corrections

- Updated old tests that expected duplicate model aliases and reusable stale cache
  prefixes; retained explicit tests that those aliases still resolve as inputs.
- Adjusted a simulated disconnect to occur after subprocess acquisition, so it still
  tests teardown rather than the new early-disconnect admission rejection.
- Fixed omitted-workdir continuation integration (`str` session workdir versus `Path`).
- Reproduced and fixed malformed schema declarations returning HTTP 500: now HTTP 400
  before workspace/session acquisition.
- Reproduced and fixed indefinitely waiting config locks, including shielded rollback.
- Added actual HTTP MCP tests for conversation-bound sessions, malformed handshakes,
  undeclared dispatch and owned SDK transport eviction without disturbing another owner.
- The first production E2E reported 20/21 because it sampled counters immediately after
  PID exit, before the ASGI finally block had finished. Logs and readback showed release.
  The test now waits on the real zero-counter condition with a fixed 10-second deadline;
  the final full production rerun passed 21/21. This is not an unconditional sleep or
  an ignored failure.
- Standalone external pytest invocation needs `-c pyproject.toml`; otherwise pytest
  chooses an external root and does not enable the project's asyncio mode.

Independent agent review was unavailable: native delegation returned OAuth 403, and
the final Codex worker hit its usage limit after writing and testing its files. The
parent performed direct specification/code review and reran all acceptance gates;
worker self-reports were not used as deployment proof.

## Remaining explicit boundaries

See [hardening.md](hardening.md): one serving worker, per-workspace attachment quotas,
POSIX locking, advisory shared-file coordination, loopback-only MCP, and no OS/tenant
sandbox guarantee. Sampling/token-limit fields without CLI equivalents are reported
as ignored. MCP SDK cleanup uses a narrow adapter to the constrained 1.28 series;
retest before upgrading it. No persistent CLI process pool was introduced.
