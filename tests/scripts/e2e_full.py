"""Comprehensive LIVE end-to-end test of a running hermes-antigravity server.

Talks to the server over HTTP only (OpenAI-compatible API + compat shims), the
same way the Hermes agent does. Every check is independent and prints
PASS / WARN / FAIL / SKIP with its wall-clock time and a short reason.

    .venv/Scripts/python tests/scripts/e2e_full.py
        [--base http://127.0.0.1:8788] [--api-key KEY]
        [--model gemini-3.8-flash-low]
        [--only health,chat_basic] [--skip image_input,concurrency]
        [--json-report out.json] [--timeout 300]

COST WARNING: this makes real model calls through agy (roughly 40-50 small
turns, several of them concurrent). Prompts are tiny and the default model is a
low-effort fast one, but it is not free. Target runtime is under ~6 minutes.

Checks (names usable with --only / --skip):
  health, chat_basic, chat_stream, models_resolution, system_prompt,
  multi_turn_reuse, long_prompt, image_input, tool_single, tool_parallel,
  tool_rebuild_on_expiry, tool_duplicate, strict_mcp, concurrency,
  disconnect_cleanup, auth

Exit code is 1 if any check FAILs (WARN does not fail the run), else 0.
Model output is nondeterministic, so content checks are case-insensitive
substring matches.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

# ── configuration (set from argv in main) ──────────────────────────────────────


@dataclass
class Cfg:
    base: str = "http://127.0.0.1:8788"
    api_key: Optional[str] = None
    model: str = "gemini-3.8-flash-low"
    timeout: float = 300.0


CFG = Cfg()


# ── outcomes ───────────────────────────────────────────────────────────────────


class CheckFail(Exception):
    pass


class CheckSkip(Exception):
    pass


@dataclass
class Result:
    name: str
    status: str
    secs: float
    reason: str
    details: dict = field(default_factory=dict)


def ensure(cond: Any, reason: str) -> None:
    if not cond:
        raise CheckFail(reason)


def warn(reason: str) -> tuple[str, str]:
    return ("WARN", reason)


# ── HTTP helpers ───────────────────────────────────────────────────────────────


@dataclass
class Resp:
    status: int
    data: Any
    text: str


def _headers(auth: bool, extra: Optional[dict] = None) -> dict:
    h = {"Content-Type": "application/json"}
    if auth and CFG.api_key:
        h["Authorization"] = f"Bearer {CFG.api_key}"
    if extra:
        h.update(extra)
    return h


def http(method: str, path: str, body: Any = None, *, auth: bool = True,
         headers: Optional[dict] = None, timeout: Optional[float] = None) -> Resp:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(CFG.base + path, data=data, method=method,
                                 headers=_headers(auth, headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout or CFG.timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    text = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(text) if text else None
    except ValueError:
        parsed = None
    return Resp(status, parsed, text)


def chat(payload: dict) -> Resp:
    body = {"model": CFG.model, **payload}
    body["stream"] = False
    return http("POST", "/v1/chat/completions", body)


def chat_ok(payload: dict) -> dict:
    r = chat(payload)
    ensure(r.status == 200, f"HTTP {r.status}: {r.text[:300]}")
    ensure(isinstance(r.data, dict) and r.data.get("choices"), f"no choices: {r.text[:300]}")
    return r.data


def content_of(resp: dict) -> str:
    return (resp["choices"][0]["message"].get("content") or "")


@dataclass
class StreamResult:
    status: int
    error_text: str = ""
    chunks: list = field(default_factory=list)
    content: str = ""
    role_seen: bool = False
    content_deltas: int = 0
    finish_reason: Optional[str] = None
    finish_chunk_idx: Optional[int] = None
    usage: Optional[dict] = None
    usage_in_last_chunk: bool = False
    saw_done: bool = False
    ttft: Optional[float] = None
    total: float = 0.0
    tool_calls: list = field(default_factory=list)
    model: Optional[str] = None
    aborted: bool = False


def chat_stream(payload: dict, *, abort_after_first_delta: bool = False) -> StreamResult:
    body = {"model": CFG.model, **payload, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(CFG.base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers=_headers(True, {"Accept": "text/event-stream"}))
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=CFG.timeout)
    except urllib.error.HTTPError as e:
        return StreamResult(status=e.code, error_text=e.read().decode("utf-8", "replace"))
    sr = StreamResult(status=resp.status)
    calls: dict[int, dict] = {}
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                sr.saw_done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            sr.chunks.append(chunk)
            sr.model = sr.model or chunk.get("model")
            if chunk.get("usage"):
                sr.usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("role") == "assistant":
                    sr.role_seen = True
                if delta.get("content"):
                    if sr.ttft is None:
                        sr.ttft = time.monotonic() - t0
                    sr.content += delta["content"]
                    sr.content_deltas += 1
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", len(calls)),
                                            {"id": None, "type": "function",
                                             "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    sr.finish_reason = ch["finish_reason"]
                    sr.finish_chunk_idx = len(sr.chunks) - 1
            if abort_after_first_delta and sr.content_deltas >= 1:
                sr.aborted = True
                break
    finally:
        resp.close()
    sr.total = time.monotonic() - t0
    sr.tool_calls = [calls[i] for i in sorted(calls)]
    if sr.chunks:
        sr.usage_in_last_chunk = bool(sr.chunks[-1].get("usage"))
    return sr


def norm_start(s: str) -> str:
    return s.strip().lstrip("*_`#>\"' \n").upper()


def parallel(fns: list[Callable[[], Any]]) -> list[Any]:
    """Run callables concurrently; return results (or the exception) in order."""
    with ThreadPoolExecutor(max_workers=len(fns)) as ex:
        futs = [ex.submit(f) for f in fns]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001 — surfaced to the caller
                out.append(e)
        return out


_MODEL_IDS: Optional[list[str]] = None


def model_ids() -> list[str]:
    global _MODEL_IDS
    if _MODEL_IDS is None:
        r = http("GET", "/v1/models")
        ensure(r.status == 200, f"/v1/models HTTP {r.status}")
        _MODEL_IDS = [m.get("id", "") for m in (r.data or {}).get("data", [])]
    return _MODEL_IDS


def looks_like_display_name(name: Optional[str]) -> bool:
    return bool(name) and (" " in name or "(" in name)


# ── tools ──────────────────────────────────────────────────────────────────────

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns the current local time in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}
TOOL_RESULTS = {
    "get_weather": "sunny, 28 degrees Celsius, light breeze",
    "get_time": "14:05 local time",
}
TOOL_SYSTEM = "You are a helpful assistant. Use the provided tools when needed."


def tool_loop(messages: list, tools: list, *, stream: bool, max_rounds: int = 4) -> dict:
    """Drive the OpenAI tool loop until a non-tool answer; return a trace."""
    trace = {"rounds": [], "calls": [], "final": "", "batch_sizes": []}
    for _ in range(max_rounds):
        payload = {"messages": messages, "tools": tools}
        if stream:
            sr = chat_stream(payload)
            ensure(sr.status == 200, f"stream HTTP {sr.status}: {sr.error_text[:300]}")
            ensure(sr.saw_done, "stream ended without [DONE]")
            finish, calls, content = sr.finish_reason, sr.tool_calls, sr.content
        else:
            data = chat_ok(payload)
            ch = data["choices"][0]
            finish, calls = ch.get("finish_reason"), ch["message"].get("tool_calls") or []
            content = ch["message"].get("content") or ""
        trace["rounds"].append(finish)
        if finish != "tool_calls":
            trace["final"] = content
            return trace
        ensure(calls, "finish_reason=tool_calls but no tool_calls")
        trace["batch_sizes"].append(len(calls))
        messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            ensure(c.get("id"), f"tool call without id: {c}")
            try:
                json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                raise CheckFail(f"tool args not JSON: {c['function'].get('arguments')!r}")
            trace["calls"].append(name)
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": TOOL_RESULTS.get(name, "unknown tool")})
    raise CheckFail(f"no final answer after {max_rounds} rounds: {trace['rounds']}")


# ── checks ─────────────────────────────────────────────────────────────────────


def check_health():
    r = http("GET", "/healthz", auth=False)
    ensure(r.status == 200 and (r.data or {}).get("status") == "ok", f"/healthz {r.status} {r.text[:200]}")
    ids = model_ids()
    ensure(ids, "/v1/models is empty")
    ensure(any("gemini-4-argon" in i for i in ids), f"no gemini-4-argon family in {ids[:10]}")
    ensure(any(looks_like_display_name(i) for i in ids), "no display names in /v1/models")
    tags = http("GET", "/api/tags")
    ensure(tags.status == 200 and (tags.data or {}).get("models"), f"/api/tags {tags.status}")
    props = http("GET", "/v1/props")
    ensure(props.status == 200 and isinstance(props.data, dict), f"/v1/props {props.status}")
    ver = http("GET", "/api/version")
    ensure(ver.status == 200 and (ver.data or {}).get("version"), f"/api/version {ver.status}")
    return f"{len(ids)} models, version {ver.data['version']}, props.model_path={props.data.get('model_path')!r}"


def check_chat_basic():
    data = chat_ok({"messages": [{"role": "user", "content": "Reply with exactly: PONG-17"}]})
    content = content_of(data)
    ensure("pong-17" in content.lower(), f"content {content[:120]!r}")
    fr = data["choices"][0].get("finish_reason")
    ensure(fr == "stop", f"finish_reason={fr!r}")
    usage = data.get("usage")
    ensure(isinstance(usage, dict) and "total_tokens" in usage, f"usage missing: {usage}")
    model = data.get("model")
    ensure(model, "response has no model")
    if not looks_like_display_name(model):
        return warn(f"model {model!r} does not look like a display name")
    return f"model={model!r}, usage={usage}"


def check_chat_stream():
    sr = chat_stream({"messages": [{"role": "user", "content": "Reply with exactly: STREAM-OK"}]})
    ensure(sr.status == 200, f"HTTP {sr.status}: {sr.error_text[:200]}")
    ensure(sr.role_seen, "no role=assistant chunk")
    ensure(sr.content_deltas >= 1, "no content delta")
    ensure(sr.finish_reason == "stop", f"finish_reason={sr.finish_reason!r}")
    ensure(sr.saw_done, "no [DONE]")
    ensure("stream-ok" in sr.content.lower(), f"content {sr.content[:120]!r}")
    ensure(sr.usage, "no usage in any chunk")
    info = f"{len(sr.chunks)} chunks, {sr.content_deltas} deltas, ttft={sr.ttft:.2f}s"
    if not sr.usage_in_last_chunk:
        return warn(f"usage present but not in the final chunk; {info}")
    return info


def check_models_resolution():
    notes = []
    r1 = chat({"model": "argon", "effort": "low",
               "messages": [{"role": "user", "content": "Reply with exactly: ARGON-OK"}]})
    ensure(r1.status == 200, f"argon+low HTTP {r1.status}: {r1.text[:200]}")
    ensure("argon-ok" in content_of(r1.data).lower(), f"argon content {content_of(r1.data)[:80]!r}")
    notes.append(f"argon+low->{r1.data.get('model')!r}")
    ensure("argon" in (r1.data.get("model") or "").lower(), f"argon resolved to {r1.data.get('model')!r}")

    displays = [i for i in model_ids() if "(" in i]
    pick = next((i for i in displays if "flash" in i.lower() and "low" in i.lower()), None) \
        or next((i for i in displays if "low" in i.lower()), None) \
        or (displays[0] if displays else None)
    if pick:
        r2 = chat({"model": pick, "messages": [{"role": "user", "content": "Reply with exactly: DISPLAY-OK"}]})
        ensure(r2.status == 200, f"display {pick!r} HTTP {r2.status}: {r2.text[:200]}")
        ensure("display-ok" in content_of(r2.data).lower(), f"display content {content_of(r2.data)[:80]!r}")
        notes.append(f"{pick!r}->{r2.data.get('model')!r}")
    else:
        notes.append("no display name to test")

    r3 = chat({"model": "no-such-model-e2e-xyz",
               "messages": [{"role": "user", "content": "Reply with exactly: FALLBACK-OK"}]})
    if r3.status == 200:
        notes.append(f"unknown->200 fallback {r3.data.get('model')!r}")
    elif r3.status == 400 and "model_not_found" in r3.text:
        notes.append("unknown->400 model_not_found (strict)")
    else:
        raise CheckFail(f"unknown model: HTTP {r3.status} {r3.text[:200]}")
    return "; ".join(notes)


def _sys_turn(token: str, word: str) -> str:
    data = chat_ok({"messages": [
        {"role": "system", "content": f"Every reply must start with the exact token {token}."},
        {"role": "user", "content": f"Say the word {word}."},
    ]})
    return content_of(data)


def check_system_prompt():
    a = _sys_turn("ZEBRA-42", "apple")
    ensure(norm_start(a).startswith("ZEBRA-42"), f"first reply {a[:80]!r}")
    b = _sys_turn("OKAPI-7", "banana")
    ensure(norm_start(b).startswith("OKAPI-7"), f"second reply (new prompt) {b[:80]!r}")
    ensure("zebra" not in b.lower(), f"stale system prompt leaked: {b[:80]!r}")
    res = parallel([lambda: _sys_turn("ZEBRA-42", "cherry"), lambda: _sys_turn("OKAPI-7", "plum")])
    for r in res:
        if isinstance(r, Exception):
            raise CheckFail(f"concurrent turn failed: {r}")
    ensure(norm_start(res[0]).startswith("ZEBRA-42"), f"concurrent ZEBRA got {res[0][:80]!r}")
    ensure(norm_start(res[1]).startswith("OKAPI-7"), f"concurrent OKAPI got {res[1][:80]!r}")
    return "sequential + concurrent system prompts isolated"


def check_multi_turn_reuse():
    m1 = [{"role": "user", "content": "Remember the secret word PAPRIKA. Reply with exactly: ok"}]
    s1 = chat_stream({"messages": m1})
    ensure(s1.status == 200 and s1.saw_done, f"turn1 HTTP {s1.status} {s1.error_text[:200]}")
    m2 = m1 + [{"role": "assistant", "content": s1.content},
               {"role": "user", "content": "What was the secret word? Reply with just the word."}]
    s2 = chat_stream({"messages": m2})
    ensure(s2.status == 200 and s2.saw_done, f"turn2 HTTP {s2.status} {s2.error_text[:200]}")
    ensure("paprika" in s2.content.lower(), f"turn2 {s2.content[:120]!r}")
    t1, t2 = s1.ttft or s1.total, s2.ttft or s2.total
    info = f"ttft turn1={t1:.2f}s turn2={t2:.2f}s"
    if t2 >= t1:
        return warn(f"turn2 not faster (no warm reuse?): {info}")
    return info


def check_long_prompt():
    unit = "The quick brown fox jumps over the lazy dog. "
    filler = (unit * (42_000 // len(unit) + 1))[:42_000]
    msg = ("Ignore the filler text below.\n\n" + filler +
           "\n\nEnd of filler. Reply exactly: big-ok")
    ensure(len(msg) > 40_000, "message too short")
    r = chat({"messages": [{"role": "user", "content": msg}]})
    ensure(r.status == 200, f"HTTP {r.status}: {r.text[:300]}")
    c = content_of(r.data)
    ensure("big-ok" in c.lower(), f"content {c[:120]!r}")
    return f"{len(msg)} chars accepted"


def red_png(size: int = 64) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def check_image_input():
    url = "data:image/png;base64," + base64.b64encode(red_png()).decode()
    data = chat_ok({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What color is this image? Answer with one word."},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]})
    c = content_of(data)
    ensure("red" in c.lower(), f"content {c[:120]!r}")
    return f"answer {c.strip()[:40]!r}"


def _weather_msgs() -> list:
    return [{"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": "What's the weather in Budapest right now? Use the get_weather tool."}]


def check_tool_single():
    notes = []
    for stream in (False, True):
        tr = tool_loop(_weather_msgs(), [WEATHER_TOOL], stream=stream)
        label = "stream" if stream else "non-stream"
        ensure("get_weather" in tr["calls"], f"{label}: get_weather never called ({tr['rounds']})")
        fin = tr["final"].lower()
        ensure("28" in fin or "sunny" in fin, f"{label}: final ignores tool result {tr['final'][:120]!r}")
        notes.append(f"{label}: {len(tr['calls'])} call(s)")
    return ", ".join(notes)


def check_tool_parallel():
    msgs = [{"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": "Use the tools to get BOTH the weather in Budapest and "
                                        "the current time in Tokyo, then report both."}]
    tr = tool_loop(msgs, [WEATHER_TOOL, TIME_TOOL], stream=False, max_rounds=5)
    ensure("get_weather" in tr["calls"] and "get_time" in tr["calls"], f"calls={tr['calls']}")
    fin = tr["final"].lower()
    ensure(("28" in fin or "sunny" in fin) and "14" in fin, f"final {tr['final'][:160]!r}")
    mode = "parallel batch" if any(n >= 2 for n in tr["batch_sizes"]) else "sequential"
    return f"{mode}, batches={tr['batch_sizes']}"


def check_tool_rebuild_on_expiry():
    fake_id = f"call_{uuid.uuid4().hex[:24]}"
    msgs = _weather_msgs() + [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": fake_id, "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Budapest"})}}]},
        {"role": "tool", "tool_call_id": fake_id, "content": TOOL_RESULTS["get_weather"]},
    ]
    r = chat({"messages": msgs, "tools": [WEATHER_TOOL]})
    if r.status == 409:
        return f"409 (rebuild disabled): {(r.data or {}).get('error', {}).get('code')}"
    ensure(r.status == 200, f"HTTP {r.status}: {r.text[:300]}")
    ch = r.data["choices"][0]
    if ch.get("finish_reason") == "tool_calls":
        return warn("200 rebuilt, but the model re-issued tool_calls instead of answering")
    c = (ch["message"].get("content") or "").lower()
    ensure("28" in c or "sunny" in c, f"rebuilt answer ignores result {c[:120]!r}")
    return "200 (rebuilt from history)"


def check_tool_duplicate():
    msgs = _weather_msgs()
    data = chat_ok({"messages": msgs, "tools": [WEATHER_TOOL]})
    ch = data["choices"][0]
    ensure(ch.get("finish_reason") == "tool_calls", f"turn1 finish={ch.get('finish_reason')!r}")
    calls = ch["message"]["tool_calls"]
    cont = msgs + [{"role": "assistant", "content": ch["message"].get("content"), "tool_calls": calls}]
    cont += [{"role": "tool", "tool_call_id": c["id"],
              "content": TOOL_RESULTS.get(c["function"]["name"], "n/a")} for c in calls]
    payload = {"messages": cont, "tools": [WEATHER_TOOL]}
    res = parallel([lambda: chat(payload), lambda: chat(payload)])
    for r in res:
        if isinstance(r, Exception):
            raise CheckFail(f"request raised {r}")
    statuses = sorted(r.status for r in res)
    ensure(all(s < 500 for s in statuses), f"server error: {[r.text[:150] for r in res]}")
    if statuses == [200, 409]:
        loser = next(r for r in res if r.status == 409)
        return f"200 + 409 ({(loser.data or {}).get('error', {}).get('code')})"
    if statuses == [200, 200]:
        ok = all(r.data["choices"][0].get("finish_reason") in ("stop", "tool_calls") for r in res)
        ensure(ok, "both 200 but a response is malformed")
        return warn("both 200 (second continuation rebuilt/handled rather than rejected)")
    raise CheckFail(f"unexpected statuses {statuses}: {[r.text[:150] for r in res]}")


def check_strict_mcp():
    data = chat_ok({"tools": [WEATHER_TOOL], "messages": [
        {"role": "system", "content": "Answer questions about your environment honestly."},
        {"role": "user", "content": "Do not call any tools. List the names of every MCP server "
                                    "you can access, comma separated."},
    ]})
    ch = data["choices"][0]
    if ch.get("finish_reason") == "tool_calls":
        return warn("model called a tool instead of listing servers")
    c = (ch["message"].get("content") or "").lower()
    leaks = [s for s in ("sequential-thinking", "sequential_thinking", "sequentialthinking") if s in c]
    if leaks:
        return warn(f"user MCP server leaked: {c[:160]!r}")
    if "hermes" not in c:
        return warn(f"hermes server not named: {c[:160]!r}")
    return f"servers: {c.strip()[:120]!r}"


def _two_turn(word: str) -> str:
    m1 = [{"role": "user", "content": f"Remember the secret word {word}. Reply with exactly: ok"}]
    d1 = chat_ok({"messages": m1})
    m2 = m1 + [{"role": "assistant", "content": content_of(d1)},
               {"role": "user", "content": "What was the secret word? Reply with just the word."}]
    return content_of(chat_ok({"messages": m2}))


def check_concurrency():
    words = ["MANGO", "WALRUS", "COBALT"]
    res = parallel([lambda w=w: _two_turn(w) for w in words])
    problems = []
    for w, r in zip(words, res):
        if isinstance(r, Exception):
            problems.append(f"{w}: {r}")
            continue
        low = r.lower()
        if w.lower() not in low:
            problems.append(f"{w}: got {r[:60]!r}")
        others = [o for o in words if o != w and o.lower() in low]
        if others:
            problems.append(f"{w}: cross-talk {others}")
    ensure(not problems, "; ".join(problems))
    return "3 parallel conversations isolated"


def check_disconnect_cleanup():
    sr = chat_stream({"messages": [{"role": "user", "content":
                                    "Count from 1 to 300, one number per line, nothing else."}]},
                     abort_after_first_delta=True)
    ensure(sr.status == 200, f"stream HTTP {sr.status}: {sr.error_text[:200]}")
    ensure(sr.aborted, "stream finished before we could disconnect")
    t0 = time.monotonic()
    data = chat_ok({"messages": [{"role": "user", "content": "Reply with exactly: ALIVE-3"}]})
    c = content_of(data)
    ensure("alive-3" in c.lower(), f"follow-up {c[:120]!r}")
    return f"disconnected after {sr.total:.1f}s; follow-up answered in {time.monotonic() - t0:.1f}s"


def check_auth():
    if not CFG.api_key:
        raise CheckSkip("no --api-key given")
    r = http("GET", "/v1/models", auth=False)
    ensure(r.status == 401, f"missing bearer -> {r.status}")
    r = http("GET", "/v1/models", auth=False, headers={"Authorization": "Bearer wrong-e2e-key"})
    ensure(r.status == 401, f"wrong bearer -> {r.status}")
    host = urllib.parse.urlparse(CFG.base).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return f"401s ok; /mcp probe skipped (base host {host!r} is not loopback)"
    try:
        m = http("POST", "/mcp/e2e-probe", {}, auth=False, timeout=15)
        mstat = m.status
    except Exception as e:  # noqa: BLE001 — a hanging SSE endpoint is still "not 500"
        return f"401s ok; /mcp probe did not answer in 15 s ({type(e).__name__})"
    ensure(mstat != 500, f"/mcp/e2e-probe -> 500: {m.text[:200]}")
    ensure(mstat != 401, "/mcp is not token-exempt (401)")
    return f"401s ok; /mcp/e2e-probe -> {mstat}"


CHECKS: list[tuple[str, Callable[[], Any]]] = [
    ("health", check_health),
    ("chat_basic", check_chat_basic),
    ("chat_stream", check_chat_stream),
    ("models_resolution", check_models_resolution),
    ("system_prompt", check_system_prompt),
    ("multi_turn_reuse", check_multi_turn_reuse),
    ("long_prompt", check_long_prompt),
    ("image_input", check_image_input),
    ("tool_single", check_tool_single),
    ("tool_parallel", check_tool_parallel),
    ("tool_rebuild_on_expiry", check_tool_rebuild_on_expiry),
    ("tool_duplicate", check_tool_duplicate),
    ("strict_mcp", check_strict_mcp),
    ("concurrency", check_concurrency),
    ("disconnect_cleanup", check_disconnect_cleanup),
    ("auth", check_auth),
]


# ── runner ─────────────────────────────────────────────────────────────────────


def run_one(name: str, fn: Callable[[], Any]) -> Result:
    t0 = time.monotonic()
    try:
        out = fn()
        if isinstance(out, tuple):
            status, reason = out
        else:
            status, reason = "PASS", str(out or "")
    except CheckSkip as e:
        status, reason = "SKIP", str(e)
    except CheckFail as e:
        status, reason = "FAIL", str(e)
    except Exception as e:  # noqa: BLE001 — any crash is a FAIL for that check
        status, reason = "FAIL", f"{type(e).__name__}: {e}"
    res = Result(name, status, time.monotonic() - t0, reason)
    print(f"[{status:4}] {name:24} {res.secs:7.1f}s  {reason}", flush=True)
    return res


def _names(s: Optional[str]) -> set[str]:
    return {x.strip() for x in (s or "").split(",") if x.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8788")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default="gemini-3.8-flash-low")
    ap.add_argument("--only", default=None, help="comma-separated check names")
    ap.add_argument("--skip", default=None, help="comma-separated check names")
    ap.add_argument("--json-report", default=None, help="write results as JSON to this path")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    args = ap.parse_args()

    CFG.base, CFG.api_key = args.base.rstrip("/"), args.api_key
    CFG.model, CFG.timeout = args.model, args.timeout

    known = {n for n, _ in CHECKS}
    only, skip = _names(args.only), _names(args.skip)
    unknown = (only | skip) - known
    if unknown:
        print(f"unknown check name(s): {sorted(unknown)}; known: {sorted(known)}", file=sys.stderr)
        return 2

    print(f"hermes-antigravity e2e  base={CFG.base}  model={CFG.model}  "
          f"auth={'on' if CFG.api_key else 'off'}\n", flush=True)
    started = time.time()
    results: list[Result] = []
    for name, fn in CHECKS:
        if (only and name not in only) or name in skip:
            results.append(Result(name, "SKIP", 0.0, "not selected"))
            print(f"[SKIP] {name:24} {0.0:7.1f}s  not selected", flush=True)
            continue
        results.append(run_one(name, fn))
    elapsed = time.time() - started

    counts = {s: sum(r.status == s for r in results) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print("\n" + "=" * 78)
    print(f"{'check':24} {'status':6} {'time':>8}  reason")
    print("-" * 78)
    for r in results:
        print(f"{r.name:24} {r.status:6} {r.secs:7.1f}s  {r.reason[:90]}")
    print("-" * 78)
    print("  ".join(f"{k}={v}" for k, v in counts.items()) + f"  total={elapsed:.1f}s")

    if args.json_report:
        report = {"base": CFG.base, "model": CFG.model, "started": started,
                  "elapsed_s": round(elapsed, 2), "counts": counts,
                  "results": [asdict(r) for r in results]}
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.json_report}")

    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
