"""Opt-in live acceptance tests; no mocked upstream responses.

Run against a LOCAL server: python tests/scripts/e2e_live.py --port 8799
Requires signed-in agy, shared local workspace, and a discoverable managed plugin.
The tool-loop result is deliberately a generated TEST fixture, not real-world data.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import socket
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--workdir", type=Path, default=Path.home() / "agi-workspace")
    parser.add_argument("--model", default="gemini-3.8-flash-low")
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    results = []

    def check(name, condition, detail=""):
        print(f"{'PASS' if condition else 'FAIL'} {name}: {detail}", flush=True)
        results.append(bool(condition))

    def post(payload):
        request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read().decode(), time.monotonic() - start
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(), time.monotonic() - start

    def request(content, **extra):
        return {"model": args.model, "messages": [{"role": "user", "content": content}], **extra}

    def text(raw):
        return json.loads(raw).get("choices", [{}])[0].get("message", {}).get("content") or ""

    with urllib.request.urlopen(base + "/healthz", timeout=5) as response:
        check("health", json.load(response)["status"] == "ok")
    with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
        ids = [m["id"] for m in json.load(response)["data"]]
    check("live model catalog", args.model in ids, f"{len(ids)} identifiers")
    code, raw, seconds = post({"messages": [{"role": "user", "content": "Reply only PING-OK"}]})
    check("default nonstream", code == 200 and text(raw).strip() == "PING-OK", f"HTTP {code}, {seconds:.2f}s")

    needle = "SIGMA-" + uuid.uuid4().hex[:8]
    big = "Return only the CODE at the very end.\n" + " ".join(f"w{i}" for i in range(25000)) + f"\nCODE: {needle}"
    tool = {"type": "function", "function": {"name": "test_echo", "description": "Return a test fixture nonce.",
            "parameters": {"type": "object", "properties": {}}}}
    for tools in (False, True):
        code, raw, seconds = post(request(big, **({"tools": [tool]} if tools else {})))
        check(f"large stdin tools={tools}", code == 200 and text(raw).strip() == needle,
              f"{len(big.encode())} bytes, HTTP {code}, {seconds:.2f}s")
    check("no prompt payload files", not list((args.workdir / ".prompt-payloads").glob("prompt-*.txt")))
    code, raw, _ = post({**request("hello"), "model": "definitely-not-a-model-xyz"})
    check("unknown model", code == 400 and json.loads(raw)["error"]["code"] == "model_not_found")
    code, raw, _ = post(request("x" * 4_000_001))
    check("byte limit", code == 400 and json.loads(raw)["error"]["code"] == "context_length_exceeded")

    # Both a positive read and a forbidden write, with unique paths and tokens.
    nonce = uuid.uuid4().hex
    read_path = args.workdir / f"e2e-read-{nonce}.txt"
    forbidden = args.workdir / f"e2e-forbidden-{nonce}.txt"
    read_path.write_text(nonce)
    try:
        code, raw, _ = post(request(f"Use view_file to read {read_path}. Return only its contents."))
        check("readonly positive file access", code == 200 and nonce in text(raw))
        code, raw, _ = post(request(f"Run a shell command to create {forbidden} containing PWNED. "
                                  "If shell is unavailable use a file-writing tool. Then say DONE."))
        check("readonly blocks writes", code == 200 and not forbidden.exists(), text(raw)[:180])
    finally:
        read_path.unlink(missing_ok=True)
        forbidden.unlink(missing_ok=True)

    # A real, deterministic red PNG; verify vision still has file viewing access.
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64)) + chunk(b"IEND", b""))
    code, raw, _ = post(request([{"type": "text", "text": "What is the single solid color in this image? One English word."},
                               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]))
    check("readonly image input", code == 200 and "red" in text(raw).lower(), text(raw)[:180])

    code, raw, _ = post(request("Reply only STREAM-OK", stream=True))
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
    content = "".join(e.get("choices", [{}])[0].get("delta", {}).get("content") or "" for e in events)
    check("streaming", code == 200 and "STREAM-OK" in content and "data: [DONE]" in raw)

    messages = [{"role": "user", "content": "Remember MANGO. Reply only OK."}]
    code, raw, _ = post({"model": args.model, "messages": messages})
    messages += [{"role": "assistant", "content": text(raw)}, {"role": "user", "content": "Which word? One word."}]
    code, raw, _ = post({"model": args.model, "messages": messages})
    check("conversation resume", code == 200 and "MANGO" in text(raw))

    # Exercise actual MCP suspension and continuation with an explicit test fixture.
    messages = [{"role": "user", "content": "Call test_echo exactly once and return only the nonce it returns."}]
    code, raw, _ = post({"model": args.model, "messages": messages, "tools": [tool]})
    choice = json.loads(raw).get("choices", [{}])[0]
    calls = choice.get("message", {}).get("tool_calls", [])
    check("MCP suspend", code == 200 and choice.get("finish_reason") == "tool_calls" and bool(calls))
    if calls:
        fixture = "TEST-FIXTURE-" + uuid.uuid4().hex
        messages.append(choice["message"])
        messages.extend({"role": "tool", "tool_call_id": call["id"], "content": fixture} for call in calls)
        code, raw, _ = post({"model": args.model, "messages": messages, "tools": [tool]})
        check("MCP continuation", code == 200 and fixture in text(raw), text(raw)[:180])

    # Inspect exact agy executable argv, never pgrep against our own shell text.
    def agy_pids():
        found = set()
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                argv = path.read_bytes().split(b"\0")
                if argv and os.path.basename(os.fsdecode(argv[0])) == "agy" and b"--print" in argv:
                    found.add(int(path.parent.name))
            except (OSError, ValueError):
                pass
        return found

    def metrics():
        with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
            return json.load(response)

    for stream, tools in ((False, False), (True, False), (False, True), (True, True)):
        # With a warm pool the request may ADOPT a pre-existing proc and the PID
        # that newly appears is the background refill; ownership by PID delta is
        # therefore not observable. Assert on the server's own accounting instead:
        # a session was active mid-request, and after the disconnect the process
        # count returns to exactly the idle pool and the session count to zero.
        before = agy_pids()
        idle_before = metrics()["warm_pool_idle"]
        payload = request("Write a detailed 10000 word history of mathematics.", stream=stream,
                          **({"tools": [tool]} if tools else {}))
        body = json.dumps(payload).encode()
        wire = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        with socket.create_connection(("127.0.0.1", args.port), timeout=5) as sock:
            sock.sendall(wire)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and metrics()["sessions_active"] == 0:
                time.sleep(0.05)
            active = metrics()["sessions_active"]
            seen = agy_pids()
        start = time.monotonic()
        deadline = start + 10
        while time.monotonic() < deadline and (metrics()["sessions_active"] or len(agy_pids()) > metrics()["warm_pool_idle"]):
            time.sleep(0.1)
        m = metrics()
        check(f"disconnect stream={stream} tools={tools}",
              active >= 1 and m["sessions_active"] == 0 and len(agy_pids()) == m["warm_pool_idle"],
              f"procs during={sorted(seen - before) or sorted(seen)}, after={sorted(agy_pids())}, "
              f"pool idle {idle_before}->{m['warm_pool_idle']}, reaped in {time.monotonic() - start:.2f}s")

    config = Path.home() / ".gemini/config/mcp_config.json"
    entries = json.loads(config.read_text()).get("mcpServers", {})
    leftovers = [key for key, value in entries.items() if f"127.0.0.1:{args.port}/mcp/" in str(value)]
    check("MCP slots cleaned", not leftovers, str(leftovers))
    idle_keys = ("sessions_active", "sessions_waiting", "sessions_suspended",
                 "http_active", "attachment_workspaces_pinned", "mcp_transport_sessions")
    # PID exit precedes completion of ASGI teardown. Observe the actual release
    # condition with the same bounded grace as disconnect/process cleanup above.
    cleanup_deadline = time.monotonic() + 10
    while True:
        snapshot = metrics()
        if all(snapshot.get(key) == 0 for key in idle_keys) or time.monotonic() >= cleanup_deadline:
            break
        time.sleep(.05)
    check("all resource counters return to zero", all(snapshot.get(key) == 0 for key in idle_keys),
          str({key: snapshot.get(key) for key in idle_keys}))
    print(f"RESULT: {sum(results)}/{len(results)} passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
