"""Live end-to-end tool loop against a running antigravity-interface server.

Acts as an OpenAI client with one function (get_weather): sends the request,
expects a tool_calls response, "runs" the tool, sends the result back, and
expects a final answer that uses it.

    .venv/Scripts/python tests/scripts/e2e_tool.py [--base http://127.0.0.1:8788]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

TOOLS = [{
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
}]


def post(base: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8788")
    ap.add_argument("--model", default="gemini-3.5-flash")
    args = ap.parse_args()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": "What's the weather in Budapest right now? Use the get_weather tool."},
    ]

    print("=== turn 1: expecting tool_calls ===")
    r1 = post(args.base, {"model": args.model, "messages": messages, "tools": TOOLS})
    choice = r1["choices"][0]
    print("finish_reason:", choice["finish_reason"])
    if choice["finish_reason"] != "tool_calls":
        print("FAIL: expected tool_calls, got:", json.dumps(choice, indent=2)[:800])
        return 1
    calls = choice["message"]["tool_calls"]
    print("tool_calls:", json.dumps(calls, indent=2))
    call = calls[0]
    fn_args = json.loads(call["function"]["arguments"] or "{}")
    assert call["function"]["name"] == "get_weather", call
    print("args:", fn_args)

    print("=== turn 2: returning tool result ===")
    messages.append({"role": "assistant", "content": choice["message"].get("content"),
                     "tool_calls": calls})
    messages.append({"role": "tool", "tool_call_id": call["id"],
                     "content": "sunny, 28 degrees Celsius, light breeze"})
    r2 = post(args.base, {"model": args.model, "messages": messages, "tools": TOOLS})
    choice2 = r2["choices"][0]
    print("finish_reason:", choice2["finish_reason"])
    content = choice2["message"]["content"] or ""
    print("final:", content[:400])
    if "28" not in content and "sunny" not in content.lower():
        print("FAIL: final answer does not reflect the tool result")
        return 1
    print("usage:", r2.get("usage"))
    print("OK: full tool loop completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
