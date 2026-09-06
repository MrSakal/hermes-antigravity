"""A stand-in for the `agy` CLI used by unit tests.

Mirrors the real transport: the prompt arrives on stdin as one NDJSON line
(``{"event":"user","message":{"role":"user","content":...}}``) selected by
``--print "" --input-format stream-json``. The fake reads it, then emits the
canned stdout stream picked by FAKE_AGY_SCRIPT:
  * "simple"   — init, two text deltas, result SUCCESS
  * "tool"     — init, text, builtin tool step, text, result SUCCESS
  * "error"    — result with status ERROR
  * "noise"    — non-JSON noise then EOF (no result), plus stderr text
  * "echoargs" — emit argv + the stdin prompt as JSON on one line
  * "echoprompt" — result whose response is "len=<chars>;tail=<last 8 chars>"
  * "hang"     — init + text, then sleep (simulates being blocked in MCP)
  * "models"   — (as `agy models` subcommand) print a catalog
Sleeps FAKE_AGY_DELAY seconds (float, default 0) before the result line.
"""
import json
import os
import sys
import time

MODE = os.environ.get("FAKE_AGY_SCRIPT", "simple")
DELAY = float(os.environ.get("FAKE_AGY_DELAY", "0"))
CONV = os.environ.get("FAKE_AGY_CONV_ID", "conv-fake-1")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_stdin_prompt() -> str:
    """Consume the single stream-json user line exactly like agy does."""
    line = sys.stdin.buffer.readline()
    if not line:
        return ""
    try:
        obj = json.loads(line.decode("utf-8"))
    except ValueError:
        return ""
    if obj.get("event") != "user":
        result(status="ERROR", error='stream input message is missing the "event" field')
        sys.exit(0)
    return (obj.get("message") or {}).get("content") or ""


def init_line():
    emit({
        "event": "init",
        "conversation_id": CONV,
        "init": {"cwd": os.getcwd(), "tools": ["call_mcp_tool", "list_dir"],
                 "permission_mode": "always-proceed"},
    })


def text(step, s, state="DONE", usage=None):
    su = {"conversation_id": CONV, "step_index": step, "state": state,
          "step_type": "agent_response", "text_delta": s}
    if usage:
        su["usage"] = usage
    emit({"event": "step_update", "step_update": su})


def result(status="SUCCESS", response="", error=None, usage=None):
    r = {"conversation_id": CONV, "status": status, "response": response,
         "duration_seconds": 0.5, "num_turns": 1,
         "usage": usage or {"input_tokens": 100, "output_tokens": 20,
                            "thinking_tokens": 5, "total_tokens": 120}}
    if error:
        r["error"] = error
    emit({"event": "result", "result": r})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        # `agy models` subcommand: banner + slug\tdisplay lines.
        sys.stdout.write("Fetching available models...\n")
        for slug, disp in [
            ("gemini-9.9-flash-high", "Gemini 9.9 Flash (High)"),
            ("gemini-9.9-flash-medium", "Gemini 9.9 Flash (Medium)"),
            ("gemini-9.9-flash-low", "Gemini 9.9 Flash (Low)"),
            ("claude-sonnet-4-6", "Claude Sonnet 4.6 (Thinking)"),
        ]:
            sys.stdout.write(f"{slug}\t{disp}\n")
        return

    prompt = read_stdin_prompt()

    if MODE == "hang":
        # For suspend/resume tests: emit init + one text delta, then stay alive
        # (as a real agy does while blocked inside our MCP call_tool).
        init_line()
        text(2, "Working ", state="ACTIVE")
        time.sleep(float(os.environ.get("FAKE_AGY_HANG_S", "60")))
        return

    if MODE == "echoargs":
        emit({"argv": sys.argv[1:], "prompt": prompt})
        result(response="argv echoed")
        return

    if MODE == "echoprompt":
        init_line()
        result(response=f"len={len(prompt)};tail={prompt[-8:]}")
        return

    if MODE == "noise":
        sys.stdout.write("jetski: no output produced -- permission denied\n")
        sys.stdout.flush()
        sys.stderr.write("some stderr detail\n")
        sys.stderr.flush()
        return

    init_line()
    if MODE == "error":
        if DELAY:
            time.sleep(DELAY)
        result(status="ERROR", response="", error="invalid --model \"bogus\": not recognized")
        return

    text(2, "Hello ", state="ACTIVE")
    if MODE == "tool":
        emit({"event": "step_update", "step_update": {
            "conversation_id": CONV, "step_index": 3, "state": "ACTIVE",
            "step_type": "tool", "tool_name": "list_dir",
            "tool_info": {"name": "list_dir", "parameters": {"DirectoryPath": "C:/x"}}}})
        emit({"event": "step_update", "step_update": {
            "conversation_id": CONV, "step_index": 3, "state": "DONE",
            "step_type": "tool", "tool_name": "list_dir", "duration_seconds": 0.01,
            "tool_info": {"name": "list_dir", "parameters": {"DirectoryPath": "C:/x"},
                          "output": "a.txt\nb.txt"}}})
    text(4, "world", state="DONE")
    if DELAY:
        time.sleep(DELAY)
    result(response="Hello world")


if __name__ == "__main__":
    main()
