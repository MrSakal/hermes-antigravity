"""A stand-in for the `agy` CLI used by unit tests.

Reads its behavior from the FAKE_AGY_SCRIPT env var:
  * "simple"  — init, two text deltas, result SUCCESS
  * "tool"    — init, text, builtin tool step, text, result SUCCESS
  * "error"   — result with status ERROR
  * "noise"   — non-JSON noise then EOF (no result), plus stderr text
  * "echoargs"— emit argv as JSON on one line (for arg-construction tests)
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
    if MODE == "hang":
        # For suspend/resume tests: emit init + one text delta, then stay alive
        # (as a real agy does while blocked inside our MCP call_tool).
        init_line()
        text(2, "Working ", state="ACTIVE")
        time.sleep(float(os.environ.get("FAKE_AGY_HANG_S", "60")))
        return

    if MODE == "echoargs":
        emit({"argv": sys.argv[1:]})
        result(response="argv echoed")
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
